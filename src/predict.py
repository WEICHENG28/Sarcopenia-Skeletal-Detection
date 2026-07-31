"""推論服務：從 MediaPipe 骨架 JSON 產出肌少症風險評估。

供伺服器端呼叫。介面規格見 `docs/API_CONTRACT.md`。

設計重點：

1. **直接讀取 MediaPipe JSON**，不再經過 `.trc` / `.mot` 轉檔。
   舊版 `inference.py` 會先轉成 `.trc` 再推論，但訓練時讀的是 JSON，
   兩者座標處理不同，導致訓練與上線的輸入不一致。

2. **動態分母的 late fusion**。受試者僅完成 2/3 項動作時，
   分母只加總實際完成者的權重，仍可產出風險分數。
   （A003 缺 Sit 的實測結果為正確預測。）

3. **部位注意力以臨床區塊呈現**，而非 22 個關節點編號。
   醫師無法解讀「節點 25 最重要」，但能理解「下肢近端」。

用法：
    from src.predict import predict

    result = predict(
        profile={"age": 78, "height_cm": 152, "weight_kg": 48, "gender": "female"},
        actions={
            "Sit": "data/raw/P001_Sit.json",
            "Walk": "data/raw/P001_Walk.json",
        },
    )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import joblib
import numpy as np
import torch

from src.models_stgcn import STGCN
from src.skeleton_io import (
    BODY_NODES, load_landmarks, normalize_sequence, resample_time, to_model_tensor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# 關節點對應的臨床部位。醫師以部位理解，不以關節編號理解。
CLINICAL_REGIONS: Dict[str, Sequence[int]] = {
    "軀幹核心": (11, 12, 23, 24),
    "下肢近端": (23, 24, 25, 26),   # 髖、大腿——肌少症主要影響區
    "下肢遠端": (25, 26, 27, 28),   # 膝、小腿
    "足踝": (27, 28, 29, 30, 31, 32),
    "上肢": (13, 14, 15, 16),
}

# 可見度低於此值代表關節被遮擋，座標為 MediaPipe 推測值
VISIBILITY_THRESHOLD = 0.5

# 低於此比例的可見關節時，該動作的結果標記為低可信度
LOW_CONFIDENCE_VISIBILITY = 0.80


@dataclass
class ActionResult:
    """單一動作的推論結果。"""

    action: str
    performed: bool
    probability: float | None = None
    weight: float = 0.0
    frames: int = 0
    visibility_mean: float = 0.0
    low_confidence: bool = False
    reason: str = "ok"


@dataclass
class PredictionResult:
    """完整的評估結果。"""

    ok: bool
    risk_probability: float | None
    risk_level: str
    used_actions: List[str] = field(default_factory=list)
    per_action: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    region_attention: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _ModelBundle:
    """延遲載入並快取模型，避免每次呼叫都讀檔。"""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self._metadata: Dict[str, Any] | None = None
        self._models: Dict[str, STGCN] = {}
        self._stacking = None

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            path = self.models_dir / "metadata.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"找不到模型設定檔：{path}\n請先執行：python -m src.train_final"
                )
            self._metadata = json.loads(path.read_text(encoding="utf-8"))
        return self._metadata

    def model(self, action: str) -> STGCN:
        if action not in self._models:
            path = self.models_dir / f"stgcn_{action.lower()}.pth"
            if not path.exists():
                raise FileNotFoundError(f"找不到模型權重：{path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = STGCN(
                in_channels=checkpoint["in_channels"],
                static_dim=0,
                num_nodes=checkpoint["num_nodes"],
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self._models[action] = model
        return self._models[action]

    @property
    def stacking(self):
        if self._stacking is None:
            path = self.models_dir / "stacking.joblib"
            if not path.exists():
                raise FileNotFoundError(f"找不到融合層：{path}")
            self._stacking = joblib.load(path)
        return self._stacking


_BUNDLE = _ModelBundle()


def _risk_level(probability: float) -> str:
    """把機率轉成臨床可讀的分級。

    門檻依交叉驗證結果設定：0.513 為 F1 最佳切點。
    低於 0.35 與高於 0.70 分別視為較明確的低／高風險。
    """
    if probability >= 0.70:
        return "高"
    if probability >= 0.513:
        return "中高"
    if probability >= 0.35:
        return "中低"
    return "低"


def _region_attention(
    node_importance: np.ndarray,
    node_indices: Sequence[int],
) -> Dict[str, float]:
    """把關節層級的注意力彙總到臨床部位，並正規化成佔比。"""
    position = {node: i for i, node in enumerate(node_indices)}
    scores: Dict[str, float] = {}
    for region, members in CLINICAL_REGIONS.items():
        values = [node_importance[position[m]] for m in members if m in position]
        scores[region] = float(np.mean(values)) if values else 0.0

    total = sum(scores.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: round(v / total, 4) for k, v in scores.items()}


def _profile_vector(profile: Mapping[str, Any]) -> List[float]:
    """由基本資料組出融合層所需的特徵。"""
    age = float(profile["age"])
    height = float(profile["height_cm"])
    weight = float(profile["weight_kg"])
    gender = str(profile.get("gender", "")).strip().lower()

    height_m = max(height / 100.0, 1e-6)
    bmi = float(profile.get("bmi", weight / (height_m * height_m)))
    return [age, bmi, height, weight, 1.0 if gender.startswith("f") else 0.0]


def _validate_profile(profile: Mapping[str, Any]) -> List[str]:
    """檢查基本資料是否在合理範圍，回傳警告訊息。"""
    warnings: List[str] = []
    for key in ("age", "height_cm", "weight_kg"):
        if key not in profile:
            raise KeyError(f"profile 缺少必要欄位：{key}")

    age = float(profile["age"])
    height = float(profile["height_cm"])
    weight = float(profile["weight_kg"])

    if not 40 <= age <= 110:
        warnings.append(f"年齡 {age:.0f} 超出常見範圍（40–110），請確認輸入是否正確")
    if not 120 <= height <= 200:
        warnings.append(f"身高 {height:.0f} cm 超出常見範圍（120–200）")
    if not 25 <= weight <= 150:
        warnings.append(f"體重 {weight:.1f} kg 超出常見範圍（25–150）")

    # 模型的訓練族群為 69–88 歲長者，超出此範圍的推論外推性有限
    if age < 65:
        warnings.append(
            f"年齡 {age:.0f} 低於模型訓練族群（69–88 歲），評估結果外推性有限"
        )
    return warnings


def _infer_action(
    action: str,
    json_path: Path,
    bundle: _ModelBundle,
) -> tuple[ActionResult, np.ndarray | None]:
    """對單一動作推論，回傳結果與關節注意力。"""
    metadata = bundle.metadata
    weight = metadata["fusion_weights"][action]

    if not json_path.exists():
        return ActionResult(action, False, weight=weight,
                            reason=f"檔案不存在：{json_path}"), None

    try:
        raw = load_landmarks(json_path)
    except Exception as exc:
        return ActionResult(action, False, weight=weight,
                            reason=f"骨架檔解析失敗：{exc}"), None

    visibility = float(raw[..., 3].mean())
    normalized = normalize_sequence(raw)
    resampled = resample_time(normalized, metadata["target_frames"][action])
    tensor = to_model_tensor(resampled)

    model = bundle.model(action)
    with torch.no_grad():
        logits, importance = model(
            torch.from_numpy(tensor).unsqueeze(0), return_importance=True
        )
        probability = float(torch.sigmoid(logits)[0].item())

    return (
        ActionResult(
            action=action,
            performed=True,
            probability=probability,
            weight=weight,
            frames=int(raw.shape[0]),
            visibility_mean=round(visibility, 4),
            low_confidence=visibility < LOW_CONFIDENCE_VISIBILITY,
        ),
        importance["node_importance"][0].numpy(),
    )


def predict(
    profile: Mapping[str, Any],
    actions: Mapping[str, str | Path],
    *,
    bundle: _ModelBundle | None = None,
) -> PredictionResult:
    """執行肌少症風險評估。

    Args:
        profile: 受測者基本資料，需含 ``age``、``height_cm``、``weight_kg``；
            ``gender`` 選填（``"female"`` / ``"male"``）。
        actions: 動作名稱對應骨架 JSON 路徑。可只提供 1~3 項。

    Returns:
        :class:`PredictionResult`，呼叫 ``.to_dict()`` 可直接序列化為 JSON。
    """
    bundle = bundle or _BUNDLE
    metadata = bundle.metadata
    known_actions = metadata["actions"]

    warnings = _validate_profile(profile)

    unknown = [a for a in actions if a not in known_actions]
    if unknown:
        raise ValueError(f"不支援的動作：{unknown}，可用動作為 {known_actions}")

    per_action: Dict[str, ActionResult] = {}
    attentions: Dict[str, np.ndarray] = {}

    for action in known_actions:
        if action not in actions:
            per_action[action] = ActionResult(
                action, False,
                weight=metadata["fusion_weights"][action],
                reason="未提供此動作",
            )
            continue

        result, attention = _infer_action(action, Path(actions[action]), bundle)
        per_action[action] = result
        if attention is not None:
            attentions[action] = attention

    used = [a for a, r in per_action.items() if r.performed]
    if not used:
        reasons = "；".join(
            f"{a}：{r.reason}" for a, r in per_action.items() if r.reason != "未提供此動作"
        )
        return PredictionResult(
            ok=False,
            risk_probability=None,
            risk_level="無法評估",
            per_action={a: asdict(r) for a, r in per_action.items()},
            warnings=warnings,
            message=f"沒有可用的動作資料。{reasons}" if reasons else "未提供任何動作資料。",
        )

    # 動態分母：只加總實際完成的動作權重
    weight_sum = sum(per_action[a].weight for a in used)
    motion_probability = sum(
        per_action[a].weight * per_action[a].probability for a in used
    ) / weight_sum

    # 融合層需要三個動作的機率；未完成者以純骨架加權平均值代入，
    # 使缺項不致被解讀為「風險為零」。
    ordered = sorted(known_actions)
    stacking_row = [
        per_action[a].probability if per_action[a].performed else motion_probability
        for a in ordered
    ] + _profile_vector(profile)
    risk = float(bundle.stacking.predict_proba([stacking_row])[0][1])

    if len(used) < len(known_actions):
        missing = [a for a in known_actions if a not in used]
        warnings.append(
            f"僅完成 {len(used)}/{len(known_actions)} 項動作（缺 {', '.join(missing)}），"
            f"評估結果的不確定性較高"
        )

    low_quality = [a for a in used if per_action[a].low_confidence]
    if low_quality:
        warnings.append(
            f"下列動作的關節可見度偏低，座標多為系統推測值，結果可信度較低："
            f"{', '.join(low_quality)}"
        )

    attention_stack = [
        per_action[a].weight * attentions[a] for a in used if a in attentions
    ]
    region_attention = (
        _region_attention(np.sum(attention_stack, axis=0) / weight_sum, BODY_NODES)
        if attention_stack else {}
    )

    return PredictionResult(
        ok=True,
        risk_probability=round(risk, 4),
        risk_level=_risk_level(risk),
        used_actions=used,
        per_action={a: asdict(r) for a, r in per_action.items()},
        region_attention=region_attention,
        warnings=warnings,
        message="評估完成",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="肌少症風險評估（單筆）")
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--height_cm", type=float, required=True)
    parser.add_argument("--weight_kg", type=float, required=True)
    parser.add_argument("--gender", default="")
    parser.add_argument("--sit", type=Path)
    parser.add_argument("--walk", type=Path)
    parser.add_argument("--squat", type=Path)
    args = parser.parse_args()

    actions = {
        name: path
        for name, path in (("Sit", args.sit), ("Walk", args.walk), ("Squat", args.squat))
        if path is not None
    }
    result = predict(
        profile={
            "age": args.age,
            "height_cm": args.height_cm,
            "weight_kg": args.weight_kg,
            "gender": args.gender,
        },
        actions=actions,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
