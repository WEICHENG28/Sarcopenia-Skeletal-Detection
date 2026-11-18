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

4. **最終風險只依骨架**。人口學 Stacking 的 AUC 雖較高（0.823），
   但主要貢獻來自年齡，敏感度僅 0.545；純骨架 late fusion
   （AUC 0.789、敏感度 0.750）較符合篩檢需求，且結果完全來自動作。
   年齡只用於判斷 AWGS 2019 適用範圍，身高、體重與性別不影響分數。

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

# 平均可見度會被清楚的時段稀釋，因此另外檢查「下半身看不到」的影格比例。
# 訓練資料中被遮擋幀比例最高為行走的 7.7%（見 results/data_health_report.md）。
LOWER_BODY_LANDMARKS = (23, 24, 25, 26, 27, 28)
OCCLUDED_FRAME_VISIBILITY = 0.5
MAX_OCCLUDED_FRAME_RATIO = 0.20

# 少於此影格數視為拍攝失敗。訓練資料中最短的樣本為 93 幀（約 3.1 秒），
# 低於 30 幀（1 秒）不足以涵蓋任何一個完整動作週期。
MIN_USABLE_FRAMES = 30

# 灰色地帶：交叉驗證中此區間的陽性、陰性個案大量重疊，模型無法可靠區分。
# 落在此區間時不給明確分級，改建議由治療師評估。
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.70

# 低於此年齡超出訓練族群（69–88 歲），與 _validate_profile 的警告一致
MIN_SUPPORTED_AGE = 65

# AWGS 2019 的篩檢對象為 60 歲以上長者；未滿此年齡不提供風險判斷
AWGS_MIN_AGE = 60

# 少於此數量的動作時，單一影片的雜訊無法被其他動作平衡
MIN_ACTIONS_FOR_CONFIDENT_RESULT = 2


@dataclass
class ActionResult:
    """單一動作的推論結果。"""

    action: str
    performed: bool
    probability: float | None = None
    weight: float = 0.0
    frames: int = 0
    visibility_mean: float = 0.0
    occluded_ratio: float = 0.0
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
    # 為 True 時前端不應呈現明確的風險分級，而是建議由治療師評估
    uncertain: bool = False
    uncertainty_reasons: List[str] = field(default_factory=list)
    # 為 True 時受測者未達 AWGS 2019 適用年齡，risk_probability 為 None
    age_not_applicable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _ModelBundle:
    """延遲載入並快取模型，避免每次呼叫都讀檔。"""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self._metadata: Dict[str, Any] | None = None
        self._models: Dict[str, STGCN] = {}

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


def _missing_actions_message(used: Sequence[str], known: Sequence[str]) -> str:
    missing = [a for a in known if a not in used]
    return (
        f"僅完成 {len(used)}/{len(known)} 項動作（缺 {', '.join(missing)}），"
        f"評估結果的不確定性較高"
    )


def _low_visibility_message(low_quality: Sequence[str]) -> str:
    return (
        f"下列動作的關節可見度偏低或有一段時間下半身不在畫面內，"
        f"座標多為系統推測值，結果可信度較低：{', '.join(low_quality)}"
    )


def _young_age_message(age: float) -> str:
    return f"年齡 {age:.0f} 低於模型訓練族群（69–88 歲），評估結果外推性有限"


def _below_awgs_age_message(age: float) -> str:
    return (
        f"年齡 {age:.0f} 歲未達 AWGS 2019 適用年齡（{AWGS_MIN_AGE} 歲以上），"
        f"本次不提供風險判斷"
    )


def _age_message(age: float) -> str | None:
    """年齡相關提示只回傳一則，避免同一原因重複出現。"""
    if age < AWGS_MIN_AGE:
        return _below_awgs_age_message(age)
    if age < MIN_SUPPORTED_AGE:
        return _young_age_message(age)
    return None


def _uncertainty_reasons(
    probability: float,
    used: Sequence[str],
    low_quality: Sequence[str],
    profile: Mapping[str, Any],
    known: Sequence[str] = ("Sit", "Walk", "Squat"),
) -> List[str]:
    """列出此次結果不可靠的原因；空串列代表可給出明確分級。

    醫療情境下寧可回報「不確定」，也不要對不可靠的輸入給出明確結論。
    與 warnings 重疊的原因沿用同一段文字，前端可直接去重。
    """
    reasons: List[str] = []
    if UNCERTAIN_LOW <= probability < UNCERTAIN_HIGH:
        reasons.append(
            f"風險機率 {probability:.0%} 落在模型無法可靠區分的區間"
            f"（{UNCERTAIN_LOW:.0%}–{UNCERTAIN_HIGH:.0%}）"
        )
    if low_quality:
        reasons.append(_low_visibility_message(low_quality))
    if len(used) < MIN_ACTIONS_FOR_CONFIDENT_RESULT:
        reasons.append(_missing_actions_message(used, known))
    age_message = _age_message(float(profile["age"]))
    if age_message:
        reasons.append(age_message)
    return reasons


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


def _validate_profile(profile: Mapping[str, Any]) -> List[str]:
    """檢查基本資料是否在合理範圍，回傳警告訊息。"""
    warnings: List[str] = []
    for key in ("age", "height_cm", "weight_kg"):
        if key not in profile:
            raise KeyError(f"profile 缺少必要欄位：{key}")

    age = float(profile["age"])
    height = float(profile["height_cm"])
    weight = float(profile["weight_kg"])

    # 偏低的年齡由下方的 AWGS／訓練族群提示處理，這裡只檢查明顯的輸入錯誤
    if age > 110:
        warnings.append(f"年齡 {age:.0f} 超出常見範圍（40–110），請確認輸入是否正確")
    if not 120 <= height <= 200:
        warnings.append(f"身高 {height:.0f} cm 超出常見範圍（120–200）")
    if not 25 <= weight <= 150:
        warnings.append(f"體重 {weight:.1f} kg 超出常見範圍（25–150）")

    # 模型的訓練族群為 69–88 歲長者，超出此範圍的推論外推性有限
    age_message = _age_message(age)
    if age_message:
        warnings.append(age_message)
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
                            reason=f"找不到骨架檔：{json_path.name}"), None

    try:
        raw = load_landmarks(json_path)
    except Exception as exc:
        # 訊息會直接顯示給治療師或受測者，因此描述「該怎麼做」而非技術細節，
        # 且不揭露完整檔案路徑。
        text = str(exc)
        if "非空" in text or "empty" in text.lower():
            reason = "影片中未偵測到人體，請確認人物完整入鏡且光線充足後重新拍攝"
        elif "關節數不足" in text:
            reason = "影片中人體偵測不完整，請確認全身入鏡後重新拍攝"
        else:
            reason = f"骨架檔無法解析（{json_path.name}）"
        return ActionResult(action, False, weight=weight, reason=reason), None

    if raw.shape[0] < MIN_USABLE_FRAMES:
        return ActionResult(
            action, False, weight=weight, frames=int(raw.shape[0]),
            reason=(
                f"有效影格僅 {raw.shape[0]} 幀（至少需 {MIN_USABLE_FRAMES} 幀），"
                "請確認人物全程入鏡後重新拍攝"
            ),
        ), None

    visibility = float(raw[..., 3].mean())
    lower_body_visibility = raw[:, LOWER_BODY_LANDMARKS, 3].mean(axis=1)
    occluded_ratio = float((lower_body_visibility < OCCLUDED_FRAME_VISIBILITY).mean())
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
            occluded_ratio=round(occluded_ratio, 4),
            low_confidence=(
                visibility < LOW_CONFIDENCE_VISIBILITY
                or occluded_ratio > MAX_OCCLUDED_FRAME_RATIO
            ),
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

    # 動態分母：只加總實際完成的動作權重。最終風險即為此純骨架加權平均。
    weight_sum = sum(per_action[a].weight for a in used)
    risk = sum(
        per_action[a].weight * per_action[a].probability for a in used
    ) / weight_sum

    if len(used) < len(known_actions):
        warnings.append(_missing_actions_message(used, known_actions))

    low_quality = [a for a in used if per_action[a].low_confidence]
    if low_quality:
        warnings.append(_low_visibility_message(low_quality))

    uncertainty_reasons = _uncertainty_reasons(
        risk, used, low_quality, profile, known_actions
    )

    attention_stack = [
        per_action[a].weight * attentions[a] for a in used if a in attentions
    ]
    region_attention = (
        _region_attention(np.sum(attention_stack, axis=0) / weight_sum, BODY_NODES)
        if attention_stack else {}
    )

    if float(profile["age"]) < AWGS_MIN_AGE:
        # 仍保留各動作結果供工作人員檢查錄影品質，但不輸出整體風險
        return PredictionResult(
            ok=True,
            risk_probability=None,
            risk_level="不適用",
            used_actions=used,
            per_action={a: asdict(r) for a, r in per_action.items()},
            region_attention=region_attention,
            warnings=warnings,
            message="未達 AWGS 2019 適用年齡，不提供風險判斷",
            uncertain=True,
            uncertainty_reasons=uncertainty_reasons,
            age_not_applicable=True,
        )

    return PredictionResult(
        ok=True,
        risk_probability=round(risk, 4),
        risk_level=_risk_level(risk),
        used_actions=used,
        per_action={a: asdict(r) for a, r in per_action.items()},
        region_attention=region_attention,
        warnings=warnings,
        message="評估完成，但結果不確定" if uncertainty_reasons else "評估完成",
        uncertain=bool(uncertainty_reasons),
        uncertainty_reasons=uncertainty_reasons,
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
