# 推論介面契約 v1

模型端與伺服器端的對接規格。

**適用版本**：`src/predict.py`
**取代**：`src/inference.py`、`src/api_inference.py`（舊架構，勿使用）

---

## 1. 快速開始

```python
from src.predict import predict

result = predict(
    profile={
        "age": 78,
        "height_cm": 152,
        "weight_kg": 48,
        "gender": "female",
    },
    actions={
        "Sit": "data/uploads/P001_Sit.json",
        "Walk": "data/uploads/P001_Walk.json",
        "Squat": "data/uploads/P001_Squat.json",
    },
)

payload = result.to_dict()   # 可直接 json.dumps
```

命令列（供除錯用）：

```bash
python -m src.predict \
    --age 78 --height_cm 152 --weight_kg 48 --gender female \
    --sit data/uploads/P001_Sit.json \
    --walk data/uploads/P001_Walk.json
```

---

## 2. 輸入規格

### 2.1 `profile`（必填）

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `age` | number | ✔ | 歲 |
| `height_cm` | number | ✔ | 公分 |
| `weight_kg` | number | ✔ | 公斤 |
| `gender` | string | — | `"female"` / `"male"`，未提供時以男性處理 |
| `bmi` | number | — | 未提供時由身高體重自動計算 |

缺少必填欄位會拋出 `KeyError`。數值超出合理範圍不會中斷，
但會在 `warnings` 中回報。

### 2.2 `actions`（必填，至少一項）

動作名稱對應骨架 JSON 的檔案路徑：

| 動作名 | 說明 | 融合權重 |
|---|---|---|
| `"Sit"` | 五次起立坐下 | 0.4 |
| `"Walk"` | 向前行走 | 0.4 |
| `"Squat"` | 半蹲（最低點停留 3 秒） | 0.2 |

**可只提供 1~2 項。** 系統採動態分母，僅加總實際提供的動作權重。
動作名稱大小寫需完全相符，未知名稱會拋出 `ValueError`。

### 2.3 骨架 JSON 格式

MediaPipe Pose 的 **world landmarks**，每幀 33 個關節點：

```json
[
  [
    {"x": -0.0617, "y": -0.5830, "z": -0.3519, "visibility": 0.9999},
    ... 共 33 個關節點 ...
  ],
  ... 每幀一個陣列 ...
]
```

| 要求 | 說明 |
|---|---|
| 根層級 | 陣列，每個元素為一幀 |
| 每幀 | 恰好 33 個關節點，順序依 MediaPipe 官方定義 |
| 座標 | **必須使用 `pose_world_landmarks`**，非 `pose_landmarks` |
| `visibility` | 必填，用於判斷關節是否被遮擋 |
| 幀率 | 30 fps |

> **為何必須用 world landmarks**：`pose_landmarks` 是畫面正規化座標，
> 數值會隨拍攝距離與畫面裁切改變，模型會把「拍攝距離」當成特徵學習。
> `pose_world_landmarks` 以髖部中心為原點、單位為公尺，不受此影響。

長度不限——系統會自動重採樣至各動作的固定長度
（Sit 192 幀、Walk 128 幀、Squat 128 幀）。

---

## 3. 輸出規格

```json
{
  "ok": true,
  "risk_probability": 0.7834,
  "risk_level": "高",
  "used_actions": ["Sit", "Walk"],
  "per_action": {
    "Sit": {
      "action": "Sit",
      "performed": true,
      "probability": 0.8121,
      "weight": 0.4,
      "frames": 451,
      "visibility_mean": 0.9492,
      "low_confidence": false,
      "reason": "ok"
    },
    "Squat": {
      "action": "Squat",
      "performed": false,
      "probability": null,
      "weight": 0.2,
      "frames": 0,
      "visibility_mean": 0.0,
      "low_confidence": false,
      "reason": "未提供此動作"
    }
  },
  "region_attention": {
    "軀幹核心": 0.1964,
    "下肢近端": 0.2079,
    "下肢遠端": 0.2069,
    "足踝": 0.1885,
    "上肢": 0.2003
  },
  "warnings": [
    "僅完成 2/3 項動作（缺 Squat），評估結果的不確定性較高"
  ],
  "message": "評估完成"
}
```

### 3.1 欄位說明

| 欄位 | 型別 | 說明 |
|---|---|---|
| `ok` | boolean | 是否成功產出評估 |
| `risk_probability` | number \| null | 0–1 的風險機率；失敗時為 `null` |
| `risk_level` | string | `"低"` / `"中低"` / `"中高"` / `"高"` / `"無法評估"` |
| `used_actions` | string[] | 實際納入計算的動作 |
| `per_action` | object | 各動作的細節，**含未執行者** |
| `region_attention` | object | 模型關注的身體部位佔比，總和為 1 |
| `warnings` | string[] | 需提醒使用者的事項，可能為空陣列 |
| `message` | string | 人可讀的狀態說明 |

### 3.2 風險分級門檻

| 機率範圍 | 分級 |
|---|---|
| ≥ 0.70 | 高 |
| 0.513 – 0.70 | 中高 |
| 0.35 – 0.513 | 中低 |
| < 0.35 | 低 |

0.513 為交叉驗證中 F1 最佳的切點。
**此門檻應由臨床端依可接受的漏診率調整**，目前為統計最佳值而非臨床決策值。

### 3.3 `region_attention` 的正確用法

此欄位表示模型在判斷時**關注哪些身體部位**，數值為佔比（總和 1.0）。

**必須標示為「模型關注區域」，不得呈現為診斷結論。**
本研究沒有部位層級的臨床標註可供驗證，且側面拍攝時被遮擋側的關節座標
為系統推測值，任何單側或左右比較的結論皆無依據。

建議前端文案：「模型主要參考的動作部位」，而非「問題部位」。

---

## 4. 錯誤處理

### 4.1 不會拋出例外的情況（回傳 `ok: false`）

所有動作皆無法使用時：

```json
{
  "ok": false,
  "risk_probability": null,
  "risk_level": "無法評估",
  "used_actions": [],
  "message": "沒有可用的動作資料。Sit：檔案不存在：/path/x.json"
}
```

個別動作失敗的原因會記錄於 `per_action[動作].reason`：

| `reason` | 意義 |
|---|---|
| `"ok"` | 正常 |
| `"未提供此動作"` | 呼叫時未包含 |
| `"檔案不存在：..."` | 路徑錯誤 |
| `"骨架檔解析失敗：..."` | JSON 格式不符（幀數不足、關節數不足等）|

只要**至少一項動作成功**，即回傳 `ok: true` 並附加警告。

### 4.2 會拋出例外的情況

| 例外 | 觸發條件 |
|---|---|
| `KeyError` | `profile` 缺少 `age` / `height_cm` / `weight_kg` |
| `ValueError` | `actions` 含未知的動作名稱 |
| `FileNotFoundError` | 找不到模型檔（`models/` 未初始化）|

建議伺服器端統一以 try/except 包住，回傳 HTTP 400（前兩者）
與 HTTP 500（最後一項）。

### 4.3 常見警告

| 警告 | 建議處理 |
|---|---|
| 「僅完成 N/3 項動作」 | 於報表標註資料完整度 |
| 「關節可見度偏低」 | 提示重新拍攝，或標註結果可信度較低 |
| 「年齡低於模型訓練族群」 | 模型訓練族群為 69–88 歲，年輕受測者結果外推性有限 |
| 「身高／體重超出常見範圍」 | 多為鍵入錯誤，建議請使用者確認 |

---

## 5. 效能與部署

| 項目 | 數值 |
|---|---|
| 單次推論耗時 | 約 0.3–1.5 秒（CPU，視影片長度）|
| 記憶體 | 模型合計 < 1 MB |
| 執行裝置 | CPU 即可，不需 GPU |
| 執行緒安全 | 模型為唯讀，可並行呼叫 |

模型於首次呼叫時載入並快取，後續呼叫不重複讀檔。
若服務為長駐程序，建議啟動時先呼叫一次以完成預載。

### 必要檔案

```
models/
├── stgcn_sit.pth
├── stgcn_walk.pth
├── stgcn_squat.pth
├── stacking.joblib
└── metadata.json
```

由 `python -m src.train_final` 產生。缺少任一檔案會拋出 `FileNotFoundError`。

---

## 6. 模型效能與使用限制

**請勿以單次呼叫的信心值判斷模型準確度。**

正式效能（來自 31 人的交叉驗證，見 [RESULTS.md](../RESULTS.md)）：

| 指標 | 數值 |
|---|---|
| AUC | 0.823 |
| 95% 信賴區間 | [0.64, 0.95] |
| 敏感度（recall） | 0.545–0.750（依門檻） |
| 特異度 | 0.895–1.000 |

### 必須向使用者揭露的限制

1. **樣本規模**：模型以 31 位受試者訓練，屬**可行性研究**等級，
   非經臨床驗證之醫療器材。
2. **統計顯著性**：AUC 0.823 的信賴區間涵蓋「僅依年齡判斷」的 0.794，
   兩者在此樣本量下無法區分。
3. **適用族群**：訓練資料為 69–88 歲長者（女性 20 / 男性 11），
   其他族群的適用性未經驗證。
4. **不可作為診斷依據**：本系統為篩檢輔助工具，
   最終判定應由物理治療師依 AWGS 2019 準則（含握力、步速、肌肉量）為之。

建議前端固定顯示：
> 本評估為篩檢參考，不可取代專業醫療診斷。

---

## 7. 版本與相容性

| 項目 | 現況 |
|---|---|
| 契約版本 | v1 |
| 模型版本 | 見 `models/metadata.json` 的 `trained_at_utc` |
| 破壞性變更 | 將提高契約版本號並另行通知 |

重新訓練模型後，`metadata.json` 的內容可能改變
（如 `target_frames`、`fusion_weights`）。
伺服器端**不應寫死這些數值**，一律由 `predict()` 內部讀取。
