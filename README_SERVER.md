# 伺服器端開發者說明（Server Integration Guide）

本文件面向後端/平台整合工程師，說明如何以「單鏡頭姿態 JSON」驅動目前推論服務。

---

## 1. 系統架構

目前伺服器側流程為：

1. 單鏡頭 MediaPipe Pose JSON（每幀 33 keypoints）
2. 3D 物理約束轉換（平滑、身高尺度化、地面校正、骨長錨定、足部鎖定）
3. 匯出 `.trc` / `.mot`
4. 1D-CNN（PyTorch）各動作推論（Sit/Walk/Squat）
5. Late Fusion 輸出最終風險機率與 Top-3 解釋特徵

對應實作：
- 轉換模組：`src/bridge_mediapipe.py`
- API CLI 入口：`src/api_inference.py`
- 模型推論：`src/inference.py`

---

## 2. API 呼叫方式（`src/api_inference.py`）

### 2.1 必填參數

`python src/api_inference.py` 的參數定義如下：

- `--user_id`（`str`）：受測者 ID。
- `--age`（`float`）：年齡。
- `--height_cm`（`float`）：身高（公分）。
- `--weight_kg`（`float`）：體重（公斤）。
- `--gender`（`str`）：性別字串（直接傳入模型特徵流程）。

### 2.2 選填參數

- `--mock_when_missing`（flag）：當 `data/raw/` 找不到該使用者任一動作 JSON 時，自動建立 Sit/Walk/Squat 最小 mock JSON 再執行推論。

### 2.3 命令範例

```bash
python src/api_inference.py \
  --user_id P001 \
  --age 68 \
  --height_cm 168 \
  --weight_kg 70 \
  --gender female
```

若要允許找不到資料時使用 mock：

```bash
python src/api_inference.py \
  --user_id P999 \
  --age 70 \
  --height_cm 165 \
  --weight_kg 62 \
  --gender male \
  --mock_when_missing
```

---

## 3. 預期輸入

### 3.1 檔案位置與命名規則

- 目錄：`data/raw/`
- 命名：`[ID]_[Action].json`
- 支援動作：`Sit`、`Walk`、`Squat`

範例：
- `data/raw/P001_Sit.json`
- `data/raw/P001_Walk.json`
- `data/raw/P001_Squat.json`

### 3.2 JSON 格式重點

轉換器要求每幀至少可解析到 33 個 keypoints，且每個 keypoint 需具備 `x/y/z`（可有額外欄位如 `visibility`，會被忽略）。

常見可接受格式：

1. 最外層為 frame list：`[frame0, frame1, ...]`
2. frame 可以是：
   - 33 長度陣列，每個元素是 `{ "x":..., "y":..., "z":... }` 或 `[x, y, z]`
   - 或含 `landmarks` / `pose_landmarks` / `keypoints` 欄位的物件

最小範例（單幀示意，實務上應為多幀）：

```json
[
  [
    {"x": 0.01, "y": 0.02, "z": -0.03},
    {"x": 0.02, "y": 0.01, "z": -0.04}
    // ... 需補到 33 個 keypoints
  ]
]
```

---

## 4. 程式標準輸出（stdout JSON）

`src/api_inference.py` 只輸出單一 JSON 物件到 stdout。

### 4.1 成功範例

```json
{
  "user_id": "P001",
  "risk_probability": 0.4721,
  "top_features": ["seq_f004", "static_bmi", "seq_f017"],
  "used_actions": ["Sit", "Walk", "Squat"],
  "status": "ok",
  "message": "inference_success"
}
```

### 4.2 失敗範例（找不到輸入 JSON）

```json
{
  "user_id": "P001",
  "risk_probability": null,
  "top_features": [],
  "used_actions": [],
  "status": "error",
  "message": "No action JSON found under ..."
}
```

---

## 5. 環境建置與模型檔部署

### 5.1 安裝套件

```bash
python -m pip install -r requirements.txt
```

### 5.2 模型檔位置

請確認下列檔案位於 `models/`：

- `models/phase4_cnn_metadata.json`
- `models/phase4_cnn_sit.pth`
- `models/phase4_cnn_walk.pth`
- `models/phase4_cnn_squat.pth`

`src/inference.py` 會固定從上述路徑載入 metadata 與 checkpoint；缺任何一個檔案都會導致推論失敗。

---

## 6. 伺服器整合建議

- 將 CLI 視為「純函式型」接口：輸入參數 + `data/raw/*.json`，輸出 stdout JSON。
- 建議由上游服務先確保檔名符合 `[ID]_[Action].json`，再呼叫 CLI。
- 若要做 smoke test，可使用 `--mock_when_missing` 驗證推論鏈路是否可用。

