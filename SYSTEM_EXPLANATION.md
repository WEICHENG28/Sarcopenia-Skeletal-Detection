# SYSTEM EXPLANATION（Phase 4 深度學習交接版）

本文件提供交接人員快速理解目前系統（Phase 4）如何由傳統機器學習升級到深度學習，並掌握資料準備、訓練、推論與對接重點。

---

## 1) Phase 4 版本關鍵變更摘要

### 1.1 模型從 RandomForest 升級為 1D-CNN（PyTorch）

- 舊版（Phase 3）：三個動作（Sit/Walk/Squat）各一個 RandomForest。
- 新版（Phase 4）：三個動作各一個 **1D-CNN + static MLP** 二元分類模型。
- 實作位置：[`src/train.py`](src/train.py:1)、[`src/inference.py`](src/inference.py:1)。

升級後優勢：
- 能直接利用時間序列局部形狀與相鄰時間點關係。
- 避免只看統計摘要而遺失時序模式。

### 1.2 模型檔案格式更新

`models/` 在 Phase 4 需納入以下檔案：

- `phase4_cnn_sit.pth`
- `phase4_cnn_walk.pth`
- `phase4_cnn_squat.pth`
- `phase4_cnn_metadata.json`

說明：
- `.pth`：各動作 CNN 權重與輸入規格（`T/C/S`）。
- `metadata.json`：全域張量維度與欄位定義（如 `static_cols`），供訓練與推論對齊。

---

## 2) 核心原理：1D-CNN 如何掃描時間序列特徵

在 [`src/inference.py`](src/inference.py:57) 中，模型輸入的序列形狀為 `[B, T, C]`，會先轉成 `[B, C, T]` 再進 CNN：

1. `Conv1d(c_dim, 32, kernel_size=5, padding=2)`
   - 每個卷積核沿著時間軸滑動，掃描長度為 5 的局部片段。
2. `Conv1d(32, 64, kernel_size=3, padding=1)`
   - 在第一層抽出的模式上再做更高階時序特徵擷取。
3. `AdaptiveAvgPool1d(1)`
   - 把整段時間序列壓縮成固定長度表示，輸出與原始序列長度一致可控。

直覺上可理解為：
- 1D-CNN 像「時間視窗掃描器」，會在每個時間段找出有意義的動作變化模式。
- 比只用平均/最大值更能保留「動作發生順序與局部波形」資訊。

---

## 3) [`src/prepare_data.py`](src/prepare_data.py:1) 的關鍵角色

[`src/prepare_data.py`](src/prepare_data.py:1) 是 **Registry → Tensor 訓練格式** 的核心轉換器，功能如下：

1. 讀取 [`data/raw_registry.csv`](data/raw_registry.csv)
   - 每筆使用者資料包含個人欄位與三個動作路徑（`*.mot` / `*.trc`）。

2. 拆成 long-format（每個使用者 × 每個動作 一列）
   - 透過 [`build_rows()`](src/prepare_data.py:147) 產生 Sit/Walk/Squat 三筆 action sample。

3. 呼叫 [`extract_sequence_features()`](src/feature_extraction.py:127)
   - 將 `.mot`（必要）與 `.trc`（可選）轉成 sequence。
   - 產生 static 特徵（age/height/weight/bmi/gender）。

4. 固定序列長度（pad / truncate）
   - 透過 `target_len` 將每筆樣本統一成固定 `T`，使模型可 batch 訓練。

5. 輸出訓練矩陣 [`data/database.csv`](data/database.csv)
   - static 欄位：`static_*`
   - sequence 欄位：`seq_t###_f###`

這一步是深度學習訓練能否成功的前提：
- 若沒有 [`src/prepare_data.py`](src/prepare_data.py:1) 做欄位對齊與固定長度，[`src/train.py`](src/train.py:119) 無法穩定還原 `(N, T, C)` 張量。

---

## 4) 訓練與推論流程（Phase 4）

### 4.1 訓練

由 [`main()`](src/train.py:385) 啟動流程：

1. 讀 [`data/database.csv`](data/database.csv) 並過濾 `Label ∈ {0,1}`。
2. 依 Action 分開訓練三個模型。
3. 損失函數使用 `BCEWithLogitsLoss(pos_weight=...)` 處理類別不平衡。
4. 輸出 `phase4_cnn_<action>.pth` 與 `phase4_cnn_metadata.json`。

### 4.2 推論

由 [`run_inference()`](src/inference.py:208) 執行：

1. 讀 metadata + 三個 `.pth`。
2. 對每個 performed 的 action 取出 sequence/static 後推論機率。
3. 用 Late Fusion（動態分母）整合最終風險。
4. 輸出 `explanations_top3`（梯度解釋，失敗時 fallback）。

---

## 5) 前後端串接重點（含 `.trc`）

輸入 payload 建議格式（節錄）：

```json
{
  "profile": {
    "age": 34,
    "height_cm": 172,
    "weight_kg": 74
  },
  "actions": {
    "Sit": {
      "performed": true,
      "mot_path": "data/raw/17570_10_yes.mot",
      "trc_path": "data/raw/17570_10_yes.trc"
    },
    "Walk": {
      "performed": true,
      "mot_path": "data/raw/17570_11_yes.mot",
      "trc_path": null
    },
    "Squat": {
      "performed": false,
      "mot_path": "data/raw/17570_12_yes.mot",
      "trc_path": null
    }
  }
}
```

注意事項：
- `.trc` 可選；若提供則會併入時序通道。
- 若 `.trc` 未提供或不存在，系統仍可推論（以 fallback 方式運作）。

---

## 6) 常用指令

在專案根目錄執行：

```bat
python -m src.prepare_data --target-len 128
python -m src.train
python test_inference.py
```

交接建議：每次資料更新時，依序重跑資料準備、訓練、推論驗證，並同步更新 `models/` 中的 `.pth` 與 `metadata.json`。

