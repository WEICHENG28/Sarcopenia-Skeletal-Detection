> **文件性質：原始設計構想（歷史紀錄）**
>
> 本文件記錄專案初期的規劃，**與目前實作已有相當差異**。
> 保留的目的是作為「原始計畫 vs 實際成果」的對照依據。
>
> | 原始規劃 | 實際實作 | 變更原因 |
> |---|---|---|
> | Random Forest × 3 | 手工特徵 + ST-GCN + Shift-GCN 三系統比較 | 教授指定改用 ST-GCN；保留傳統模型作為對照基線 |
> | `.mot` / `.trc` 檔案 | MediaPipe world landmarks JSON 直讀 | 轉檔會抹除代償動作訊號，且造成訓練與推論輸入不一致 |
> | `database.csv` 中介格式 | `dataset_manifest.py` 直接對應標籤與骨架檔 | 中介 CSV 易與來源不同步 |
> | SHAP 解釋 | 關節注意力彙總為臨床部位 | SHAP 不適用於圖神經網路的時空輸入 |
> | 排程重訓（MLOps） | 未實作 | 31 人的樣本量下，重訓的效益低於改善收案品質 |
>
> **現況請見**：[README.md](README.md) · [RESULTS.md](RESULTS.md) ·
> [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) · [docs/API_CONTRACT.md](docs/API_CONTRACT.md)

---

# 系統設計文件 (DESIGN.md)：基於 OpenCap 骨架數據的肌少症偵測與 MLOps 系統

## [cite_start]1. 系統目標與使用情境 [cite: 33]
* **系統目標**：開發一套具備可解釋性（XAI）的醫療 AI 系統，利用 OpenCap 輸出的運動學數據（Kinematics data）以及病患基本生理數值，自動偵測肌少症風險。系統採用「多模型延遲融合 (Late Fusion Multi-Model Ensemble)」架構，並整合 MLOps 自動化資料入庫與排程重訓機制。
* **使用情境**：
  1. **臨床檢測**：病患在前端輸入基本資料（年齡、性別、身高、體重），並自由選擇執行 1~3 種功能性測試（坐到站、行走、半蹲）。
  2. **即時推論與解釋**：伺服器接收資料後，呼叫推論模組，即時回傳「肌少症風險分數」以及「SHAP 關鍵影響特徵」（向醫師解釋判斷依據）。
  3. **自動化進化**：測驗結果自動進入資料庫，待臨床醫師於後台確認真實標籤（Label）後，系統於每週背景排程觸發批次重訓，達到模型持續學習（Continuous Learning）。

## [cite_start]2. 為什麼這個系統「合理需要 Orchestration 模式」 [cite: 38]
本系統絕對無法透過單一 Prompt 完成，原因如下：
1. **多模態與多模型架構**：系統需同時處理結構化資料（Metadata）、時間序列資料（.mot 關節角度）及 3D 空間資料（可選的 .trc 軌跡）。且需訓練 3 個獨立的隨機森林模型並實作動態權重融合，邏輯高度複雜。
2. **醫療領域特殊要求**：需實作類別不平衡處理（`class_weight`）、自定義醫療評估指標（Recall、F1、ROC-AUC），以及導入 SHAP 套件進行模型解釋。
3. **MLOps 生命週期整合**：涵蓋了從資料清洗、特徵萃取、模型訓練、推論 API 到背景入庫機制的完整 Data Pipeline，需要「資料工程師」、「機器學習工程師」與「後端工程師」多重角色的協作。

## [cite_start]3. 功能模組切分 [cite: 34]
1. **Data Pipeline & Feature Extraction 模組**：負責解析 `.mot` 與 `.trc` 檔案，計算通用特徵（ROM、極值、角速度、對稱性）與任務專屬特徵（如步幅、下蹲深度），並結合 BMI 進行 `StandardScaler` 標準化縮放。
2. **Model Training & Evaluation 模組**：讀取 `database.csv` 中已標記（Label=0 或 1）的資料，處理資料不平衡，分別訓練 Sit、Walk、Squat 三個模型，輸出 Recall/AUC 評估報告並序列化儲存（`.pkl`）。
3. **Inference & XAI API 模組**：載入模型，處理病患動態任務組合（Late Fusion 加權計算：坐到站 0.4、行走 0.4、半蹲 0.2），並透過 SHAP 生成決策解釋。
4. **Data Management & Scheduler 模組**：管理新進資料的自動入庫，維護 `database.csv`。

## [cite_start]4. Orchestration 角色設計 [cite: 35]
為了讓 Roo Code 能順利推進，設定以下虛擬角色進行 Orchestration：
* **Planner (系統架構師)**：負責確立專案目錄結構，定義各個 Python 模組之間的介面、傳遞的資料格式（例如 dictionary 結構），以及 MLOps 的執行順序。
* **Data Engineer (資料工程師)**：負責撰寫 `feature_extraction.py` 與 `utils.py`，處理複雜的物理與運動學數學運算。
* **ML Engineer (機器學習工程師)**：負責實作 `train.py` 與 `inference.py`，引入 Random Forest、處理不平衡、設定預設權重，並整合 SHAP 套件。
* **Reviewer (QA 測試員)**：在每個模組寫完後，負責撰寫簡單的 mock data 進行單元測試，檢查程式碼是否符合醫療 AI 的 Recall 標準，並修復 Bug。

## [cite_start]5. 資料流與控制流 [cite: 36]
* **推論控制流 (Inference Flow)**：
  前端表單 + OpenCap 檔案 -> 儲存至 `data/raw/` -> 呼叫 `inference.py` -> 讀取模型 `.pkl` -> 特徵萃取 -> 各模型獨立預測 -> Late Fusion 加權計算 -> 計算 SHAP 值 -> 回傳 JSON 給前端。
* **資料與訓練流 (Data & Training Flow)**：
  推論結束的資料 -> 寫入 `data/database.csv` (Label初始為 -1) -> 醫師後台更新 Label 為 0 或 1 -> 排程器觸發 `train.py` -> 讀取有效資料 -> 重新訓練 3 個模型 -> 覆寫 `models/` 目錄下的 `.pkl` 檔案。

## [cite_start]6. 預期 Roo Code 要如何拆解任務與協作 [cite: 37]
我將指示 Roo Code 啟用 Orchestration 模式，並預期它依照以下階段自主規劃與執行：
* **Phase 1 (架構與環境搭建)**：由 Planner 角色建立專案目錄結構（`data/`, `src/`, `models/`），並建立空的 `database.csv` 格式。
* **Phase 2 (資料與特徵工程)**：由 Data Engineer 角色撰寫 `src/utils.py` 與 `src/feature_extraction.py`，實作處理 `.mot`、`.trc` 與基本資料（計算 BMI）的邏輯。完成後自行撰寫測試腳本驗證數值計算。
* **Phase 3 (模型訓練管線)**：由 ML Engineer 角色撰寫 `src/train.py`，實作讀取 CSV、訓練三個獨立模型、處理資料不平衡、輸出 Recall 報告及儲存模型的邏輯。
* **Phase 4 (推論與解釋模組)**：由 ML Engineer 角色撰寫 `src/inference.py`，實作動態權重融合邏輯與 SHAP 分析。
* **Phase 5 (整合測試與重構)**：由 Reviewer 角色串接所有模組，模擬一次「新資料入庫 -> 推論 -> 批次重訓」的完整生命週期，並修復過程中的整合錯誤。