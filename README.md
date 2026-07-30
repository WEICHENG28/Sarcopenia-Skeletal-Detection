# 肌少症骨架偵測系統 — 模型端

從 MediaPipe 骨架動作資料判斷受試者是否有肌少症風險，並提供可解釋的判斷依據。

## 文件導覽

| 文件 | 內容 | 給誰看 |
|---|---|---|
| **README.md**（本檔） | 專案總覽、環境設定、如何執行 | 所有人 |
| [RESULTS.md](RESULTS.md) | 三系統比較的實驗數據 | 想看模型表現 |
| [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | 開發歷程與**決策理由** | 接手/回顧/寫報告 |
| [DESIGN.md](DESIGN.md) | 原始系統設計構想 | 了解整體架構 |

---

## 資料

**來源**：物理治療師於長照中心現場收案，使用握力器與量尺評估，
並錄製起立坐下／行走／半蹲三項動作。三項動作對應 AWGS 2019
（亞洲肌少症工作小組）診斷準則中的體能表現項目。

| 項目 | 數量 |
|---|---|
| 受試者 | 31 人（陽性 12 / 陰性 19） |
| 動作樣本 | 92 筆 |
| 影片規格 | 720p / 30fps |
| 骨架格式 | MediaPipe `pose_world_landmarks`，33 點 × (x, y, z, visibility) |

```
data/
├── A_sit_json/     A003_Sit.json ...
├── A_walk_json/    A003_Walk.json ...
└── A_squat_json/   A003_Squat.json ...

目前檔案範例/patient_data(summer).csv   標籤與基本資料
```

> **注意**：本 repo 含真實個案資料（編號化，無姓名）。
> 請維持 repository 為 private，勿轉載至公開場所。

---

## 環境設定

```bash
# 建立虛擬環境
python -m venv .venv_phase3

# Windows
.venv_phase3\Scripts\activate
# macOS / Linux
source .venv_phase3/bin/activate

pip install -r requirements.txt
```

---

## 執行方式

```bash
# 檢視訓練名冊（受試者數、排除紀錄）
python -m src.dataset_manifest

# 檢視交叉驗證切分
python -m src.splits

# 訓練系統 A（手工特徵基線）
python -m src.train_baseline --classifier logreg --save
python -m src.train_baseline --classifier rf --save

# 執行測試
pytest tests/ -q
```

> Windows 終端機若出現中文亂碼，加上 `PYTHONIOENCODING=utf-8`。

---

## 程式結構

```
src/
├── dataset_manifest.py     訓練名冊：標籤 CSV ↔ 骨架 JSON 對應，排除規則
├── skeleton_io.py          骨架前處理：座標軸修正、正規化、bone stream
├── splits.py               交叉驗證切分（按受試者分組）
├── features_handcrafted.py 臨床運動學特徵（系統 A）
├── evaluation.py           共用評估指標（out-of-fold 彙總）
├── train_baseline.py       系統 A 訓練
├── models_shiftgcn.py      ⚠ 舊版，待重寫
├── train.py                ⚠ 舊版，待重寫
├── inference.py            ⚠ 舊版，待修正
├── api_inference.py        ⚠ 舊版，待修正
└── bridge_mediapipe.py     JSON → .trc/.mot 轉換（目前流程已不使用）

tests/                      85 個單元測試
results/                    訓練結果 JSON
_archive/                   Phase 1~3 舊版程式碼（RandomForest / 1D-CNN 時期）
```

---

## 三系統比較架構

| 系統 | 模型 | 角色 | 狀態 |
|---|---|---|---|
| A | 手工特徵 + LogReg / RandomForest | 對照基線 | ✅ 完成 |
| B | ST-GCN（含 2s-AGCN 的 bone stream 與 adaptive graph） | 教授指定主力 | ⬜ 開發中 |
| C | Shift-GCN | 額外驗證 | ⬜ 未開始 |

三系統共用同一份名冊、前處理與交叉驗證切分，各自獨立訓練與評估後並排比較。
現階段不做加權合併（ensemble）——31 人的樣本量不足以支撐權重校準。

每個系統內部 Sit / Walk / Squat 各訓練一個模型，
推論時以 late fusion **動態分母**整合，支援受試者僅完成 2/3 項動作的情境。

---

## 目前進度

```
【準備階段】
  ✅ 訓練名冊（31 人 / 92 樣本）
  ✅ 骨架前處理（座標軸修正、正規化、bone stream）
  ✅ 交叉驗證切分（按人分組）
  ✅ 手工特徵萃取
  ✅ 單元測試（85 個）

【模型訓練階段】
  ✅ 系統 A：手工特徵 + LogReg / RandomForest
  ⬜ 系統 B：ST-GCN
  ⬜ 系統 C：Shift-GCN

【收尾階段】
  ⬜ 資料健檢報告（回饋收案端拍攝品質）
  ⬜ 推論模組修正 + 伺服器整合契約
  ⬜ 三系統比較報告
```

---

## 給協作組員

模型端與前端／伺服器端的介面尚在調整中：
現有的 `src/api_inference.py` 是舊架構（先把 JSON 轉成 `.trc` 再推論），
會改為直接讀取 JSON。**API 契約文件完成後會另行提供**，
在此之前請勿依賴現有介面。

### 分支流程

```bash
git checkout -b feat/你的功能
# ... 開發 ...
git push -u origin feat/你的功能
# 在 GitHub 開 Pull Request，合併後刪除分支
```

請勿直接推送到 `main`。
