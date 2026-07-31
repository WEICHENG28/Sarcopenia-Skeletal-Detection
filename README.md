# 肌少症骨架偵測系統 — 模型端

從 MediaPipe 骨架動作資料判斷受試者是否有肌少症風險，並提供可解釋的判斷依據。

## 文件導覽

| 文件 | 內容 | 給誰看 |
|---|---|---|
| **README.md**（本檔） | 專案總覽、環境設定、如何執行 | 所有人 |
| [RESULTS.md](RESULTS.md) | 三系統比較的實驗數據 | 想看模型表現 |
| [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | 開發歷程與**決策理由** | 接手/回顧/寫報告 |
| [docs/API_CONTRACT.md](docs/API_CONTRACT.md) | **推論介面規格** | 前端／伺服器組員 |
| [results/data_health_report.md](results/data_health_report.md) | 資料品質與收案建議 | 與醫療機構討論 |
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

# 訓練系統 B / C（圖神經網路，含合成資料預訓練）
python -m src.train_gcn --model stgcn --all-groups --pretrain --save
python -m src.train_gcn --model shiftgcn --all-groups --pretrain --save

# 動作組合消融
python -m src.ablation_actions --save

# 產生資料健檢報告
python -m src.data_health_report --save

# 訓練並封存正式模型
python -m src.train_final

# 單筆推論
python -m src.predict --age 78 --height_cm 152 --weight_kg 48 --gender female     --sit data/A_sit_json/A010_Sit.json     --walk data/A_walk_json/A010_Walk.json

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
├── datasets.py             PyTorch Dataset：各動作序列長度、資料增強
├── features_handcrafted.py 臨床運動學特徵（系統 A）
├── synthetic_data.py       合成資料名冊（僅供預訓練）
├── evaluation.py           共用評估指標（out-of-fold 彙總）
├── models_stgcn.py         ST-GCN（系統 B）
├── models_shiftgcn2.py     Shift-GCN（系統 C）
├── train_baseline.py       系統 A 訓練
├── train_gcn.py            系統 B / C 交叉驗證訓練
├── train_final.py          訓練並封存正式模型（供推論）
├── predict.py              ★ 推論服務入口（見 docs/API_CONTRACT.md）
├── ablation_actions.py     動作組合消融
├── data_health_report.py   資料健檢報告產生器
│
├── models_shiftgcn.py      ⚠ 舊版，已停用
├── train.py                ⚠ 舊版，已停用
├── inference.py            ⚠ 舊版，已停用（改用 predict.py）
├── api_inference.py        ⚠ 舊版，已停用
└── bridge_mediapipe.py     ⚠ JSON → .trc 轉換，現行流程不使用

models/                     正式模型（由 train_final.py 產生）
tests/                      151 個單元測試
results/                    訓練結果、消融結果、資料健檢報告
_archive/                   Phase 1~3 舊版程式碼（RandomForest / 1D-CNN 時期）
```

---

## 三系統比較架構

| 系統 | 模型 | 角色 | 狀態 |
|---|---|---|---|
| A | 手工特徵 + LogReg / RandomForest | 對照基線 | ✅ 0.825 |
| B | ST-GCN（含 2s-AGCN 的 bone stream 與 adaptive graph） | 教授指定主力 | ✅ 0.823 |
| C | Shift-GCN | 額外驗證 | ✅ 0.828＊ |

＊系統 C 的 0.828 幾乎全部來自人口學特徵，其純骨架表現僅 0.439。
詳見 [RESULTS.md](RESULTS.md)。

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
  ✅ 系統 B：ST-GCN（含合成資料預訓練）
  ✅ 系統 C：Shift-GCN
  ✅ 動作組合消融

【收尾階段】
  ✅ 資料健檢報告（回饋收案端拍攝品質）
  ✅ 推論模組 + 伺服器整合契約
  ✅ 三系統比較報告
```

---

## 給協作組員

**推論介面規格見 [docs/API_CONTRACT.md](docs/API_CONTRACT.md)。**

入口為 `src/predict.py` 的 `predict()`。舊版 `src/inference.py` 與
`src/api_inference.py` 已停用（走 `.trc` 轉檔的舊架構），請勿使用。

```python
from src.predict import predict

result = predict(
    profile={"age": 78, "height_cm": 152, "weight_kg": 48, "gender": "female"},
    actions={"Sit": "path/to/Sit.json", "Walk": "path/to/Walk.json"},
)
payload = result.to_dict()
```

使用前請先執行 `python -m src.train_final` 產生 `models/` 下的模型檔。

### 分支流程

```bash
git checkout -b feat/你的功能
# ... 開發 ...
git push -u origin feat/你的功能
# 在 GitHub 開 Pull Request，合併後刪除分支
```

請勿直接推送到 `main`。
