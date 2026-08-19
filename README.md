# 肌少症骨架偵測系統

從影片判斷受試者是否有肌少症風險，並提供可解釋的判斷依據。

涵蓋前端網頁、後端伺服器、影片轉骨架、以及模型訓練與推論的完整流程。

## 文件導覽

| 文件 | 內容 | 給誰看 |
|---|---|---|
| **README.md**（本檔） | 專案總覽、環境設定、如何執行 | 所有人 |
| [RESULTS.md](RESULTS.md) | 三系統比較的實驗數據 | 想看模型表現 |
| [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | 開發歷程與**決策理由** | 接手/回顧/寫報告 |
| [docs/API_CONTRACT.md](docs/API_CONTRACT.md) | **推論介面規格** | 前端／伺服器組員 |
| [results/data_health_report.md](results/data_health_report.md) | 資料品質與收案建議 | 與醫療機構討論 |
| [DESIGN.md](DESIGN.md) | 原始設計構想（歷史紀錄，與現況有差異） | 寫報告時對照「原始計畫 vs 實際成果」 |

---

## 資料

**來源**：物理治療師於長照中心現場收案，使用握力器與量尺評估，
並錄製起立坐下／行走／半蹲三項動作。三項動作對應 AWGS 2019
（亞洲肌少症工作小組）診斷準則中的體能表現項目。

| 項目 | 數量 |
|---|---|
| 真實受試者 | 31 人（陽性 12 / 陰性 19） |
| 動作樣本 | 92 筆（Sit 30、Walk 31、Squat 31） |
| 合成受試者 | 58 人（僅供預訓練，不參與評估） |
| 影片規格 | 720p / 30fps |
| 骨架格式 | MediaPipe `pose_world_landmarks`，33 點 × (x, y, z, visibility) |

```
data/
├── A_sit_json/     真實個案：A003_Sit.json ...
├── A_walk_json/
├── A_squat_json/
└── uploads/        伺服器上傳的資料（不進版控）

真資料+假資料/
├── patient_data(summer).csv   真實個案的標籤與基本資料
├── patient_data.csv           合成個案的標籤
├── sit_json/ walk/ squat_json/   合成骨架（S001–S066）
```

**合成資料只用於預訓練**，且僅使用骨架、不使用其年齡等欄位——
合成資料的年齡與標籤關係與真實資料相反（詳見 [RESULTS.md](RESULTS.md)）。

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

## 啟動伺服器（前端 + 後端）

```bash
setup.bat          # 首次執行：安裝套件並訓練模型
start_server.bat   # 啟動，瀏覽器開啟 http://localhost:8000
```

或手動：

```bash
pip install -r requirements.txt
python -m src.train_final                  # 產生 models/
python -m uvicorn api_server:app --port 8000
```

| 端點 | 用途 |
|---|---|
| `GET /` | 前端網頁 |
| `GET /health` | 健康檢查（確認模型檔是否就緒） |
| `POST /upload_data/` | 上傳影片 → 擷取骨架 → 評估風險 |
| `GET /result/{user_id}` | 以現有骨架重新評估，不需重新上傳 |
| `POST /label/` | 治療師標記診斷結果 |
| `GET /docs` | FastAPI 自動產生的 API 文件 |

---

## 程式結構

```
api_server.py               FastAPI 伺服器（上傳、評估、標記）
extract_to_json.py          影片 → MediaPipe 骨架 JSON
static/index.html           前端網頁
pose_landmarker_heavy.task  MediaPipe 模型檔

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
└── data_health_report.py   資料健檢報告產生器

models/                     正式模型（由 train_final.py 產生）
docs/API_CONTRACT.md        推論介面規格
tests/                      154 個單元測試
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
【資料與基礎建設】
  ✅ 訓練名冊（31 人 / 92 樣本）
  ✅ 骨架前處理（座標軸修正、正規化、bone stream）
  ✅ 交叉驗證切分（按人分組，避免資料洩漏）
  ✅ 單元測試（154 個）

【模型】
  ✅ 系統 A：手工特徵 + LogReg / RandomForest
  ✅ 系統 B：ST-GCN（含合成資料預訓練）
  ✅ 系統 C：Shift-GCN
  ✅ 動作組合消融、錯誤案例分析

【產品】
  ✅ 前端網頁 + FastAPI 伺服器 + 影片轉骨架
  ✅ 推論模組與 API 契約
  ✅ 資料健檢報告（回饋收案端）
```

---

## 持續收案與重新訓練

每收到一批新個案後的流程：

```bash
# 1. 收案：治療師用網頁上傳影片並標記
start_server.bat        # 資料存到 data/uploads/

# 2. 人工策展：把要納入訓練的個案搬進訓練集
#    骨架 JSON → data/A_sit_json/ 等
#    標籤      → 真資料+假資料/patient_data(summer).csv

# 3. 檢查資料完整性（會抓出人數不符、生理數值異常）
pytest tests/test_dataset_manifest.py -q

# 4. 重新訓練與評估
python -m src.train_gcn --model stgcn --all-groups --pretrain --save
python -m src.train_baseline --classifier logreg --save

# 5. 更新健檢報告
python -m src.data_health_report --save

# 6. 滿意後封存新模型（前後端無須改動，自動使用新模型）
python -m src.train_final
```

**第 2 步刻意保持人工**——伺服器不會自動把上傳資料寫入訓練集，
避免未經稽核的資料在無人察覺的情況下影響實驗結果。

### 判斷「準確度是否足夠」的依據

| 指標 | 目前 | 目標 |
|---|---|---|
| AUC 的 95% 信賴區間 | [0.64, 0.95] | 收窄至不再涵蓋 0.794（單看年齡的基準） |
| 純骨架 AUC | 0.789 | 明顯超過人口學基準，方能證明動作分析的價值 |
| 兩組年齡差距 | 13 歲 | 補收 80 歲以上健康者與 75 歲以下患者以縮小 |

信賴區間會隨樣本數增加而收窄，是最直接的進度指標。

---

## 給協作組員

**推論介面規格見 [docs/API_CONTRACT.md](docs/API_CONTRACT.md)。**

入口為 `src/predict.py` 的 `predict()`。

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
