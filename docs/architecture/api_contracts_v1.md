# API 契約規格 v1

> 文件狀態：Draft v1 可審閱  
> 子任務：Phase 1.1 System Architect  
> 對齊文件：
> - docs/governance/data_dictionary_and_label_definition_v1.md
> - docs/governance/audit_logging_and_retention_policy_v1.md
> - docs/governance/latency_budget_and_measurement_spec_v1.md
> - docs/governance/raci_and_decision_sla_v1.md

---

## 1. 契約通則

### 1.1 共通欄位與標頭

| 類型 | 欄位 | 規則 |
|---|---|---|
| Header | X-Trace-Id | 必填，跨鏈路一致，格式為不含個資之唯一字串 |
| Header | X-Request-Id | 必填，每次 hop 需唯一 |
| Header | Idempotency-Key | 依端點規則必填或選填，重試去重使用 |
| Header | X-Actor-Id | 人員或服務主體識別，需對齊審計欄位 actor_id |
| Body | schema_version | 涉及資料內容端點必填 |
| Body | model_version | 推論相關端點回傳必填 |

### 1.2 共通錯誤碼命名

| 類別 | 範例 | 說明 |
|---|---|---|
| 驗證 | VAL-400-INVALID_SCHEMA | 請求結構或欄位不符合契約 |
| 授權 | AUTH-401-INVALID | 驗證失敗 |
| 權限 | AUTH-403-DENIED | 無權限存取目標資源 |
| 找不到 | RES-404-NOT_FOUND | 資源不存在 |
| 衝突 | IDEMP-409-CONFLICT | 冪等鍵衝突或重複狀態不一致 |
| 服務 | SVC-503-UNAVAILABLE | 下游服務暫不可用 |
| 逾時 | SVC-504-TIMEOUT | 關鍵路徑逾時 |

### 1.3 審計與 SLA 標記規則

1. 所有端點都需產生審計事件，且須含 `trace_id` 與 `request_id`。  
2. SLA 關鍵端點需對應延遲段 S1 至 S7。  
3. 非同步端點回應 202 時，必須回傳可追蹤之 job id 或 export id。

---

## 2. API 契約清單

## 2.1 上傳資料

### 基本資訊

| 項目 | 定義 |
|---|---|
| API 名稱 | upload_csv |
| Method | POST |
| Path | /api/v1/uploads |
| SLA 關鍵端點 | 是 |
| 主要延遲段 | S1 S2 S3 |
| 審計事件 | data.upload.received data.upload.validated |
| Idempotency | 必填 `Idempotency-Key`，相同鍵重送不可重複建立批次 |

### Request Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| file_name | string | 是 | 檔名模板，不含本機敏感路徑 |
| file_size_bytes | integer | 是 | 檔案大小，需小於平台限制 |
| file_content_base64 | string | 是 | CSV 內容編碼字串 |
| schema_version | string | 是 | 上傳資料 schema 版本 |
| source_system | string | 是 | 來源系統代碼白名單 |
| upload_batch_id | string | 否 | 客戶端帶入可選，未帶入由系統產生 |

### Response Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| upload_batch_id | string | 是 | 批次追蹤識別 |
| status | string | 是 | received 或 validated 或 rejected |
| schema_version | string | 是 | 受理 schema 版本 |
| trace_id | string | 是 | 追蹤識別 |
| request_id | string | 是 | 請求識別 |

### 錯誤碼

`VAL-400-INVALID_SCHEMA` `AUTH-401-INVALID` `AUTH-403-DENIED` `VAL-422-QUALITY_GATE_FAILED` `IDEMP-409-CONFLICT` `SVC-503-UNAVAILABLE`

---

## 2.2 啟動前處理作業

### 基本資訊

| 項目 | 定義 |
|---|---|
| API 名稱 | start_preprocess_job |
| Method | POST |
| Path | /api/v1/preprocess/jobs |
| SLA 關鍵端點 | 條件式關鍵 |
| 主要延遲段 | S3 S4 |
| 審計事件 | preprocess.job.completed |
| Idempotency | 必填 `Idempotency-Key`，避免重複建立作業 |

### Request Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| upload_batch_id | string | 是 | 來源上傳批次 |
| preprocess_rule_version | string | 是 | 前處理規則版本 |
| priority | string | 否 | normal 或 high |
| trigger_by | string | 是 | user 或 system |

### Response Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| preprocess_job_id | string | 是 | 前處理作業識別 |
| status | string | 是 | queued 或 running |
| accepted_at_utc | string | 是 | ISO-8601 UTC |
| trace_id | string | 是 | 追蹤識別 |
| request_id | string | 是 | 請求識別 |

### 錯誤碼

`VAL-400-INVALID_SCHEMA` `RES-404-NOT_FOUND` `VAL-409-INVALID_BATCH_STATE` `IDEMP-409-CONFLICT` `SVC-503-UNAVAILABLE`

---

## 2.3 單筆推論

### 基本資訊

| 項目 | 定義 |
|---|---|
| API 名稱 | run_inference |
| Method | POST |
| Path | /api/v1/inference |
| SLA 關鍵端點 | 是 |
| 主要延遲段 | S1 S2 S4 S5 S6 S7 |
| 審計事件 | inference.request.received inference.response.completed |
| Idempotency | 選填 `Idempotency-Key`，建議呼叫端重試時提供 |

### Request Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| inference_request_id | string | 是 | 業務端請求識別 |
| schema_version | string | 是 | 特徵欄位版本 |
| features | object | 是 | 特徵鍵值集合，欄位需對齊資料字典 |
| explain_required | boolean | 是 | 是否回傳解釋結果 |
| model_version_hint | string | 否 | 期望模型版本 |

### Response Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| inference_request_id | string | 是 | 原請求識別 |
| result_id | string | 是 | 結果識別 |
| model_version | string | 是 | 實際推論模型版本 |
| risk_score_bucket | string | 是 | 分箱等級 |
| explanation | object | 否 | explain_required 為 true 時回傳 |
| latency_ms | integer | 是 | 服務端 E2E 內部延遲 |
| trace_id | string | 是 | 追蹤識別 |
| request_id | string | 是 | 請求識別 |

### 錯誤碼

`VAL-400-INVALID_SCHEMA` `AUTH-401-INVALID` `AUTH-403-DENIED` `SVC-503-UNAVAILABLE` `SVC-504-TIMEOUT`

---

## 2.4 查詢推論結果

### 基本資訊

| 項目 | 定義 |
|---|---|
| API 名稱 | get_inference_result |
| Method | GET |
| Path | /api/v1/inference/results/{result_id} |
| SLA 關鍵端點 | 是 |
| 主要延遲段 | S1 S2 S7 |
| 審計事件 | result.viewed |
| Idempotency | 不適用 |

### Request Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| path.result_id | string | 是 | 推論結果識別 |
| query.include_explanation | boolean | 否 | 是否包含解釋摘要 |

### Response Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| result_id | string | 是 | 推論結果識別 |
| model_version | string | 是 | 推論模型版本 |
| risk_score_bucket | string | 是 | 風險分箱 |
| explanation_summary | object | 否 | 解釋摘要 |
| created_at_utc | string | 是 | 建立時間 |
| trace_id | string | 是 | 追蹤識別 |
| request_id | string | 是 | 請求識別 |

### 錯誤碼

`AUTH-401-INVALID` `AUTH-403-DENIED` `RES-404-NOT_FOUND` `SVC-503-UNAVAILABLE`

---

## 2.5 匯出報表

### 基本資訊

| 項目 | 定義 |
|---|---|
| API 名稱 | export_report |
| Method | POST |
| Path | /api/v1/reports/exports |
| SLA 關鍵端點 | 否 非同步任務 |
| 主要延遲段 | S1 S2 S7 啟動段 |
| 審計事件 | report.export.generated |
| Idempotency | 必填 `Idempotency-Key`，同條件不得重複建立匯出任務 |

### Request Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| report_type | string | 是 | 例如 risk_summary |
| time_range_start_utc | string | 是 | 起始時間 |
| time_range_end_utc | string | 是 | 結束時間 |
| export_format | string | 是 | csv 或 xlsx |
| filter | object | 否 | 去識別化條件 |

### Response Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| export_id | string | 是 | 匯出任務識別 |
| status | string | 是 | accepted 或 processing |
| download_url | string | 否 | 完成後提供短效連結 |
| expires_at_utc | string | 否 | 下載連結有效期 |
| trace_id | string | 是 | 追蹤識別 |
| request_id | string | 是 | 請求識別 |

### 錯誤碼

`VAL-400-INVALID_SCHEMA` `AUTH-401-INVALID` `AUTH-403-DENIED` `VAL-422-INVALID_TIME_RANGE` `IDEMP-409-CONFLICT` `SVC-503-UNAVAILABLE`

---

## 2.6 健康檢查

### 基本資訊

| 項目 | 定義 |
|---|---|
| API 名稱 | health_check |
| Method | GET |
| Path | /api/v1/health |
| SLA 關鍵端點 | 否 |
| 主要延遲段 | S1 |
| 審計事件 | system.exception.raised 僅異常時 |
| Idempotency | 不適用 |

### Request Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| query.scope | string | 否 | basic 或 deep |

### Response Schema

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| status | string | 是 | ok 或 degraded 或 fail |
| service | string | 是 | 服務名稱 |
| version | string | 是 | 服務版本 |
| timestamp_utc | string | 是 | 回應時間 |
| trace_id | string | 是 | 追蹤識別 |
| request_id | string | 是 | 請求識別 |

### 錯誤碼

`SVC-503-UNAVAILABLE`

---

## 3. Trace Request Id 與審計對齊規範

| 規範項目 | 要求 |
|---|---|
| trace_id 來源 | 由 Gateway 接收或產生，沿所有下游沿用 |
| request_id 粒度 | 每次服務 hop 產生新值，需保留 parent 參照 |
| 審計一致性 | API 回應中的 trace_id request_id 必須可對照審計欄位 trace_id request_id |
| 量測一致性 | 指標資料需同步攜帶 trace_id request_id 以供追溯 |

---

## 4. 跨文件一致性檢查清單

| 檢查面向 | 對齊基準 | 本文件狀態 |
|---|---|---|
| 資料欄位命名 | data dictionary 欄位 snake_case 與 schema_version | 已對齊 |
| 標籤洩漏防護 | 不在 API 記錄原始敏感值 | 已對齊 |
| 審計事件名稱 | audit logging event_name 字典 | 已對齊 |
| 追蹤欄位 | trace_id request_id 必填 | 已對齊 |
| 延遲關鍵端點標記 | latency spec S1 至 S7 | 已對齊 |
| RACI 決策與升級 | P1 P2 事件升級路徑 | 已對齊 |

---

## 5. 版本控管

- 版本：v1  
- 狀態：Draft v1 可審閱  
- 進版條件：API 契約審查完成與待確認事項簽核

