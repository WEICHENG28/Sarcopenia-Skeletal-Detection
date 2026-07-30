# 非功能需求規格 v1

> 文件狀態：Draft v1 可審閱  
> 子任務：Phase 1.2 Security and Observability Architect  
> 對齊文件：
> - docs/architecture/system_architecture_spec_v1.md
> - docs/architecture/api_contracts_v1.md
> - docs/architecture/security_privacy_observability_spec_v1.md
> - docs/governance/audit_logging_and_retention_policy_v1.md
> - docs/governance/latency_budget_and_measurement_spec_v1.md
> - docs/governance/raci_and_decision_sla_v1.md

---

## 1. 文件目的與適用範圍

本文件定義 MVP 階段可審閱之 NFR 基線，供 PM Security Backend MLOps QA 共同驗收。每個 NFR 維度均提供：

1. 量化指標  
2. 量測方法  
3. 驗收門檻 Must Should  
4. 證據來源

適用流程：資料上傳 前處理 推論 解釋 報表匯出。

---

## 2. NFR 維度與驗收規格

## 2.1 安全 Security

| 項目 | 規格 |
|---|---|
| 量化指標 | 高風險事件偵測覆蓋率 100%；法遵必記錄事件遺失率 0%；關鍵端點未授權存取阻擋率 100% |
| 量測方法 | 以攻防測試與審計抽測比對 `auth.authorization.denied` `system.exception.raised` 事件；核對 API 錯誤碼與事件鏈 |
| Must | 關鍵端點全部啟用驗證授權；審計關鍵事件不可關閉 |
| Should | 每次重大版本前完成威脅模型回顧與控制抽驗 |
| 證據來源 | 審計日誌報告 告警紀錄 滲透測試報告 設定快照 |

## 2.2 隱私 Privacy

| 項目 | 規格 |
|---|---|
| 量化指標 | 審計與觀測資料機敏欄位明文外洩率 0%；去敏規則符合率 100% |
| 量測方法 | 抽樣 logs traces metrics 與匯出檔，驗證遮罩 token 化與欄位白名單；比對資料字典與 API 契約 |
| Must | 不記錄原始特徵值 憑證 token secret；敏感識別欄位需遮罩或 token 化 |
| Should | 建立自動化去敏規則檢測與每週抽驗報告 |
| 證據來源 | 去敏抽驗報告 欄位字典比對報告 匯出檔抽查紀錄 |

## 2.3 效能 Performance

| 項目 | 規格 |
|---|---|
| 量化指標 | E2E P95 小於 500ms；E2E P99 小於 800ms；S4 S5 S6 無連續 3 窗超標 |
| 量測方法 | 依延遲規格 SCN-01 至 SCN-05 量測；輸出 stage_latency_ms 與 e2e_latency_ms 分位數 |
| Must | 主驗收情境 SCN-01 達成 P95 小於 500ms；錯誤率與逾時率各小於 1% |
| Should | 穩態情境 P95 小於 450ms；尖峰回落 10 分鐘內恢復到小於 500ms |
| 證據來源 | 延遲量測報告 壓測報表 告警與復原時間軸 |

## 2.4 可用性 Availability

| 項目 | 規格 |
|---|---|
| 量化指標 | 服務成功率 建議大於等於 99.5%；P1 事件首次回應遵循率 100% |
| 量測方法 | 以 request_error_rate timeout_rate 與 incident 紀錄統計；按週檢核事件處置時限 |
| Must | P1 30 分鐘內回應；P2 2 小時內回應；重大故障期間可啟用降級策略 |
| Should | 服務成功率持續高於 99.5% 建議值，且告警漏報率趨近 0 |
| 證據來源 | 監控儀表板 事故工單 決策紀錄 |

## 2.5 可靠性 Reliability

| 項目 | 規格 |
|---|---|
| 量化指標 | 冪等衝突處理正確率 100%；非同步作業最終完成率 建議大於等於 99%；資料處理一致性錯誤率 小於 0.5% |
| 量測方法 | 比對重試鏈與 job 狀態轉移；抽測 `Idempotency-Key` 同鍵重送行為；驗證批次與結果一致 |
| Must | 同鍵重送不得重複建立資源；重試不可無限循環；失敗需可追溯 |
| Should | 建立自動化回補與異常自癒規則，降低人工介入比例 |
| 證據來源 | 作業狀態報告 審計追溯報告 異常處理紀錄 |

## 2.6 可維運性 Maintainability

| 項目 | 規格 |
|---|---|
| 量化指標 | 可觀測欄位完整率 100%；Runbook 覆蓋率 建議大於等於 90%；告警到緩解平均時間趨勢下降 |
| 量測方法 | 盤點 logs metrics traces 欄位完整性；抽測事件對應 Runbook；統計 incident 處理時間 |
| Must | 每項 P1 P2 告警必有對應處置步驟與責任角色 |
| Should | 主要故障場景完成演練並產出改進行動清單 |
| 證據來源 | Runbook 清單 演練紀錄 事故復盤報告 |

## 2.7 可審計性 Auditability

| 項目 | 規格 |
|---|---|
| 量化指標 | 必填審計欄位完整率 100%；完整性驗證通過率 100%；關聯追溯成功率 100% |
| 量測方法 | 依審計政策抽樣驗證 `integrity_hash` 與簽章參照；以 trace_id request_id 串接 API 量測與審計事件 |
| Must | 法遵必記錄事件不可抽樣不可關閉；online 與 archive 皆可查證樣本 |
| Should | 取證封裝流程標準化，支援快速產出 Evidence Package |
| 證據來源 | 欄位完整率報告 完整性驗證報告 保留層查詢紀錄 |

---

## 3. 跨文件一致性檢核

### 3.1 對齊 API 契約與治理規格

| 對齊項目 | 參照文件 | 本文件落點 | 狀態 |
|---|---|---|---|
| 共通追蹤欄位 `X-Trace-Id` `X-Request-Id` | api contracts v1 | 全維度量測與證據要求均納入 trace_id request_id | 已對齊 |
| 端點錯誤碼與授權拒絕 | api contracts v1 | 安全 可用性 NFR 驗收指標納入授權拒絕與逾時 | 已對齊 |
| 審計事件必記錄與保留分層 | audit logging and retention policy v1 | 安全與可審計性 Must 條件明確繼承 | 已對齊 |
| E2E 與分段延遲治理 | latency budget and measurement spec v1 | 效能 NFR 直接採 S1 到 S7 與 SCN 情境 | 已對齊 |
| P1 P2 P3 決策時限 | raci and decision sla v1 | 可用性與維運性納入事件回應時限 | 已對齊 |

### 3.2 一致性結論

1. NFR 指標命名與量測口徑可回連至既有 API 契約與治理文件。  
2. 安全 隱私 可審計性三者以同一組關聯 ID 串接，降低取證斷點風險。  
3. 尚未簽核之門檻均以建議值標註待確認，避免誤宣告承諾。

---

## 4. 尚待確認事項

1. 服務成功率建議值是否採 99.5% 作為正式可用性門檻。  
2. 非同步作業最終完成率建議值是否採 99%。  
3. 去敏規則抽驗頻率是否固定每週，或依風險分級調整。  
4. Runbook 覆蓋率門檻是否採 90%。  
5. 尖峰恢復情境是否沿用 10 分鐘恢復標準。  
6. 審計取證封裝格式是否需第三方時間戳簽章。  
7. 精確風險分數保存與調閱是否需要獨立審批。  
8. 下載連結短效時間與一次性限制是否固定。  
9. 觀測平台欄位字典是否已涵蓋全部服務節點。  
10. 容量壓力告警閾值是否需依環境 dev stage prod 分層。

---

## 5. 版本控管

- 版本：v1  
- 狀態：Draft v1 可審閱  
- 進版條件：跨 PM Security Backend MLOps QA 會審通過，且第 4 節事項完成簽核

