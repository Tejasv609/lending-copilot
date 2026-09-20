# Lending Copilot — Eval Results

Evaluated 15 questions against the sample corpus (top-k=5, answer mode: extractive).

| Metric | Value |
|---|---|
| Hit rate@5 | 1.0 |
| MRR | 0.756 |
| Avg keyword coverage | 0.978 |
| Citation rate | 1.0 |

## Per-question breakdown

| # | Question | Expected doc | Hit | RR | Keyword cov. | Citations |
|---|---|---|---|---|---|---|
| 1 | What is the minimum CIBIL score needed to get a personal loa | retail_credit_policy | yes | 1.0 | 1.0 | 5 |
| 2 | What interest rate will I get on a personal loan with a CIBI | interest_rate_card | yes | 0.5 | 1.0 | 5 |
| 3 | What documents are required for KYC before disbursement? | kyc_aml_checklist | yes | 1.0 | 0.667 | 5 |
| 4 | When do field visits start for delinquent accounts? | delinquency_collections_playbook | yes | 1.0 | 1.0 | 5 |
| 5 | What does DPD stand for in collections? | credit_glossary | yes | 0.333 | 1.0 | 5 |
| 6 | What is the maximum DTI allowed for a personal loan? | retail_credit_policy | yes | 0.5 | 1.0 | 5 |
| 7 | What are the prepayment charges on a personal loan? | interest_rate_card | yes | 1.0 | 1.0 | 5 |
| 8 | How often must KYC be refreshed for high-risk customers? | kyc_aml_checklist | yes | 1.0 | 1.0 | 5 |
| 9 | What is the maximum personal loan amount offered? | interest_rate_card | yes | 0.5 | 1.0 | 5 |
| 10 | What is the turnaround time for a personal loan credit decis | loan_origination_sla | yes | 0.5 | 1.0 | 5 |
| 11 | What is a restructuring in the context of delinquent loans? | credit_glossary | yes | 0.5 | 1.0 | 5 |
| 12 | What is the late payment fee on a loan? | interest_rate_card | yes | 1.0 | 1.0 | 5 |
| 13 | What cash transaction threshold triggers AML reporting? | kyc_aml_checklist | yes | 1.0 | 1.0 | 5 |
| 14 | How long is a personal loan sanction letter valid? | loan_origination_sla | yes | 1.0 | 1.0 | 5 |
| 15 | What is the minimum acceptable settlement for bucket 3 accou | delinquency_collections_playbook | yes | 0.5 | 1.0 | 5 |
