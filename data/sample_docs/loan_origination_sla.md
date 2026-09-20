# Loan Origination SLA and Process (Sample)

> **SYNTHETIC SAMPLE DOCUMENT.** This file is fictional demo data for the Lending Copilot
> portfolio project. Any resemblance to real institutions, products, or persons is coincidental.

## 1. Origination stages

1. Application and document collection (branch, app, or DSA channel).
2. KYC verification and de-duplication check.
3. Bureau pull and credit decision (automated engine or manual underwriter).
4. Sanction letter and customer acceptance.
5. Disbursement to the borrower's verified bank account.

## 2. Turnaround time (TAT) SLAs

| Stage | SLA |
|-------|-----|
| KYC verification | 4 working hours |
| Credit decision (personal loan) | 24 working hours |
| Credit decision (home loan) | 72 working hours |
| Disbursement after acceptance | 4 working hours |

## 3. De-duplication

Every application is checked against the existing customer base on mobile number, PAN,
and Aadhaar. Matches are merged into a single customer ID before underwriting.

## 4. Sanction validity

Sanction letters are valid for 30 days for personal loans and 90 days for home loans.
Revalidation requires a fresh bureau pull if the validity has lapsed.

## 5. Disbursement controls

- Disbursement only to a bank account verified by penny-drop.
- First EMI date must be at least 15 days after disbursement.
- The welcome kit with the repayment schedule is sent within 7 days of disbursement.
