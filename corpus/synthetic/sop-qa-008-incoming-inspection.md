---
document_id: sop-qa-008-incoming-inspection
document_title: "SOP-QA-008 — Incoming Material Inspection Procedure"
revision: "Fictional Rev. 4, effective 2025-01-15 (synthetic)"
source_type: synthetic
source_url_or_note: "Synthetic document authored for this portfolio project — not a real facility record. Modeled loosely on generic ISO 9001-style incoming inspection practice; no text is copied from the ISO 9001 standard itself."
source_page_range: null
---

# SOP-QA-008 — Incoming Material Inspection Procedure

> **SYNTHETIC DOCUMENT.** This is a fictional standard operating procedure authored for this portfolio project, illustrating typical QMS incoming-inspection structure. It does not describe a real facility's actual procedure and does not reproduce any actual ISO or other standard's text.

## 1. Purpose

This procedure defines the requirements for inspecting, testing, and dispositioning purchased materials, components, and subassemblies upon receipt, to prevent nonconforming incoming material from entering production.

## 2. Scope

Applies to all purchased raw materials, components, and outsourced subassemblies received at the facility's receiving dock, with the exception of: (a) materials received under a certified supplier skip-lot program (see Section 6), and (b) direct-ship materials consigned to a specific customer order that bypass stock (handled under a separate customer-specific procedure).

## 3. Responsibilities

- **Receiving clerk**: logs receipt, verifies shipment matches purchase order quantity and part number, places material in the quarantine/hold area pending inspection.
- **Incoming quality inspector**: performs sampling inspection per the applicable inspection plan, records results, and applies disposition status.
- **Quality engineer**: reviews and approves deviations, first-article inspection reports, and supplier corrective action requests arising from incoming rejections.

## 4. Procedure

### 4.1 Receipt and identification

1. Upon receipt, the receiving clerk verifies the shipment against the purchase order: part number, revision level, quantity, and supplier certificate of conformance (if required by the purchase order).
2. Each lot is assigned a unique receiving lot number and placed in the quarantine area. Quarantined material shall be physically or systematically segregated such that it cannot be inadvertently moved to production or unrestricted stock before inspection is complete.
3. Any shipment with damaged packaging, missing documentation, or a quantity discrepancy exceeding ±2% of the ordered quantity is flagged for quality engineer review before inspection proceeds.

### 4.2 Sampling and inspection

4. The incoming quality inspector selects the applicable inspection plan based on the part number's classification (Critical, Major, or Minor — per the approved parts classification list maintained by Quality Engineering).
5. Sample size is determined per the facility's sampling plan (based on ANSI/ASQ Z1.4 general inspection level II unless the part-specific inspection plan specifies otherwise).
6. Inspection characteristics, methods, and acceptance criteria are defined on the part-specific inspection plan. At minimum, dimensional characteristics called out on the purchase specification/print as inspection-required, and any characteristics flagged Critical or Major on the parts classification list, shall be verified.
7. For parts requiring certificate-based acceptance only (i.e., visual/documentation review without dimensional sampling), the inspector verifies the certificate of conformance content matches the purchase order requirements and that all required test results are within specified limits.

### 4.3 Disposition

8. Material that passes inspection is labeled with an "Accepted" tag bearing the lot number, inspector initials, and inspection date, and is released to unrestricted stock.
9. Material that fails inspection is labeled "Rejected — Hold" and routed per SOP-QA-014 (Nonconforming Material Disposition Procedure). Rejected material shall not be released to production under any circumstances without a documented and approved deviation.
10. Where a sampling inspection result is borderline (i.e., within the sampling plan's indifference zone, if applicable) or where inspection results are inconsistent with the supplier's certificate of conformance, the inspector shall escalate to the quality engineer before dispositioning, rather than defaulting to acceptance or rejection.

### 4.4 Records

11. Inspection results (sample size, characteristics measured, measured values, pass/fail determination, inspector, date) shall be recorded in the incoming inspection log and retained for a minimum of 3 years or per the applicable customer contract's record retention requirement, whichever is longer.
12. Rejection records shall be linked to the corresponding supplier corrective action request (if issued) to support supplier quality performance tracking.

## 5. First Article Inspection

For a new part number, a new supplier for an existing part number, or a part number returning to production after a design change affecting form/fit/function, a first article inspection (FAI) is required in addition to routine sampling inspection. The FAI shall verify 100% of print characteristics on a minimum of one piece from the first production lot, documented on a first article inspection report, and approved by the quality engineer before the lot (or any subsequent lot from that supplier/revision) is released.

## 6. Supplier Skip-Lot Program

Suppliers who have demonstrated sustained conforming quality (defined as zero rejections across the most recent 10 consecutive lots) may be placed on a skip-lot inspection program by the quality engineer, under which every third lot receives full sampling inspection and intervening lots receive certificate-of-conformance review only. Any rejection while on skip-lot status immediately returns the supplier/part combination to full lot-by-lot inspection for a minimum of 10 subsequent lots.

## 7. Related Documents

- SOP-QA-014 — Nonconforming Material Disposition Procedure
- Approved Parts Classification List (maintained by Quality Engineering)
- Part-specific inspection plans (maintained in the quality document control system)

## 8. Revision History

| Rev | Date | Change |
|---|---|---|
| 1 | 2021-06-01 | Initial release |
| 2 | 2022-09-12 | Added supplier skip-lot program (Section 6) |
| 3 | 2024-02-20 | Added escalation requirement for borderline/inconsistent results (Step 10) |
| 4 | 2025-01-15 | Clarified quarantine segregation requirement (Step 2) |
