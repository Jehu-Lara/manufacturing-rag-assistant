---
document_id: cmms-work-orders-line3-q2
document_title: "CMMS Work Order Export — Production Line 3, Q2 2026 (sample)"
revision: "Synthetic sample export, generated 2026-07-01"
source_type: synthetic
source_url_or_note: "Synthetic document authored for this portfolio project — not a real facility record. Equipment IDs, dates, technician names, and downtime figures are fictional and illustrative only."
source_page_range: null
---

# CMMS Work Order Export — Production Line 3, Q2 2026 (sample)

> **SYNTHETIC DOCUMENT.** This is a fictional sample export from a Computerized Maintenance Management System (CMMS), authored for this portfolio project to illustrate the kind of maintenance work-order records a RAG assistant might need to answer questions against. All equipment IDs, technician names, dates, and figures below are invented and do not describe any real facility, incident, or person.

## Report Parameters

- **Line**: Production Line 3 (Packaging)
- **Period**: 2026-04-01 through 2026-06-30
- **Work order types included**: Corrective (unplanned), Preventive (scheduled)
- **Export generated**: 2026-07-01

## Work Order Log

### WO-30142

- **Type**: Corrective (unplanned)
- **Equipment**: CONV-301 (Model XJ-450 belt conveyor, discharge section)
- **Reported**: 2026-04-03 06:47
- **Reported by**: Line operator (Shift A)
- **Symptom**: Belt tracking off-center by approximately 15 mm at the discharge end; belt cleaner blade contacting frame edge, causing intermittent stoppages on the safety interlock.
- **Technician**: J. Alvarez
- **Diagnosis**: Tail pulley take-up assembly found near end of travel; frame found out of level by ~4 mm/m following the recent line relocation.
- **Action taken**: Re-leveled conveyor frame per XJ-450 maintenance manual Section 5, Step 1. Adjusted tail pulley take-up per Section 5, Steps 4-5. Verified tracking over 15 minutes of run time.
- **Downtime**: 2h 05m
- **Status**: Closed 2026-04-03 09:10
- **Follow-up**: Recommended re-survey of all conveyors on Line 3 for level following the relocation. See WO-30190.

### WO-30167

- **Type**: Preventive (scheduled)
- **Equipment**: MILL-204 (CNC vertical mill, Cell 2)
- **Reported**: 2026-04-10 (scheduled, monthly PM per SOP-MNT-022 Section 3.2 equivalent equipment class)
- **Technician**: R. Okafor
- **Action taken**: Monthly PM completed — bearing lubrication, idler roller check (n/a to mill, template auto-populated in error, corrected), coupling bolt torque check on spindle drive coupling (85 N·m, within spec, no adjustment needed).
- **Downtime**: 0h 35m (scheduled, off-shift)
- **Status**: Closed 2026-04-10 22:05

### WO-30190

- **Type**: Preventive (scheduled, follow-up from WO-30142)
- **Equipment**: CONV-301, CONV-302, CONV-303 (all Line 3 conveyors)
- **Reported**: 2026-04-15
- **Technician**: J. Alvarez, M. Chen
- **Action taken**: Full level survey of all three Line 3 conveyors using laser level. CONV-302 found out of level by 2 mm/m (within tolerance, no action). CONV-303 found out of level by 5 mm/m; shimmed support legs at stations 3 and 7 to bring within tolerance.
- **Downtime**: 3h 40m (combined, scheduled during planned line changeover window — not counted against Line 3 unplanned downtime)
- **Status**: Closed 2026-04-15 14:20

### WO-30215

- **Type**: Corrective (unplanned)
- **Equipment**: MILL-202 (CNC vertical mill, Cell 2)
- **Reported**: 2026-04-22 13:12
- **Reported by**: Setup technician
- **Symptom**: First-off part on new job out of tolerance on a critical dimension (+0.18 mm over print, tolerance ±0.05 mm).
- **Technician**: R. Okafor
- **Diagnosis**: Incorrect tool length offset entered during changeover (transposed digits: 124.5 mm entered instead of 142.5 mm per tool list).
- **Action taken**: Corrected tool offset per tool list. Re-ran first-off part, verified in tolerance. Reviewed changeover log — second-check step (SOP-MNT-022 Section 5.2, Step 7) had been documented as performed but was not actually independently re-verified by a second person on this occasion.
- **Downtime**: 0h 50m
- **Status**: Closed 2026-04-22 14:05
- **Follow-up**: Reported to Quality Engineering as a near-miss for procedural compliance follow-up (not a customer-affecting nonconformance — caught by first-off inspection before production release, per SOP-MNT-022 Section 5.4, Step 14).

### WO-30268

- **Type**: Corrective (unplanned)
- **Equipment**: CONV-302 (belt conveyor, mid-line transfer section)
- **Reported**: 2026-05-08 03:15
- **Reported by**: Night shift operator
- **Symptom**: Conveyor stopped; motor overload trip on VFD.
- **Technician**: M. Chen
- **Diagnosis**: Buildup of packaging debris jammed between belt and tail pulley, increasing drive torque beyond overload threshold.
- **Action taken**: Cleared debris. Inspected belt for damage (none found). Reset VFD overload fault. Ran conveyor empty for 10 minutes to confirm normal operation before returning to production.
- **Downtime**: 1h 15m
- **Status**: Closed 2026-05-08 04:30
- **Follow-up**: Recommended review of guarding/skirting at the tail pulley transfer point to reduce debris ingress; opened as a separate engineering request, not tracked in this work order.

### WO-30301

- **Type**: Preventive (scheduled)
- **Equipment**: CONV-301, CONV-302, CONV-303
- **Reported**: 2026-05-15 (quarterly)
- **Technician**: J. Alvarez
- **Action taken**: Quarterly take-up screw thread lubrication per XJ-450 manual Section 4. All three units within normal parameters.
- **Downtime**: 0h 45m (combined, off-shift)
- **Status**: Closed 2026-05-15 21:30

### WO-30347

- **Type**: Corrective (unplanned)
- **Equipment**: MILL-206 (CNC vertical mill, Cell 2)
- **Reported**: 2026-06-02 10:40
- **Reported by**: Line operator
- **Symptom**: Unusual grinding noise from spindle area during rapid traverse.
- **Technician**: R. Okafor
- **Diagnosis**: Spindle drive coupling bolts found loose (below the 85 N·m spec on inspection with a calibrated torque wrench) — likely gradual loosening since the prior PM cycle.
- **Action taken**: Re-torqued all coupling bolts to spec. Ran spindle through full speed range with no load; no abnormal noise on re-test.
- **Downtime**: 1h 30m
- **Status**: Closed 2026-06-02 12:10
- **Follow-up**: Recommended reducing the coupling bolt torque-check interval from monthly to bi-weekly for this specific unit given two loosening incidents in the trailing 6 months (see facility maintenance history, not included in this excerpt).

## Summary Statistics (Q2 2026, Line 3)

| Metric | Value |
|---|---|
| Total work orders | 7 |
| Corrective (unplanned) | 4 |
| Preventive (scheduled) | 3 |
| Total unplanned downtime | 5h 40m |
| Total scheduled/PM downtime (off-shift, not counted against line availability) | 5h 00m |
| Equipment with repeat corrective work orders in period | MILL-202 (1), CONV-301/302 (1 each, different failure modes) |
