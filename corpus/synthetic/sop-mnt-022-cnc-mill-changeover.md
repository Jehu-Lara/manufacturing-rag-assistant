---
document_id: sop-mnt-022-cnc-mill-changeover
document_title: "SOP-MNT-022 — Preventive Maintenance Procedure: CNC Mill Tooling Changeover"
revision: "Fictional Rev. 3, effective 2024-11-01 (synthetic)"
source_type: synthetic
source_url_or_note: "Synthetic document authored for this portfolio project — not a real facility record. Equipment IDs and specific parameter values below are illustrative and do not describe a real facility."
source_page_range: null
---

# SOP-MNT-022 — Preventive Maintenance Procedure: CNC Mill Tooling Changeover

> **SYNTHETIC DOCUMENT.** This is a fictional standard operating procedure authored for this portfolio project, modeled on typical manufacturing SOP structure. It does not describe a real facility's actual procedure.

## 1. Purpose

This procedure defines the required steps for changing over tooling on 3-axis CNC vertical milling centers between production job runs, to ensure changeover is performed safely, consistently, and without introducing dimensional errors into the next job's first-off parts.

## 2. Scope

Applies to all CNC vertical milling centers in Machining Cell 2 (equipment IDs MILL-201 through MILL-206). Does not apply to lathes, 5-axis mills, or manual milling machines, which are covered under separate SOPs.

## 3. Responsibilities

- **Machine operator**: initiates changeover request, removes/stages outgoing tooling, verifies work-holding cleanliness.
- **Setup technician**: performs tool offset entry, work coordinate system (WCS) zeroing, and first-off part verification.
- **Quality technician**: inspects and approves the first-off part before the run is released to production.

## 4. Prerequisites

- The outgoing job's last part must be tagged and set aside per SOP-QA-008 (Incoming Material Inspection) traceability requirements if it is a nonconforming hold, or moved to finished-goods staging if conforming.
- The incoming job's traveler, print (latest revision), and tool list must be available at the machine before changeover begins.
- Required tooling must be pulled from the tool crib and verified against the tool list part numbers before changeover starts (do not begin changeover with incomplete tooling — stage and wait).

## 5. Procedure

### 5.1 Machine shutdown and tool removal

1. Bring the machine to a complete stop at a safe tool-change position; do not interrupt an active cutting cycle to begin changeover.
2. Place the machine in manual mode and confirm the spindle is stopped (0 RPM) before opening the enclosure door.
3. Remove outgoing tooling from the carousel/turret and return each tool to its labeled slot in the tool crib.
4. Remove the outgoing work-holding fixture if the incoming job requires a different fixture; clean the table surface and locating pins of chips and coolant residue.

### 5.2 Incoming tooling and fixture installation

5. Install the incoming work-holding fixture per the fixture setup sheet referenced on the job traveler. Torque fixture clamp bolts to the value specified on the setup sheet (typically 40-60 N·m for standard vise-style fixtures; confirm against the specific setup sheet).
6. Load incoming tooling into the carousel/turret positions specified on the tool list. Verify each tool's length and diameter against the tool list before loading — do not rely on memory or visual estimation for tools that look similar.
7. Enter tool offsets (length and diameter/radius) into the control's tool offset table. A second person (setup technician or operator, whichever did not perform the entry) must verbally confirm each entered value against the tool list before proceeding — this is a mandatory second-check, not optional.

### 5.3 Work coordinate system zeroing

8. Establish the work coordinate system (WCS) origin per the job's setup sheet, using an edge finder, touch probe, or fixture-referenced offset as specified.
9. Record the zeroed WCS values on the setup sheet.
10. Perform a dry run (air cut, no material loaded, or with rapid override reduced and single-block mode enabled) to visually verify the toolpath clears all fixture components and does not exceed expected travel.

### 5.4 First-off part run and verification

11. Load the first workpiece and run the program at reduced feed override (50% or less) for the first cutting pass.
12. If the first pass completes without incident, return feed override to programmed rate for remaining operations.
13. Remove the completed first-off part and route it to the quality technician for inspection per the job's inspection plan.
14. **Do not begin production of additional parts until the quality technician has approved the first-off part.** This hold point is mandatory and may not be waived by the operator or setup technician.

## 6. Documentation

The setup technician shall record on the job traveler: changeover start/end time, tooling used (cross-referenced to the tool list), WCS zero values, and the first-off part's serial or lot identifier. This record supports both quality traceability and downtime/changeover-time tracking in the CMMS.

## 7. Related Documents

- SOP-QA-008 — Incoming Material Inspection Procedure
- SOP-QA-014 — Nonconforming Material Disposition Procedure
- Job-specific setup sheets and tool lists (maintained in the job traveler package)

## 8. Revision History

| Rev | Date | Change |
|---|---|---|
| 1 | 2023-02-15 | Initial release |
| 2 | 2024-03-01 | Added mandatory second-check for tool offset entry (Step 7) following an internal near-miss investigation |
| 3 | 2024-11-01 | Clarified first-off part hold point (Step 14) is non-waivable |
