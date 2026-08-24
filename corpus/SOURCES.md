# Corpus Source Manifest

Every file under `corpus/public/` or `corpus/synthetic/` must have a row here. This manifest is the authoritative record of what is real (public-domain) versus synthetic (fictional, authored for this project) — see `SPEC.md`'s Data-Honesty Policy for the policy this manifest implements. `tests/test_corpus_manifest.py` enforces that this table and the corpus directories stay in sync.

All `public` rows are U.S. federal government works and carry no copyright: **U.S. government work, public domain (17 U.S.C. §105).** No copyrighted material is committed anywhere in this corpus.

## Public documents (9)

| File | Title | Revision | Source URL | Retrieved |
|---|---|---|---|---|
| `public/osha-3120-lockout-tagout.md` | Control of Hazardous Energy (Lockout/Tagout) | OSHA 3120, 2002 (Revised) | https://www.osha.gov/sites/default/files/publications/OSHA3120.pdf | 2026-08-23 |
| `public/osha-3170-machine-guarding.md` | Safeguarding Equipment and Protecting Employees from Amputations | OSHA 3170-02R, 2007 | https://www.osha.gov/sites/default/files/publications/OSHA3170.pdf | 2026-08-23 |
| `public/osha-3151-ppe.md` | Personal Protective Equipment | OSHA 3151-02R, 2023 | https://www.osha.gov/sites/default/files/publications/OSHA3151.pdf | 2026-08-23 |
| `public/doe-hdbk-1018-1-pumps.md` | DOE Fundamentals Handbook: Mechanical Science, Vol. 1 of 2 — Module 3: Pumps (excerpt) | DOE-HDBK-1018/1-93, Jan. 1993 | https://www.energy.gov/sites/default/files/2026-04/DOE-HDBK-1018-93_VOL1.pdf | 2026-08-23 |
| `public/doe-hdbk-1018-2-valves.md` | DOE Fundamentals Handbook: Mechanical Science, Vol. 2 of 2 — Module 4: Valves (excerpt) | DOE-HDBK-1018/2-93, Jan. 1993 | https://www.energy.gov/sites/default/files/2026-04/DOE-HDBK-1018-93_VOL2.pdf | 2026-08-23 |
| `public/doe-hdbk-1011-1-motors.md` | DOE Fundamentals Handbook: Electrical Science, Vol. 1 of 4 — Module 2: Basic DC Theory (excerpt) | DOE-HDBK-1011/1-92, June 1992 | https://www.energy.gov/sites/default/files/2026-04/DOE-HDBK-1011-92_VOL1.pdf | 2026-08-23 |
| `public/niosh-pocket-guide-excerpt.md` | NIOSH Pocket Guide to Chemical Hazards — Excerpt: Common Manufacturing Chemicals | DHHS (NIOSH) Pub. No. 2005-149 | https://www.cdc.gov/niosh/docs/2005-149/pdfs/2005-149.pdf | 2026-08-23 |
| `public/cfr-21-part-211-cgmp.md` | 21 CFR Part 211 — Current Good Manufacturing Practice for Finished Pharmaceuticals | Annual CFR ed., title 21 vol. 4, as of 2024-04-01 | https://www.govinfo.gov/content/pkg/CFR-2024-title21-vol4/xml/CFR-2024-title21-vol4-part211.xml | 2026-08-23 |
| `public/cfr-29-1910-1200-hazcom.md` | 29 CFR 1910.1200 — Hazard Communication | Annual CFR ed., title 29 vol. 6, as of 2024-07-01 | https://www.govinfo.gov/content/pkg/CFR-2024-title29-vol6/xml/CFR-2024-title29-vol6-sec1910-1200.xml | 2026-08-23 |

**Note on the two CFR sources**: the interactive eCFR reader (ecfr.gov), which hosts the continuously-updated version of both regulations, blocked automated retrieval (returns a bot-check "Request Access" page to non-browser clients) during ingestion prep. The GPO's official `govinfo.gov` bulk-data XML for the corresponding annual printed CFR edition was used instead — an equally authoritative, equally public-domain government source, just not continuously updated like eCFR. This substitution, and the exact annual edition used, is recorded in each file's `revision` and `source_url_or_note` frontmatter fields.

**Note on excerpts**: the three DOE Fundamentals Handbook files and the NIOSH Pocket Guide file are bounded excerpts of much larger source documents (each DOE handbook module runs 30-70+ printed pages; the NIOSH guide runs ~450 pages covering 677 chemicals). Each excerpt's frontmatter `source_page_range` records exactly what was used. This is disclosed here and in each file's banner — these are not claimed to be the complete source document.

## Synthetic documents (5)

All synthetic documents are original works authored for this portfolio project. They are clearly labeled `source_type: synthetic` in frontmatter and carry an in-file banner stating they are fictional. None reproduce text from any real standard, manual, or facility record.

| File | Title | Purpose |
|---|---|---|
| `synthetic/manual-xj450-belt-conveyor.md` | Model XJ-450 Industrial Belt Conveyor — Maintenance Manual (excerpt) | Fills the "equipment manual" corpus category with a fictional machine, since real manufacturer conveyor manuals are copyrighted and cannot be committed to this repo |
| `synthetic/sop-mnt-022-cnc-mill-changeover.md` | SOP-MNT-022 — Preventive Maintenance Procedure: CNC Mill Tooling Changeover | Illustrates a maintenance SOP typical of a machining cell |
| `synthetic/sop-qa-008-incoming-inspection.md` | SOP-QA-008 — Incoming Material Inspection Procedure | Illustrates a quality/QMS SOP (generic, ISO 9001-style structure — no ISO text reproduced) |
| `synthetic/sop-qa-014-nonconforming-material.md` | SOP-QA-014 — Nonconforming Material Disposition Procedure | Illustrates a quality/QMS SOP; cross-references SOP-QA-008 |
| `synthetic/cmms-work-orders-line3-q2.md` | CMMS Work Order Export — Production Line 3, Q2 2026 (sample) | Sample CMMS work-order records tying equipment failures back to the synthetic manual/SOPs above, echoing the OEE/downtime domain from the PARO project |

## Manifest consistency

- Every file listed above must exist at the stated path.
- Every `.md` file under `corpus/public/` and `corpus/synthetic/` must appear in this manifest with the correct `public`/`synthetic` label matching its frontmatter `source_type`.
- `tests/test_corpus_manifest.py` checks both directions of this consistency automatically.
