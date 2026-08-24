---
document_id: manual-xj450-belt-conveyor
document_title: "Model XJ-450 Industrial Belt Conveyor — Maintenance Manual (excerpt)"
revision: "Fictional Rev. 2, 2024 (synthetic)"
source_type: synthetic
source_url_or_note: "Synthetic document authored for this portfolio project — not a real facility record. The 'XJ-450' model, its manufacturer, and all specification values are fictional and do not describe any real product."
source_page_range: null
---

# Model XJ-450 Industrial Belt Conveyor — Maintenance Manual (excerpt)

> **SYNTHETIC DOCUMENT.** This is a fictional equipment manual authored for this portfolio project to demonstrate manufacturer-manual-style content where no equivalent public-domain manual exists. The "XJ-450" conveyor, its manufacturer, and every specification, torque value, and part number below are invented for this exercise and must not be treated as real product data.

## 1. General Description

The Model XJ-450 is a fixed-frame, belt-driven material handling conveyor rated for continuous-duty transport of packaged goods and bulk materials up to 45 kg per linear meter. The unit consists of a welded steel frame, a drive pulley assembly at the discharge end, a tail pulley assembly at the load end, a rubber-covered fabric belt, and a variable-frequency-drive (VFD) controlled 3-phase induction motor.

Standard lengths range from 3 m to 24 m in 1.5 m increments; belt widths are available in 500 mm, 650 mm, and 800 mm. The belt travels at a nominal speed of 0.2-1.8 m/s, adjustable via the VFD.

## 2. Major Components

| Component | Function |
|---|---|
| Drive pulley | Driven by the gearmotor; imparts belt motion via friction |
| Tail pulley | Idler pulley at the load end; maintains belt tension path |
| Snub pulleys | Increase belt wrap angle on the drive pulley to improve traction |
| Take-up assembly | Screw-adjusted tensioning device at the tail end |
| Idler rollers | Support the belt's carrying and return runs along the frame |
| Belt cleaner (scraper) | Removes carryback material from the belt after discharge |
| VFD control panel | Motor speed control, start/stop, and fault indication |

## 3. Preventive Maintenance Schedule

### 3.1 Daily (operator-level) checks

- Visually inspect the belt for tracking (should run centered on the pulleys with no more than 10 mm lateral drift at any point).
- Confirm belt cleaner blade is in contact with the belt and not excessively worn.
- Listen for unusual noise from the drive pulley bearings or gearmotor.
- Verify emergency stop cords (where fitted) move freely and are not obstructed.

### 3.2 Monthly (maintenance technician) checks

- Lubricate drive and tail pulley bearings per the lubrication schedule (Section 4).
- Inspect belt splice condition; report any fraying, separation, or exposed fabric ply to the shift supervisor.
- Check take-up assembly travel; if the take-up screw is within 25 mm of its maximum travel, schedule a belt-shortening/re-splice job.
- Inspect idler rollers for freedom of rotation; replace any roller that does not spin freely by hand within 2 rotations of coast-down.
- Torque-check drive pulley shaft coupling bolts to 85 N·m.

### 3.3 Annual (or 4,000 operating-hour) overhaul

- Full belt inspection and, if wear exceeds 40% of original cover thickness, belt replacement.
- Gearmotor oil change (see gearmotor OEM documentation, not included in this excerpt).
- Full disassembly and inspection of drive pulley bearings; replace if radial play exceeds 0.15 mm.
- VFD parameter backup and firmware version check.

## 4. Lubrication Schedule

| Point | Lubricant | Interval | Quantity |
|---|---|---|---|
| Drive pulley bearings | NLGI Grade 2 lithium complex grease | Monthly | 15 g per bearing |
| Tail pulley bearings | NLGI Grade 2 lithium complex grease | Monthly | 15 g per bearing |
| Take-up screw threads | Light machine oil | Quarterly | Light film |
| Idler roller bearings | Sealed-for-life — no relubrication | N/A | N/A |

Over-greasing bearings is a common root cause of premature seal failure on the XJ-450; do not exceed the specified quantity.

## 5. Belt Tracking Troubleshooting

Belt mistracking is the most common XJ-450 fault reported through the CMMS. Use the following diagnostic sequence:

1. **Confirm frame level.** A conveyor frame out of level by more than 3 mm/m will cause persistent mistracking that cannot be corrected by pulley adjustment alone.
2. **Check pulley alignment.** Both drive and tail pulleys must be square to the frame centerline within 1 mm across the pulley face. Use a string line or laser alignment tool.
3. **Inspect for uneven loading.** Material consistently loaded off-center will walk the belt toward the heavier side over time.
4. **Adjust the tail pulley take-up.** If the belt walks toward one side consistently, a small adjustment (2-3 mm) to the take-up on the side the belt is walking *away from* will typically correct tracking within a few minutes of run time. Make one adjustment at a time and observe at least 3 full belt revolutions before making another.
5. **Inspect snub pulleys and idlers** for buildup of material or damage that could be steering the belt locally.

## 6. Safety Notes

Lockout/tagout procedures per the facility's energy control program (see the facility SOP for lockout/tagout, and OSHA 3120 for general guidance) must be followed before any belt cleaner adjustment, pulley work, or work inside the guard rails. Never reach into the nip point between the belt and any pulley while the conveyor is capable of being energized. Belt cleaner blades and pulley nip points are the two highest-frequency injury points reported for this equipment class; guards removed for maintenance must be reinstalled and verified secure before the conveyor is returned to service.
