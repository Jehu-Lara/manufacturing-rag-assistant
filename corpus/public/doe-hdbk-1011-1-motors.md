---
document_id: doe-hdbk-1011-1-motors
document_title: "DOE Fundamentals Handbook: Electrical Science, Volume 1 of 4 — Module 2: Basic DC Theory — DC Sources, Circuit Terminology, and Circuit Faults (excerpt)"
revision: "DOE-HDBK-1011/1-92, June 1992"
source_type: public
source_url_or_note: "https://www.energy.gov/sites/default/files/2026-04/DOE-HDBK-1011-92_VOL1.pdf"
source_page_range: "PDF pp. 97-118 and 162-165 (Module 2 'Basic DC Theory', handbook-internal pages ES-02 1-22 and 66-69)"
---

# DOE Fundamentals Handbook: Electrical Science, Volume 1 of 4 — Module 2: Basic DC Theory — DC Sources, Circuit Terminology, and Circuit Faults (excerpt)

> Excerpt from a U.S. Department of Energy Fundamentals Handbook. Public domain (U.S. government work, 17 U.S.C. §105). Full document: see source_url_or_note above.
>
> **Note on scope:** Volume 1 of 4 of *Electrical Science* contains only Module 1 (Basic Electrical Theory) and Module 2 (Basic DC Theory). DC Generators (Module 5) and DC Motors (Module 6) are covered in Volume 2 of 4, not in this volume. In keeping with the task's fallback instruction, this excerpt instead draws from Module 2, "Basic DC Theory," selecting the three sections most relevant to real equipment and maintenance work: DC Sources (how DC voltage is actually produced, including by DC generators, thermocouples, and rectifiers), DC Circuit Terminology (the diagram types and circuit conditions used on the job), and DC Circuit Faults (open and short circuit behavior, directly applicable to troubleshooting).

## DC Sources

*When most people think of DC, they usually think of batteries. In addition to batteries, however, there are other devices that produce DC which are frequently used in modern technology.*

### Batteries

A battery consists of two or more chemical cells connected in series. The combination of materials within a battery is used for the purpose of converting chemical energy into electrical energy. To understand how a battery works, we must first discuss the chemical cell.

The chemical cell is composed of two electrodes made of different types of metal or metallic compounds which are immersed in an electrolyte solution. The chemical actions which result are complicated, and they vary with the type of material used in cell construction. Some knowledge of the basic action of a simple cell will be helpful in understanding the operation of a chemical cell in general.

In the cell, electrolyte ionizes to produce positive and negative ions. Simultaneously, chemical action causes the atoms within one of the electrodes to ionize. Due to this action, electrons are deposited on the electrode, and positive ions from the electrode pass into the electrolyte solution. This causes a negative charge on the electrode and leaves a positive charge in the area near the electrode.

The positive ions, which were produced by ionization of the electrolyte, are repelled to the other electrode. At this electrode, these ions will combine with the electrons. Because this action causes removal of electrons from the electrode, it becomes positively charged.

### DC Generator

A simple DC generator consists of an armature coil with a single turn of wire. The armature coil cuts across the magnetic field to produce a voltage output. As long as a complete path is present, current will flow through the circuit. In one coil position, commutator segment 1 contacts brush 1, while commutator segment 2 is in contact with brush 2.

Rotating the armature one-half turn in the clockwise direction causes the contacts between the commutator segments to be reversed. Now segment 1 is contacted by brush 2, and segment 2 is in contact with brush 1.

Due to this commutator action, that side of the armature coil which is in contact with either of the brushes is always cutting the magnetic field in the same direction. Brushes 1 and 2 have a constant polarity, and pulsating DC is delivered to the load circuit.

### Thermocouples

A thermocouple is a device used to convert heat energy into a voltage output. The thermocouple consists of two different types of metal joined at a junction.

As the junction is heated, the electrons in one of the metals gain enough energy to become free electrons. The free electrons will then migrate across the junction and into the other metal. This displacement of electrons produces a voltage across the terminals of the thermocouple. The combinations used in the makeup of a thermocouple include: iron and constantan; copper and constantan; antimony and bismuth; and chromel and alumel.

Thermocouples are normally used to measure temperature. The voltage produced causes a current to flow through a meter, which is calibrated to indicate temperature.

### Rectifiers

Most electrical power generating stations produce alternating current. The major reason for generating AC is that it can be transferred over long distances with fewer losses than DC; however, many of the devices which are used today operate only, or more efficiently, with DC. For example, transistors, electron tubes, and certain electronic control devices require DC for operation. If we are to operate these devices from ordinary AC outlet receptacles, they must be equipped with rectifier units to convert AC to DC. In order to accomplish this conversion, we use diodes in rectifier circuits. The purpose of a rectifier circuit is to convert AC power to DC.

The most common type of solid state diode rectifier is made of silicon. The diode acts as a gate, which allows current to pass in one direction and blocks current in the other direction. The polarity of the applied voltage determines if the diode will conduct. The two polarities are known as forward bias and reverse bias.

#### Forward Bias

A diode is forward biased when the positive terminal of a voltage source is connected to its anode, and the negative terminal is connected to the cathode. The power source's positive side will tend to repel the holes in the p-type material toward the p-n junction; the negative side repels electrons toward the junction. A hole is a vacancy in the electron structure of a material. Holes behave as positive charges. As the holes and the electrons reach the p-n junction, some of them break through it. Holes combine with electrons in the n-type material, and electrons combine with holes in the p-type material.

When a hole combines with an electron, or an electron combines with a hole near the p-n junction, an electron from an electron-pair bond in the p-type material breaks its bond and enters the positive side of the source. Simultaneously, an electron from the negative side of the source enters the n-type material. This produces a flow of electrons in the circuit.

#### Reverse Bias

Reverse biasing occurs when the diode's anode is connected to the negative side of the source, and the cathode is connected to the positive side of the source. Holes within the p-type material are attracted toward the negative terminal, and the electrons in the n-type material are attracted to the positive terminal. This prevents the combination of electrons and holes near the p-n junction, and therefore causes a high resistance to current flow. This resistance prevents current flow through the circuit.

#### Half-Wave Rectifier Circuit

When a diode is connected to a source of alternating voltage, it will be alternately forward-biased, and then reverse-biased, during each cycle of the AC sine-wave. When a single diode is used in a rectifier circuit, current will flow through the circuit only during one-half of the input voltage cycle. For this reason, this rectifier circuit is called a half-wave rectifier. The output of a half-wave rectifier circuit is pulsating DC.

#### Full-Wave Rectifier Circuit

A full-wave rectifier circuit is a circuit that rectifies the entire cycle of the AC sine-wave. A basic full-wave rectifier uses two diodes, one conducting during each half cycle.

Another type of full-wave rectifier circuit is the full-wave bridge rectifier. This circuit utilizes four diodes. The output of this circuit then becomes a pulsating DC, with all of the waves of the input AC being transferred. The output looks identical to that obtained from a (two-diode) full-wave rectifier.

### Summary

The important information concerning DC sources is summarized below.

**DC Sources Summary**

There are four common ways that DC voltages are produced:
- Batteries
- DC Generators
- Thermocouples
- Rectifiers

Thermocouples convert energy from temperature into a DC voltage. This voltage can be used to measure temperature.

A rectifier converts AC to DC. There are two types of rectifiers:
- Half-Wave rectifiers
- Full-Wave rectifiers

Half-wave rectifiers convert the AC to a pulsating DC and convert only one-half of the sine wave. Full-wave rectifiers convert the AC to a pulsating DC and convert all of the sine wave.

## DC Circuit Terminology

Before operations with DC circuits can be studied, an understanding of the types of circuits and common circuit terminology associated with circuits is essential.

### Schematic Diagram

Schematic diagrams are the standard means by which we communicate information in electrical and electronics circuits. On schematic diagrams, the component parts are represented by graphic symbols. Because graphic symbols are small, it is possible to have diagrams in a compact form. The symbols and associated lines show how circuit components are connected and the relationship of those components with one another.

As an example, a schematic diagram of a two-transistor radio circuit, from left to right, shows the components in the order they are used to convert radio waves into sound energy. By using this diagram it is possible to trace the operation of the circuit from beginning to end. Due to this important feature of schematic diagrams, they are widely used in construction, maintenance, and servicing of all types of electronic circuits.

### One-Line Diagram

The one-line, or single-line, diagram shows the components of a circuit by means of single lines and the appropriate graphic symbols. One-line diagrams show two or more conductors that are connected between components in the actual circuit. The one-line diagram shows all pertinent information about the sequence of the circuit, but does not give as much detail as a schematic diagram. Normally, the one-line diagram is used to show highly complex systems without showing the actual physical connections between components and individual conductors. A typical example is a one-line diagram of an electrical substation.

### Block Diagram

A block diagram is used to show the relationship between component groups, or stages in a circuit. In block form, it shows the path through a circuit from input to output. The blocks are drawn in the form of squares or rectangles connected by single lines with arrowheads at the terminal end, showing the direction of the signal path from input to output. Normally, the necessary information to describe the stages of components is contained in the blocks.

### Wiring Diagram

A wiring diagram is a very simple way to show wiring connections in an easy-to-follow manner. These types of diagrams are normally found with home appliances and automobile electrical systems. Wiring diagrams show the component parts in pictorial form, and the components are identified by name. Most wiring diagrams also show the relative location of component parts and color coding of conductors or leads.

### Resistivity

Resistivity is defined as the measure of the resistance a material imposes on current flow. The resistance of a given length of conductor depends upon the resistivity of that material, the length of the conductor, and the cross-sectional area of the conductor, according to the equation:

**R = ρL / A**

where
- R = resistance of conductor, Ω (ohms)
- ρ (rho) = specific resistance or resistivity, cm-Ω/ft
- L = length of conductor, ft
- A = cross-sectional area of conductor, cm

The resistivity ρ allows different materials to be compared for resistance, according to their nature, without regard to length or area. The higher the value of ρ, the higher the resistance.

The following table gives resistivity values for metals having the standard wire size of one foot in length and a cross-sectional area of 1 cm² (values at 20°C, in cm-Ω/ft; precise values depend on exact composition of material):

| Material | Resistivity (ρ) at 20°C, cm-Ω/ft |
|---|---|
| Aluminum | 17 |
| Carbon | (has 2500–7500 times the resistance of copper) |
| Constantan | 295 |
| Copper | 10.4 |
| Gold | 14 |
| Iron | 58 |
| Nichrome | 676 |
| Nickel | 52 |
| Silver | 9.8 |
| Tungsten | 33.8 |

### Temperature Coefficient of Resistance

Temperature coefficient of resistance, α (alpha), is defined as the amount of change of the resistance of a material for a given change in temperature. A positive value of α indicates that R increases with temperature; a negative value of α indicates R decreases; and zero α indicates that R is constant. Typical values:

| Material | Temperature Coefficient, Ω per °C |
|---|---|
| Aluminum | 0.004 |
| Carbon | -0.0003 |
| Constantan | 0 (avg) |
| Copper | 0.004 |
| Gold | 0.004 |
| Iron | 0.006 |
| Nichrome | 0.0002 |
| Nickel | 0.005 |

For a given material, α may vary with temperature; therefore, charts are often used to describe how resistance of a material varies with temperature. An increase in resistance can be approximated from the equation R = R₀[1 + α(ΔT)], where ΔT is the temperature rise above 20°C.

### Electric Circuit

Each electrical circuit has at least four basic parts: (1) a source of electromotive force, (2) conductors, (3) load or loads, and (4) some means of control. For example, in a simple circuit the source of EMF is the battery; the conductors are wires which connect the various component parts; the resistor is the load; and a switch is used as the circuit control device.

A **closed circuit** is an uninterrupted, or unbroken, path for current from the source (EMF), through the load, and back to the source.

An **open circuit**, or incomplete circuit, exists if a break in the circuit occurs; this prevents a complete path for current flow.

A **short circuit** is a circuit which offers very little resistance to current flow and can cause dangerously high current flow through a circuit. Short circuits are usually caused by an inadvertent connection between two points in a circuit which offers little or no resistance to current flow. Shorting a resistor in a circuit will probably cause the fuse to blow.

### Series Circuit

A series circuit is a circuit where there is only one path for current flow. In a series circuit, the current will be the same throughout the circuit. This means that the current flow through R₁ is the same as the current flow through R₂ and R₃.

### Parallel Circuit

Parallel circuits are those circuits which have two or more components connected across the same voltage source. Resistors R₁, R₂, and R₃ are in parallel with each other and the source. Each parallel path is a branch with its own individual current. When the current leaves the source V, part I₁ of I_T will flow through R₁; part I₂ will flow through R₂; and part I₃ will flow through R₃. Current through each branch can be different; however, voltage throughout the circuit will be equal:

**V = V₁ = V₂ = V₃**

### Equivalent Resistance

In a parallel circuit, the total resistance of the resistors in parallel is referred to as equivalent resistance. This can be described as the total circuit resistance as seen by the voltage source. In all cases, the equivalent resistance will be less than any of the individual parallel circuit resistors. Using Ohm's Law, equivalent resistance (R_EQ) can be found by dividing the source voltage (V) by the total circuit current (I_T):

**R_EQ = V / I_T**

### Summary

The important information concerning basic DC circuits is summarized below.

**DC Circuit Terminology Summary**

There are four types of circuit diagrams:
- Schematic diagram
- One-line diagram
- Block diagram
- Wiring diagram

Resistivity is defined as the measure of the resistance a material imposes on current flow.

Temperature coefficient of resistance, α (alpha), is defined as the amount of change of the resistance of a material for a given change in temperature.

A closed circuit is one that has a complete path for current flow. An open circuit is one that does not have a complete path for current flow. A short circuit is a circuit with a path that has little or no resistance to current flow.

A series circuit is one where there is only one path for current flow. A parallel circuit is one which has two or more components connected across the same voltage source. Equivalent resistance is the total resistance of the resistors in parallel.

## DC Circuit Faults

Faults within a DC circuit will cause various effects, depending upon the nature of the fault. An understanding of the effects of these faults is necessary to fully understand DC circuit operation.

### Open Circuit (Series)

A circuit must have a "complete" path for current flow, that is, from the negative side to the positive side of a power source. A series circuit has only one path for current to flow. If this path is broken, no current flows, and the circuit becomes an open circuit.

Since no current flows in an open series circuit, there are no voltage drops across the loads. No power is consumed by the loads, and total power consumed by the circuit is zero.

### Open Circuit (Parallel)

A parallel circuit has more than one path for current to flow. If one of the paths is opened, current will continue to flow as long as a complete path is provided by one or more of the remaining paths. It does not mean that you cannot stop current flow through a parallel circuit by opening it at one point; it means that the behavior of a parallel circuit depends on where the opening occurs.

If a parallel circuit is opened at a point where only a branch current flows, then only that branch is open, and current continues to flow in the rest of the circuit.

### Short Circuit (Series)

In a DC circuit, the only current limit is the circuit resistance. If there is no resistance in a circuit, or if the resistance suddenly becomes zero, a very large current will flow. This condition of very low resistance and high current flow is known as a "short circuit."

A short circuit is said to exist if the circuit resistance is so low that current increases to a point where damage can occur to circuit components. With an increase in circuit current flow, the terminal voltage of the energy source will decrease. This occurs due to the internal resistance of the energy source causing an increased voltage drop within the energy source. The increased current flow resulting from a short circuit can damage power sources, burn insulation, and start fires. Fuses are provided in circuits to protect against short circuits.

### Short Circuit (Parallel)

When a parallel circuit becomes short circuited, the same effect occurs as in a series circuit: there is a sudden and very large increase in circuit current. Parallel circuits are more likely than series circuits to develop damaging short circuits. This is because each load is connected directly across the power source. If any of the loads becomes shorted, the resistance between the power source terminals is practically zero. If a series load becomes shorted, the resistance of the other loads keeps the circuit resistance from dropping to zero.

### Summary

The important information in this chapter is summarized below.

**DC Circuit Faults Summary**

An open series DC circuit will result in no power being consumed by any of the loads.

The effect of an open in a parallel circuit is dependent upon the location of the open.

A shorted DC circuit will result in a sudden and very large increase in circuit current.
