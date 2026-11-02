# Module 3: Physics-Grounded Sensor Criticality Analysis & Minimal Sensor Set

**Clean Baseline Boundary Error:** `0.93 cm`  
**Total Available Diagnostic Sensors:** `47` probes across 11 diagnostic subsystems.  
**Safety Abort Tolerance:** `2.00 cm` (Displacement exceeding 2.0 cm risks uncontained wall contact).

## 1. Physics-Grounded Sensor Criticality Ranking (Centimeter Impact)

| Rank | Diagnostic Subsystem | Category | Probes | Boundary Error (cm) | Boundary Shift (cm) | Axis Shift (cm) | Safety Tier | Failure Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **solenoid_current** (Central solenoid ohmic current) | `solenoid` | 1 | `2.04` | **`+1.11`** | `0.80` | `Tier 1: Safety-Critical` | HIGH (Wall breach / disruption risk) |
| 2 | **pickup_probe_ccbv** (Center column vertical B-field probes) | `pickup_fields` | 8 | `1.86` | **`+0.94`** | `0.68` | `Tier 2: Plasma Shaping` | MODERATE (Loss of elongation / triangularity control) |
| 3 | **pickup_probe_obr** (Outer vessel radial B-field probes) | `pickup_fields` | 6 | `1.73` | **`+0.80`** | `0.57` | `Tier 2: Plasma Shaping` | MODERATE (Loss of elongation / triangularity control) |
| 4 | **pickup_probe_obv** (Outer vessel vertical B-field probes) | `pickup_fields` | 6 | `1.73` | **`+0.80`** | `0.57` | `Tier 2: Plasma Shaping` | MODERATE (Loss of elongation / triangularity control) |
| 5 | **active_coils_p3_p6** (Divertor shaping coils P3-P6) | `active_coils` | 4 | `1.42` | **`+0.49`** | `0.35` | `Tier 3: Auxiliary / Redundant` | LOW (Safe to impute or drop) |
| 6 | **active_coils_p1_p2** (Poloidal shaping coils P1/P2) | `active_coils` | 2 | `1.36` | **`+0.43`** | `0.31` | `Tier 3: Auxiliary / Redundant` | LOW (Safe to impute or drop) |
| 7 | **flux_loop_divertor** (Upper/lower divertor flux loops) | `flux_loop` | 5 | `1.31` | **`+0.38`** | `0.27` | `Tier 3: Auxiliary / Redundant` | LOW (Safe to impute or drop) |
| 8 | **plasma_current_ip** (Rogowski coil total plasma current) | `plasma_current` | 1 | `1.29` | **`+0.37`** | `0.26` | `Tier 3: Auxiliary / Redundant` | LOW (Safe to impute or drop) |
| 9 | **flux_loop_midplane** (Wall midplane flux loops) | `flux_loop` | 5 | `1.26` | **`+0.33`** | `0.24` | `Tier 3: Auxiliary / Redundant` | LOW (Safe to impute or drop) |
| 10 | **flux_loop_vessel** (Outboard vessel wall flux loops) | `flux_loop` | 5 | `1.26` | **`+0.33`** | `0.24` | `Tier 3: Auxiliary / Redundant` | LOW (Safe to impute or drop) |
| 11 | **saddle_voltage** (Toroidal saddle loop voltages) | `saddle_voltage` | 4 | `0.93` | **`+0.01`** | `0.00` | `Tier 3: Auxiliary / Redundant` | LOW (Safe to impute or drop) |

## 2. Minimal Sensor Set Optimization (Design Solution)

- **Initial Diagnostic Complement:** `47 probes`
- **Optimal Minimal Sensor Set:** **`25 probes`**
- **Diagnostic Sensor Reduction:** **`46.8% reduction`** while strictly obeying `< 2.0 cm` boundary clearance!
- **Final Boundary Error at Minimal Set:** `1.87 cm` (Safe confinement maintained).

### Retained Core Subsystems in Minimal Set:
- **`solenoid_current`** (Essential for core current profile & X-point tracking)
- **`pickup_probe_ccbv`** (Essential for core current profile & X-point tracking)
- **`pickup_probe_obr`** (Essential for core current profile & X-point tracking)
- **`pickup_probe_obv`** (Essential for core current profile & X-point tracking)
- **`active_coils_p3_p6`** (Essential for core current profile & X-point tracking)

### Safely Deprecated / Redundant Subsystems:
- `~saddle_voltage~` (Can be imputed or substituted without breaching 2.0 cm tolerance)
- `~flux_loop_midplane~` (Can be imputed or substituted without breaching 2.0 cm tolerance)
- `~flux_loop_vessel~` (Can be imputed or substituted without breaching 2.0 cm tolerance)
- `~plasma_current_ip~` (Can be imputed or substituted without breaching 2.0 cm tolerance)
- `~flux_loop_divertor~` (Can be imputed or substituted without breaching 2.0 cm tolerance)
- `~active_coils_p1_p2~` (Can be imputed or substituted without breaching 2.0 cm tolerance)

### Pruning Progression Curve:

| Step | Removed Diagnostic | Remaining Probes | Boundary Error (cm) | Safety Status |
| :--- | :--- | :--- | :--- | :--- |
| 0 | None (Full Diagnostic Set) | 47 (100.0%) | `0.92 cm` | **SAFE** |
| 1 | saddle_voltage (4 probes) | 43 (91.5%) | `1.04 cm` | **SAFE** |
| 2 | flux_loop_midplane (5 probes) | 38 (80.9%) | `1.19 cm` | **SAFE** |
| 3 | flux_loop_vessel (5 probes) | 33 (70.2%) | `1.34 cm` | **SAFE** |
| 4 | plasma_current_ip (1 probes) | 32 (68.1%) | `1.50 cm` | **SAFE** |
| 5 | flux_loop_divertor (5 probes) | 27 (57.4%) | `1.67 cm` | **SAFE** |
| 6 | active_coils_p1_p2 (2 probes) | 25 (53.2%) | `1.87 cm` | **SAFE** |

## 3. Visualization Artifacts

- **Spatial Error Propagation Heatmaps:** `results/robustness/figures/spatial_error_propagation_heatmaps.png`
- **Diagnostic Pruning vs Boundary Error Curve:** `results/robustness/figures/minimal_sensor_set_curve.png`
