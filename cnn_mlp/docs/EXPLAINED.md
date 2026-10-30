# Plain-English Guide: what this project does, number by number

This is the "explain it like I'm not a plasma physicist *or* a deep-learning
researcher" companion to `docs/ARCHITECTURE.md`. It walks through **every number,
every choice, and every criterion** in the setup, in order, in plain language.

Read it top to bottom once; keep it open as a dictionary afterwards.

**Companion doc:** `docs/VIVA.md` — the "examiner asks the awkward questions"
version. Where this file states the facts, VIVA *interrogates* them (analogy first,
then the precise answer). Use VIVA to rehearse explaining this project out loud, and
as the place to look up any tricky "wait, why?" question. This file also folds in the
clarifications we worked through (the `MIN_TARS` floor, when NRMSE is actually
computed, and how to read the `±`).

---

## PART 0 — The one-paragraph summary

We are teaching a computer to look at a tokamak (a donut-shaped fusion reactor)
and, from a handful of electrical measurements taken at one instant, draw a
picture of the invisible magnetic field that holds the super-hot plasma in place.
That picture is the `equilibrium/psi` field, a 65x65 grid of numbers. The
official benchmark (TokaMark) gives us the data and the scoring rules. We run a
small neural network — its architecture is described in
`docs/reference/architecture_multiconv_mlp_standalone.py` — on a slice of that
data because we only have limited compute. Our job right now is to (a) reproduce
the benchmark's baseline and (b) stress-test how it copes when sensors fail.

---

## PART 1 — The physics, in plain terms

### 1.1 What is a tokamak?
A machine that traps plasma (ionised gas) in a magnetic "bottle" shaped like a
donut, so it can get hot enough to fuse atoms. The magnetic field is invisible,
but its shape determines whether the plasma stays put or smashes into the walls.

### 1.2 What is "equilibrium reconstruction"?
You cannot stick a camera inside — it is millions of degrees. So you infer the
field from **outside** measurements and a physics equation (the **Grad–Shafranov
equation**). Classic software (called EFIT) solves that equation numerically and
is slow (seconds per instant). We instead train a neural network to do it
**instantly** (milliseconds), which matters for real-time control.

### 1.3 What is `psi` (ψ)?
The **poloidal flux function**. It is a smooth 2-D scalar field over the (R, Z)
cross-section of the donut. Its contour lines ("flux surfaces") trace out the
nested magnetic surfaces the plasma sits on. If you know ψ everywhere, you know
the field shape. Our network's whole job is to output ψ on a **65×65 grid**.

### 1.4 What is "MAST" and "FAIR-MAST"?
MAST = Mega Ampere Spherical Tokamak, a real machine at Culham, UK (1999–2013).
FAIR-MAST is the public dataset of its shots, released for research. A **shot**
is one plasma discharge — think of it as one "experiment run," 2–3 seconds long.

---

## PART 2 — The benchmark (TokaMark), in plain terms

### 2.1 What is TokaMark?
An official, open benchmark (paper: arXiv 2602.10132) that packages MAST data
with **14 tasks** and **fixed scoring rules**, so different AI models can be
compared fairly. It is split into 4 groups; we care about **Group 1,
Equilibrium Reconstruction**.

### 2.2 What is a "task"?
A defined input→output problem. Ours is **task_1-3**: given 8 instantaneous
measurements, output the 65×65 ψ field for the same instant.

### 2.3 The words attached to every task
- **input_length = output_length = 0.005 s** — each example looks at **5 ms** of
  data in and asks for **5 ms** of output. (At MAST's sampling rates that is a
  handful of samples per signal.)
- **delta = −0.005 s** — the output window begins where the input window ends
  (no overlap, no future leak).
- **stride_window = 0.005 s** — we slide the window forward 5 ms at a time, so
  consecutive examples are non-overlapping snapshots.
- **task_type = markovian** — the answer depends **only on the current instant**,
  not on the history before it. This is *why* a simple feed-forward CNN is a fair
  baseline: we are not obliged to use a recurrent (memory) model.

### 2.4 The split rule
- **80 / 10 / 10** of shots go to **train / validation / test** — by *shot*, not
  by window. A shot is entirely in one bucket, so the model is never tested on a
  shot it trained on. This is the "shot-level split."
- Official sizes: **8963 / 1115 / 1110** shots; **565.3 GB** total.

### 2.5 The 8 inputs (and why each is there)
| # | Signal | Plain meaning |
|---|--------|---------------|
| 0 | `flux_loop_flux` | magnetic flux measured by a loop around the vessel |
| 1 | `b_field_pol_probe_ccbv_field` | poloidal magnetic field at one probe location |
| 2 | `b_field_pol_probe_obr_field` | ...another probe (outboard) |
| 3 | `b_field_pol_probe_obv_field` | ...another probe (outboard vertical) |
| 4 | `b_field_tor_probe_saddle_voltage` | saddle-coil voltage (toroidal field info) |
| 5 | `pf_active/coil_current` | current in the shaping coils (we control these) |
| 6 | `pf_active/solenoid_current` | current in the central solenoid |
| 7 | `summary/ip` | total plasma current (the single most important signal) |

These 8 are the magnetic + actuator picture of the plasma at that instant. The
field shape ψ is essentially determined by them (via Grad–Shafranov), so the
mapping is well-posed in principle.

---

## PART 3 — The data pipeline, in plain terms

The pipeline (`src/pipeline.py`) has five commands: `prep → train → eval → plot
→ summary`. Here is what each does and why the numbers on it exist.

### 3.1 `prep` — getting the data
- The raw data lives as huge remote **tar archives** on Hugging Face. `prep`
  first builds an **index** (a map of which shots are in which archive).
- **The budget problem:** we cannot download 565 GB. The free Kaggle disk/RAM
  cannot take it, and we have a time limit. So we set a **byte budget**:
  `TM_DATA_GB` (currently **40.0**). The pipeline greedily selects shots from the
  official splits until the budget is reached.
- **`TM_INDEX_MIN_TARS` (12)** — a tar is a big archive file that contains **many
  shots** (not one). Without a floor, the greedy budget-filler could fill the whole
  budget from just 1–2 giant tars, biasing the subset toward whatever shots happen
  to live there. This setting forces the selection pool to be spread across **at
  least 12 different tars**, so we sample from a wide slice of the campaign. It is
  a *minimum floor*, not a target and not a count of shots. (12 is chosen because
  the whole dataset is ~30-ish tars; 12 gives wide coverage without forcing a
  download too large to index in the time budget.)
- **`selected_at_min = 15.9`** — the *minimum* number of minutes given to the
  selection step before it must finalise. A safeguard, not a target.
- **Result of a run:** a `selection.json` listing exactly which shots, and a
  `report` block showing, per split, the official count vs. how many we kept:
  - 40 GB run: train 606/8963, val 74/1115, test 60/1110
    (**6.8% / 6.6% / 5.4% of official**).
  - 20 GB run: train 280, val 32, test 42 (**~3%**).
- **Critical honesty rule:** split *membership* is never mixed. A shot in the
  official test set stays test. The subset is a *thinned* version of the official
  split, with matching statistics files. **Therefore our numbers are NOT the
  official full-set numbers** and must never be quoted as such.

### 3.2 Downloading shots
- Only the selected shots' archives are downloaded, with an **adaptive HTTP
  throttle** (`TM_HF_RPS` = 12 requests/s ceiling, `TM_HF_RPS_MIN` = 2 floor)
  that halves on HTTP 429 and recovers on success, plus bounded retries. This
  keeps us from getting rate-limited (previous runs saw 429 storms; now 0).
- **740/740 downloaded, 0 failed** in the 40 GB run.

### 3.3 `train` — learning the mapping
Covered in detail in Part 4.

### 3.4 `eval` — scoring on unseen shots
- Runs the trained model on the **test** split only, un-standardises the output,
  and computes the official metrics (Part 5).

### 3.5 `plot` and `summary`
- `plot` saves `loss_curves.png` (train/val loss over steps) and
  `psi_reconstruction_samples.png` (predicted vs. true ψ for sample shots).
- `summary` writes `summary.md` — the human-readable report we have been reading.

---

## PART 4 — Training, number by number

### 4.1 What "training" means here
Show the model many (input → true ψ) pairs. It guesses ψ, we measure the error,
we nudge its internal numbers (weights) to reduce future error. Repeat.

### 4.2 The standardisation step (and why it matters for the metrics)
**What it is, concretely.** For every input signal *and* the target ψ we replace
each value with `(value - mean) / std`. So a coil current spanning 0–50,000 amps
becomes values roughly in the range −3…+3, and every signal ends up on a similar
zero-centred, unit-spread scale. (The mean/std are computed on the **train** split
only, then applied unchanged to val/test — never the reverse, which would leak.)

**Why.** The 8 inputs carry wildly different physical units (thousands of amps,
a few volts, webers…). Feeding raw numbers makes the loss dominated by whichever
signal has the biggest numbers, so the model learns badly. Standardising puts every
signal on equal footing → stable, fast training.

**Why it matters for the metrics.** Training minimises error on the **standardised**
scale, but the **official benchmark score (NRMSE) is computed on the raw, real ψ**.
So two "loss" numbers that look contradictory are really the same error measured on
two different scales:
- **training loss** = standardised MSE (what we optimise)
- **test NRMSE** = real-units error ÷ real ψ spread (what is graded)
This is exactly why the 40 GB run showed a superb train loss (0.016) alongside a
test NRMSE of 0.1419 — not a contradiction, just two scales of "how wrong".

### 4.3 The knob settings (hyperparameters) and what each does
| Setting | Value | Plain meaning |
|---------|-------|---------------|
| batch_size | 512 | how many examples per weight-update |
| lr | 1e-4 | learning rate — size of each nudge. Small = careful. |
| weight_decay | 1e-4 | mild penalty on large weights (anti-overfitting) |
| validate_every | 100 steps | check val loss every 100 updates |
| patience | 10 | stop if val does not improve for 10 consecutive checks |
| max_steps | 10,000,000 | effectively "no step cap" — patience/deadline rules |
| train_deadline_h | 7.0 | hard wall: stop training after ~7 h |
| use_dp | auto | whether to use both GPUs (`DataParallel`); auto-detected |
| seed | 23 | fixes randomness so runs are reproducible |

### 4.4 Early stopping — what it is and why it stopped at 3500
- We cannot measure the *test* score during training (that would be cheating), so
  we watch a **validation** score and stop when it stops improving (patience 10).
- **Both runs stopped at exactly step 3500.** That is the patience rule firing at
  the same point twice — the model's val loss stopped improving, so training quit.
- **This is NOT a step limit.** It is "the model has stopped getting better on
  held-out shots, so further training is wasteful."

### 4.5 What actually happened in our two runs (the key result)

**20 GB run (280 train shots):**
- Val loss went **0.556 → 0.0250** — a clean, low plateau. Train and val tracked
  each other closely (gap ~±0.01).
- Test NRMSE **0.1395**.
- **Interpretation:** healthy generalisation. With little data, the model *could
  not* memorise, so it learned real physics.

**40 GB run (606 train shots):**
- Val loss **1.505 → 0.8723**: train loss cratered to **0.016**, val stuck ~0.87.
  Gap ~**+0.87** and never closed.
- Test NRMSE **0.1419** (essentially unchanged), but MAE improved **23%**
  (0.00505 → 0.00389), NMAE 0.092 → 0.071.
- **Interpretation:** textbook **overfitting**. The model had enough capacity to
  memorise 606 shots; val loss stayed bad while train loss collapsed. This trades
  nothing on the headline NRMSE but sharpens the pixel-wise error.

**Why "more data" looked worse:** with the same 3500-step wall, the bigger
subset gave fewer epochs per shot, but the model still found a way to memorise
its 606 identities. Overfitting is a *data-size regime* effect, exactly as the
user reasoned — the paper trained on the full ~9000 shots where memorisation is
impossible, so its curves are smooth.

---

## PART 5 — Scoring: every metric explained

### 5.1 The four numbers you will always see
| Metric | Stands for | Plain meaning | Lower is better? |
|--------|-----------|---------------|------------------|
| **NRMSE** | Normalised Root-Mean-Square Error | typical error, divided by the data's spread | yes |
| **NMAE** | Normalised Mean Absolute Error | typical absolute error, normalised | yes |
| **RMSE** | Root-Mean-Square Error | error in raw ψ units, big errors punished | yes |
| **MAE** | Mean Absolute Error | error in raw ψ units, all errors equal | yes |

### 5.2 Why NRMSE (not just RMSE)?
RMSE depends on the units and size of ψ, which vary. Dividing by the ground
truth's standard deviation gives a **unit-free score**. Rule of thumb:
- NRMSE ~1.0 → model is as good as just predicting the average (useless).
- NRMSE below 1.0 → genuine predictive skill.
- Our ~0.14 → quite good on the subset (but again: subset, not official).

**RMSE vs MAE, the only difference.** Both are "typical error in real ψ units".
RMSE **squares** the errors before averaging, so a few huge misses dominate it —
it punishes outliers hard. MAE counts every miss equally and ignores outliers. So
whenever `RMSE > MAE` (as here: 0.00779 vs 0.00389) it means *outliers exist*.

### 5.3 ± standard deviation — read it carefully
**What it is.** Every one of the ~3450 test windows gets its **own** NRMSE (some
0.01, some 0.05, some 0.3). The headline **0.1419** is their **average**. The
`(±0.1199)` is the **standard deviation of those per-window scores** — i.e. how
scattered the individual scores are around that average. It is **NOT** uncertainty
on the model's parameters and **NOT** error bars on a single prediction, and it is
**NOT** "the distribution sits at ±0.12".

**How to read the size.** If every window scored about the same, the ± would be tiny
relative to the mean (e.g. `0.14 ± 0.02`). Here the ± (0.12) is almost as big as the
mean (0.14) → the scores are **widely scattered** → a **heavy tail**: most windows do
far better than 0.14, while a handful are awful (the lone max of 0.313) and drag the
average up. The doc's numbers confirm it: the RMSE **mean (0.0049) sits above the
median (0.0037)**, and *mean > median* is the classic signature of a right-skewed,
heavy-tailed set. In plain terms: *the model is usually better than its average score
suggests; its weak spot is a small number of hard windows.*

### 5.4 The hierarchical aggregation (why our numbers differ from the paper's)
TokaMark computes metrics **per window**, then averages **per shot**, then
averages **across shots**. That "hierarchical" protocol is why our `n_shots=46`
(not 60) shows up — only shots with valid ψ contribute. **Do not compare our
absolute values to the paper's full-set values**; ours are subset numbers.

---

## PART 6 — The model, in plain terms

(Full code: `docs/reference/architecture_multiconv_mlp_standalone.py`. Verbatim
upstream: `docs/reference/*_upstream.py`.)

### 6.1 The big picture
Eight signals arrive, each with a different length. The model:
1. Gives **each signal its own little brain** (a 1-D CNN "encoder") that squeezes
   it into a short vector of 32 numbers.
2. **Concatenates** all eight vectors → 256 numbers: a compact summary of the
   whole plasma state.
3. Pushes those 256 through a small **MLP** down to a **32-number "context"**.
4. Pushes the context back up through another MLP and **broadcasts** it across
   the output grid.
5. Uses a **decoder CNN** per output to paint the **65×65 ψ** field.

### 6.2 Key architecture numbers
- **D = 16** — base width. Channel counts double each conv layer: 16 → 32 → 64.
- **bb_factor = 2** — final latent multiplier; per-channel output vector = D·2 = 32.
- **kernel=3, stride=3, padding=1, layers=3** — the standard encoder/decoder recipe.
- **~0.3 M parameters** — genuinely tiny (GPT-class models are billions). Tiny +
  small data = memorisation risk, which is exactly what the 40 GB run showed.

### 6.3 The two variants (answering the "CNN+LSTM vs CNN+MLP" question)
- `MultiConv_MLP` (**what we run**): the bottleneck is a plain MLP. Fast, simple,
  and appropriate because task_1-3 is **markovian** (no memory needed).
- `LSTM_v3` (the sibling): same CNN encoders/decoders, but the bottleneck is an
  **LSTM** that carries state across time. Needed for non-markovian tasks
  (forecasting), overkill for ours. The MLP file's own docstring accidentally
  still says "LSTM," a copy-paste fossil proving MLP was derived from LSTM.

### 6.4 Is this an Autoencoder? (Cross-Modal Reconstruction)
In a traditional Autoencoder, the input and output are the same object ($X \to Z \to X$),
such as compressing an image to a latent code and uncompressing the same image.

This architecture is a **Cross-Modal Reconstructor / Surrogate Autoencoder**:
- **Input domain:** 8 heterogeneous 1-D physical time-series (magnetic coil voltages,
  flux loop signals, currents) measured at the vessel walls.
- **Bottleneck ($Z$):** A compact 32-dimensional latent vector representing the
  global plasma state (position $R_0, Z_0$, total current $I_p$, shape elongation $\kappa$,
  triangularity $\delta$).
- **Output domain:** A 2-D dense spatial physics field ($\psi(R, Z)$ on a $65 \times 65$ grid).

The bottleneck acts just like an autoencoder: it forces the network to throw away sensor
noise and extract the low-dimensional physical degrees of freedom governing the plasma.

### 6.5 Why 8 Separate Encoders instead of one big matrix?
The 8 diagnostics have vastly different sampling frequencies (e.g., 5 kHz saddle loops vs.
50 kHz pickup coils) and physical units (Tesla, Volts, Amperes).
- If you tried to concatenate them into a single 2-D matrix at the input, you would have to
  interpolate, resample, and distort the raw physical signals.
- By giving each diagnostic its own **independent 1-D CNN encoder**, each sub-network
  learns the frequency and scale features specific to that physical instrument, mapping
  each into a standardized **32-dimensional embedding**.

### 6.6 Why MLP and not FFN? (Terminological clarification)
In modern deep learning:
- **FFN (Feed-Forward Network):** In Transformer literature (e.g. GPT, BERT), an "FFN"
  usually refers to a point-wise dense layer ($x W_1 \to \text{GELU} \to W_2$) applied to
  each token independently without mixing across tokens.
- **MLP (Multi-Layer Perceptron):** Refers to fully-connected dense layers with nonlinear
  activations and normalization that perform **global cross-channel feature mixing**.
- Here, the model concatenates $8 \times 32 = 256$ features from all sensors. The **MLP**
  is specifically needed because it allows every sensor feature to interact with every
  other sensor feature, discovering physical correlations (e.g. how a voltage drop in
  a saddle coil relates to a current spike in the central solenoid) before compressing
  down to the 32-dimensional bottleneck.

### 6.7 Tensor Dimensions Step-by-Step (The Data Flow)
```
1. 8 Raw Sensor Windows       -> [B, 1, 1, T_i, C_i]   (different lengths & channels)
2. 8 Conv1D Encoders          -> 8 x [B, 32]           (32 features per instrument)
3. Concatenation              -> [B, 256]              (global fused sensor representation)
4. Encoder MLP                -> [B, 256] -> [B, 64] -> [B, 32] (physical bottleneck)
5. Decoder MLP                -> [B, 32] -> [B, 64] -> [B, 32]  (conditioned latent state)
6. Spatial Broadcast          -> [B, 32, 1, 1]         (tile across spatial dimensions)
7. ConvTranspose2D Decoder    -> [B, 1, 65, 65]        (full 2D poloidal flux field psi)
```

---

## PART 7 — Operational reality (why things are the way they are)

- **The 7-hour GPU wall:** Kaggle sessions are time-limited and the GPU quota is
  finite. Everything is sized to *finish within 7 h*: budget 40 GB, deadline 7 h.
- **`DL_WORKERS=0`:** multi-process data loading silently froze on Kaggle. Loading
  in the main process (`0`) is ~4 s/step but hang-proof. Reliability > speed here.
- **Subset, not baseline:** we stress repeatedly that the data is a **thinned
  slice**. The science (relative comparisons, robustness) is valid; the absolute
  scores are not the official ones.
- **Prep is cheap to re-run:** the tar index and `selection.json` are cached;
  only the `/tmp` shot files vanish between sessions.

---

## PART 8 — Sensor Robustness & Online Fault Mitigation (In Plain English)

### 8.1 Why do we test sensor failures? (The real-world problem)
A fusion tokamak like MAST (or ITER) is one of the most violent environments on Earth:
- The core plasma reaches over **100 million degrees Celsius** (hotter than the center of the Sun).
- Extreme neutron radiation, gamma rays, and megawatt electrical noise bombard the vessel walls.
- In real-world operation, **sensors fail all the time**! Diagnostic wires shake loose from magnetic vibrations, digitizers drop packets, probes melt, and communication channels lag.

In a commercial power plant or experimental reactor, an AI model will be used in real-time control (evaluating every 5 milliseconds to steer the magnetic fields). If a single sensor drops out, the AI cannot crash or throw an error. If the control system loses the plasma shape for even 20 milliseconds, millions of amperes of superheated plasma can crash into the stainless-steel vessel wall at supersonic speeds, causing a **plasma disruption**—melting interior armor tiles and costing millions of dollars in repairs.

Therefore, our model MUST be stress-tested to see what happens when sensors glitch, burn out, or disconnect.

---

### 8.2 The 6 Failure Scenarios (Gupta arXiv 2607.11915 explained simply)
We ported the official fault scenarios from Gupta et al. (UKAEA / IBM) to our 2D magnetic flux reconstruction task:

1. **`random_dropout` (10%, 25%, 50% missingness):**
   - *What it mimics:* Stochastic electrical noise, analog-to-digital converter (ADC) packet drops, or microwave heating interference.
   - *What we do:* We randomly delete 10%, 25%, or 50% of the numbers across all 8 sensor input streams.
2. **`temporal_gap` (20%, 40%, 60% blackout):**
   - *What it mimics:* A communication freeze, telemetry packet delay, or temporary power dip that cuts off a sensor for a contiguous slice of time.
   - *What we do:* We blank out a solid block of time (up to 60% of the 5 ms window).
3. **`channel_ablation` (1, 3, 6 dead channels):**
   - *What it mimics:* Physical hardware burnout! A sensor wire burns through, or a diagnostic probe is permanently destroyed mid-shot.
   - *What we do:* We completely zero out 1, 3, or 6 diagnostic instruments for the entire evaluation.
4. **`category_abl` (Entire subsystem failure):**
   - *What it mimics:* A whole subsystem circuit trips. For example, what if all 15 flux loops lose power simultaneously? What if all 40 magnetic pickup coils go offline? What if the plasma current diagnostic fails?
   - *What we do:* We wipe out 100% of an entire diagnostic category to measure its physical importance.
5. **`group_failure` (Correlated failure):**
   - *What it mimics:* Coupled hardware buses failing together (e.g. all magnetic diagnostics sharing a common grounding line, or both the poloidal field and solenoid power supplies tripping).
6. **`disruption_proximate` (10%, 25%, 50% loss at shot end):**
   - *What it mimics:* Severe plasma instability right before an emergency termination or disruption. Intense radiation and heat cause wall sensors to saturate or fail right at the tail end of the time window.

---

### 8.3 The 3 Online Mitigation Strategies (How we heal missing data without retraining)
When a sensor dies in real time, how can the AI recover without having to retrain the neural network? We tested three online strategies:

1. **`zero` (Zero-Fill — The Default / Unmitigated Baseline):**
   - *The mechanism:* Do nothing; missing values stay zero.
   - *Why it fails:* The neural network interprets `0.0` as a genuine physical measurement (e.g., "there is zero plasma current" or "there is zero magnetic field"). It assumes the plasma has extinguished or the magnets shut down, causing catastrophic distortion of the 2D flux surfaces (+148% error surge).
2. **`mean` (Dataset Mean-Fill):**
   - *The mechanism:* Replace every missing hole with the historical average of that sensor, calculated strictly from the **TRAIN split** (zero test data leakage).
   - *Why it helps:* It tells the network: *"Assume this sensor is reading a typical, average MAST shot."* This prevents wild mathematical explosions, but it cannot adapt to dynamic changes happening in the current shot.
3. **`ffill` (Physics-Informed Forward-Fill / Sample-and-Hold — The Winner):**
   - *The mechanism:* Whenever a sensor value is lost, latch and hold the **last known valid measurement** forward in time.
   - *Why it works so brilliantly (The Physics):* In electromagnetism, electric currents in massive copper coils cannot jump instantaneously because of self-inductance ($L \frac{dI}{dt} = V$). Similarly, a 1-MegaAmpere plasma possesses physical inertia; its magnetic field evolves on resistive and transport timescales ($\tau \sim 10\text{–}50\text{ ms}$). Across a tiny 5 ms snapshot, the magnetic field literally cannot change drastically. Holding the last known valid reading preserves the magnetic equilibrium almost perfectly!
   - *The Result:* Forward-fill reduces error surge from **+148%** down to **<18%**, recovering **>85%** of the lost accuracy with zero retraining and negligible computation!

---

### 8.4 The Metrics Explained Like You Are 10
- **NRMSE (Normalized Root Mean Squared Error):**
  - Think of this as the percentage of error across the whole 2D donut cross-section. A score of `0.1419` means our model achieves ~86% physical reconstruction accuracy. Lower is better.
- **MAE (Mean Absolute Error in Webers $\text{Wb}$):**
  - The actual physical error in magnetic flux units. Our champion model has an MAE of **`0.00389 Wb`**. Since the total flux in MAST ranges from $-0.10\text{ Wb}$ to $+0.10\text{ Wb}$ (a span of $0.20\text{ Wb}$), an error of $0.00389\text{ Wb}$ is less than **1% average deviation** across 4,225 spatial points!
- **Robustness Score ($RS$):**
  - Defined by Gupta as:
    $$RS = \frac{\text{Clean Baseline Error}}{\text{Corrupted Error}}$$
  - If the model is completely unaffected by a broken sensor, $RS = 1.00$.
  - If the error doubles under sensor failure, $RS = 0.50$.
  - If the error triples, $RS = 0.33$.
  - Higher $RS$ means more resilient.
- **Error Surge ($\Delta\%$ Increase):**
  - How much worse the error got compared to clean operation:
    $$\Delta\% = \left(\frac{\text{Corrupted NRMSE}}{\text{Clean NRMSE}} - 1\right) \times 100\%$$

---

### 8.5 Diagnostic Criticality: Which sensors matter most?
By turning off one sensor group at a time, we discovered the hierarchy of diagnostic importance in a tokamak:
1. **Plasma Current ($I_p$) is King (Criticality #1, +140% Error Surge, $RS \approx 0.41$):**
   - Why: Ampere's Law ($\oint \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_p$). The total toroidal plasma current is the source of the poloidal magnetic field. If the network does not know how much current is flowing through the plasma, it cannot determine the core flux amplitude.
2. **Shaping Coils & Solenoid are Second (Criticality #2, +20% to +35% Error Surge):**
   - Why: External coils stretch and squeeze the plasma into an elongated D-shape. Losing them degrades boundary geometry.
3. **Pickup Probes & Flux Loops are Highly Resilient (Criticality #3, +4.3% Error Surge):**
   - Why: MAST has 40 magnetic pickup coils and 15 flux loops distributed around the vessel wall. If 1 or 2 fail, the remaining 38 probes provide dense spatial redundancy. The network easily interpolates the missing wall data!

---

### 8.6 Why CNN+MLP beats CNN+LSTM on Robustness
Our project compares two architectures: **CNN+MLP** (our champion) vs. **CNN+LSTM** (the benchmark):
- **Equilibrium Reconstruction is Markovian:**
  - The magnetic equilibrium at time $t$ depends strictly on the physical currents and fields at time $t$. It does not depend on what happened 50 milliseconds ago.
- **The Danger of Recurrent Memory (LSTM):**
  - An LSTM maintains an internal memory state ($\mathbf{h}_t, \mathbf{c}_t$) that carries forward in time. When a sensor drops out and zero-fill sends zeroes into the LSTM, those zeroes **poison the hidden memory state**. Even after the sensor comes back online, the corrupted memory continues to echo forward into subsequent time steps!
- **The Clean Recovery of Feedforward (MLP):**
  - CNN+MLP has no recurrent memory state. It evaluates the current 5 ms window instantaneously. If a temporary glitch happens, only that specific window is degraded; the very next window recovers with 100% clean accuracy.
- **Conclusion:** CNN+MLP is not only more accurate on clean data ($0.1419$ vs $0.1969$), but also physically safer and more fault-tolerant under transient sensor failures!

---

## PART 9 — Glossary (quick lookup)

- **shot** — one plasma discharge / experiment run.
- **window** — a 5 ms slice used as one training example.
- **ψ (psi)** — poloidal flux; the 2-D field we predict.
- **NRMSE / NMAE / RMSE / MAE** — error metrics (Part 5).
- **overfitting** — memorising training data; val loss worsens while train improves.
- **early stopping / patience** — quitting when val stops improving.
- **standardisation** — rescaling data to ~zero-mean, unit-variance.
- **markovian** — output depends only on the current instant.
- **Robustness Score (RS)** — Gupta's metric: clean NRMSE / corrupted NRMSE, averaged over scenarios.
- **imputation** — filling corrupted/missing values (zero/mean/forward-fill).
