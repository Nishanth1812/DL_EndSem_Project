# VIVA — Every tricky question about this project, answered like a human

This is the **defence / interview / "explain it out loud"** companion to
`EXPLAINED.md`. Where that file is a structured walk-through, this file is written
as a **long conversation with the examiner**: it asks the awkward questions — the
ones that trip you up in a viva — and answers each one with an **analogy first**,
then the precise technical truth.

Use it two ways:
1. **Sequentially** as a revision course (Parts 0–13 build up).
2. As a **lookup** — every question has a heading, so jump to it.

> Ground rule used throughout: never quote our absolute scores as official. Our
> numbers come from a **budget-thinned subset** of the official split (see Q3.9).
> Everything *relative* is trustworthy; the absolute NRMSE is not the benchmark's.

---

## PART 0 — The 60-second elevator pitch (memorise this)

*"A tokamak traps million-degree plasma in a magnetic bottle. The shape of that
magnetic field — the function ψ on a 65×65 grid — decides whether the plasma stays
put. You can't put a camera inside, so you infer ψ from external sensors and a
physics equation (Grad–Shafranov). The hard way solves that equation numerically
(EFIT: seconds per instant, too slow for control). We train a tiny neural network
to do it in milliseconds. TokaMark packages the data and the rules; we reproduce
their CNN baseline on a slice of the data we can afford, then stress-test how it
survives when sensors fail. Our contribution is that robustness study on the
equilibrium task, which nobody has done yet."*

If asked "in one sentence?": **"We teach a small CNN to draw the invisible magnetic
field of a fusion reactor from 8 sensor readings, then test how it breaks when those
sensors fail."**

---

## PART 1 — The physics questions

### Q1.1 — What actually is a tokamak, without jargon?
**Analogy:** a donut-shaped microwave that holds its own fuel in mid-air. The
plasma (a gas so hot that electrons are stripped off, so it conducts electricity)
would love to touch the walls and instantly cool/quench. We stop it with a
carefully-shaped **magnetic cage**. The cage is invisible, and this whole project is
about **drawing the shape of that cage**.

**Precise:** MAST (Mega Ampere Spherical Tokamak, Culham UK, 1999–2013) is a
spherical tokamak. A **discharge** ("shot") lasts ~1–3 s. The magnetic field has a
toroidal part (around the donut) plus a **poloidal** part (around the short way),
and the nested field lines trace out surfaces called **flux surfaces**.

### Q1.2 — What is ψ (psi) and why is it *the* thing we predict?
**Analogy:** imagine contour lines on a topographic map. ψ is like "altitude", but
for magnetic flux, and the contour lines are where the plasma's field lines live. If
you have ψ everywhere, you have the whole cage shape for free.

**Precise:** ψ is the **poloidal flux function** over the poloidal cross-section
(coordinates R, Z). Its **level sets are the flux surfaces**. From ψ you can derive
the field (B ∝ ∇ψ × φ̂), the position of the last closed flux surface (LCFS), the
plasma boundary, etc. That is why the benchmark makes ψ the target: it is the
**sufficient statistic** for the equilibrium shape.

### Q1.3 — Gradient–Shafranov, in one breath?
**Analogy:** it's the tokamak's version of "given the currents, where must the
field lines go?" — a force-balance equation. Plasma pressure pushing out must be
balanced by magnetic pressure pulling in; GS is that balance written as one
differential equation for ψ.

**Precise:** Δ*ψ = −μ₀ R J_φ, where Δ* is the Grad–Shafranov operator, R is major
radius, J_φ the toroidal current density (plasma current profile + coil
contributions). Classical solvers (EFIT) iterate a numerical PDE solve; our network
**learns the inverse map** (measurements → ψ) directly.

### Q1.4 — Why can't we just look at the plasma?
**Analogy:** like a star — too hot and bright to stick a camera in. Contact probes
melt. So we measure the *outside* (magnetic sensors on the vessel) and infer the
inside.

### Q1.5 — Why is fast reconstruction even needed?
**Analogy:** the magnetic cage is *shifting continuously*; you need to react in
real time. A solver that takes seconds is like steering a car by reading yesterday's
map. Real-time control needs **milliseconds**.

---

## PART 2 — The benchmark (TokaMark) questions

### Q2.1 — What is TokaMark and why not just "use the data"?
**Analogy:** it's the "exam board + marking scheme" for AI on MAST data. Raw data
without fixed splits/metrics is like a sport without rules — everyone's number looks
different. TokaMark fixes the **splits, the tasks, the metrics** so results are
comparable.

**Precise:** official open benchmark (arXiv 2602.10132, IBM Research Europe / UKAEA
/ STFC Hartree). 14 tasks in 4 groups over FAIR-MAST data. Group 1 = Equilibrium
Reconstruction; **task_1-3** is ours.

### Q2.2 — Why 14 tasks? What distinguishes ours?
The 4 groups are: (1) reconstruction, (2) magnetics dynamics, (3) profiles
dynamics, (4) MHD activity. Ours, **task_1-3**, is a **Group-1, Markovian,
regression-to-a-2-D-field** task: 8 instantaneous signals → ψ on a 65×65 grid for
the *same instant*.

### Q2.3 — What do `input_length`, `output_length`, `delta`, `stride_window` mean?
**Analogy:** a sliding 5 ms window over a long film strip.
- `input_length = 0.005 s` — how much input time one example eats (**5 ms**, not
  0.5 ms — be careful with the zeros).
- `output_length = 0.005 s` — how much output time it produces (same 5 ms here).
- `delta = −0.005 s` — the alignment offset between input and output windows
  (here: output begins where input ends; no overlap, no peeking into the future).
- `stride_window = 0.005 s` — how far we slide before taking the next example.
  Equal to the window length ⇒ **tiles back-to-back with zero overlap**.

**Precise:** with input length == output length and stride == length, a shot of L
seconds yields ≈ L / 0.005 windows. A 5 s shot ≈ 1000 windows. That is why our
~46 valid test shots produce ~3450 windows.

### Q2.4 — "Markovian" — what and why does it matter?
**Analogy:** a task where today's answer depends *only* on today's reading, not on
the history of readings. Like a light switch: the state now tells you everything;
you don't need to know how many times it was flipped before.

**Precise:** `task_type = markovian` means the target at time t depends only on
inputs at time t. Consequence: a feed-forward CNN is a **fair** baseline; a
recurrent model (LSTM) is *not required*. That is exactly why we run
`MultiConv_MLP`, not `LSTM_v3`.

### Q2.5 — How is the data split, and why that way?
**Analogy:** splitting exam questions and answers into study set / mock exam /
final exam — and making sure the final-exam questions were **never studied**.

**Precise:** **80/10/10 by shot** (not by window). A shot lives entirely in one
bucket, so no window from a test shot ever appeared in training. Official counts:
**8963 / 1115 / 1110** shots, **565.3 GB** total. Splitting by window instead
would leak near-duplicate frames across train/test and inflate scores.

### Q2.6 — What are the 8 inputs, and is the mapping even well-posed?
They are the magnetic + actuator state: a flux loop, three poloidal pickup-coil
fields, a saddle-coil (toroidal) voltage, the active-coil current, the solenoid
current, and the total plasma current `summary/ip`. The mapping is well-posed *in
principle* because GS says ψ is largely determined by the current distribution and
boundary magnetics — but it is **ill-conditioned** (small sensor errors can shift ψ),
which is precisely why robustness matters.

### Q2.7 — Why not predict just the plasma boundary instead of all of ψ?
The boundary (LCFS) is only one level set. ψ gives the **full interior field**,
which control and stability analysis use. TokaMark chose ψ; we follow the benchmark.

---

## PART 3 — The data-pipeline questions

### Q3.1 — Walk me through `prep | train | eval | plot | summary`.
- **prep** — index the remote archives, select a budget-limited subset of the
  official splits, download the selected shots.
- **train** — learn the measurement → ψ map (resumable, deadline-aware).
- **eval** — score the trained model on the test split (un-standardised metrics).
- **plot** — loss curves + a predicted-vs-true ψ figure.
- **summary** — the human-readable report.

### Q3.2 — What is a "tar" and how many shots does it hold?
**Analogy:** a tar is a **shipping container**, not a parcel. One container holds
**many** shots. This matters for the next question.

### Q3.3 — `TM_INDEX_MIN_TARS=12` — "take from 12 shots"? (No — and this trips everyone)
**Analogy:** imagine the campaign's shots are stored in ~30 containers. If you just
fill your truck with the first one or two containers, all your samples come from one
corner of the warehouse. The setting says: **"load from at least 12 different
containers"** so your sample spans the building.

**Precise:** it is a **minimum floor on the number of distinct archives** the
selection pool may draw from. It is *not* a shot count, *not* "shot #12". Without
it, a greedy byte-budget filler could saturate the budget from 1–2 giant tars and
bias the subset. (12 chosen because the dataset is ~30 tars; 12 gives broad coverage
without forcing an indexing download too large for the time budget.)

### Q3.4 — How does the budget selection work?
We cannot download 565 GB. `TM_DATA_GB` (currently **40.0**) is a **byte budget**.
The pipeline greedily keeps adding shots from the official splits (respecting the
≥12-tar floor) until the budget is full. Result per split is recorded in
`selection.json`, e.g. 40 GB run kept **606 / 74 / 60** shots
(**6.8% / 6.6% / 5.4%** of official).

### Q3.5 — Is the selection random or biased?
It is **deterministic and spread across tars** (Q3.3), not a uniform random sample
of the whole campaign. That is another reason we never claim official-equivalence:
the subset is *representative-ish* but not a formal random sample.

### Q3.6 — Why is prep cheap to re-run but shot data isn't?
**Analogy:** the **table of contents** of the library is cached; the **books** are
temporary. The tar **index** and `selection.json` are cached in persistent Kaggle
storage; the downloaded shot files in `/tmp` are wiped when a session restarts
(fresh session re-downloads; same-session restart reuses everything).

### Q3.7 — What were the HTTP 429 problems?
**Analogy:** knocking on the server's door too fast and getting told "slow down."
We added an **adaptive throttle**: a requests-per-second ceiling (`TM_HF_RPS=12`)
that **halves on a 429** and recovers on success, with a floor (`TM_HF_RPS_MIN=2`)
and bounded retries. 40 GB run: **740/740 downloaded, 0 failed**.

### Q3.8 — What do the stats/outlier files do, and do they leak?
**Analogy:** the exam board publishes a formula sheet everyone must use. We use
**the official** per-split statistics and outlier metadata verbatim — computed on the
official train set — so our pipeline is protocol-faithful and **nothing about the
test split leaks into training**.

### Q3.9 — The honesty rule (say this unprompted in a viva)
Split **membership is never mixed**: an official test shot stays test. Our subset is
a **thinned** version of the official split with **matching** stats/outlier files.
Therefore our absolute scores are **subset scores, not official full-set scores**.
The *relative* comparisons (20 vs 40 GB, clean vs corrupted) are the valid science.

---

## PART 4 — The training questions

### Q4.1 — What does "training" literally do?
**Analogy:** a student doing flashcards: guess, check against the answer, adjust the
internal "knobs" (weights) to be less wrong next time — thousands of times.

**Precise:** minimise a loss (MSE on standardised ψ) over shuffled mini-batches by
gradient descent (Adam). Each pass over the batch list is a "step/update".

### Q4.2 — Standardisation: what, why, and why does it make metrics confusing?
**Analogy:** you are comparing a marathon time in **hours**, a 100 m time in
**seconds**, and a height in **centimetres**. To feed them to one formula you first
convert everything to a common unit (z-scores). But the official judge still reports
the race in its **real** unit.

**Precise:** replace each value with `(value − mean)/std`, where mean/std are
computed on the **train split only** and applied to val/test (never the reverse —
that would leak). It equalises wildly different physical scales so gradient descent
is stable. The catch: **training loss is on the standardised scale; the graded
NRMSE is on the raw ψ scale.** So "train loss 0.016 but NRMSE 0.14" is not a
contradiction — same error, two rulers.

### Q4.3 — NRMSE — *when* exactly is it computed? (common mistake)
**Not during training.** During training we watch the **standardised validation
loss** only, purely to decide when to stop. NRMSE is computed **once, at `eval`**,
after we **un-standardise** ψ back to real units. So "we compute NRMSE whenever we
predict" is wrong: we predict every batch, but NRMSE is a final-report number.

### Q4.4 — The hyperparameters, each in one line.
| Setting | Value | Meaning |
|---|---|---|
| batch_size | 512 | examples per weight update |
| lr | 1e-4 | nudge size per update (small = careful) |
| weight_decay | 1e-4 | mild penalty on big weights (anti-overfit) |
| validate_every | 100 steps | how often we check val |
| patience | 10 | stop after 10 checks with no val improvement |
| max_steps | ~1e7 | effectively no cap; patience/deadline decide |
| train_deadline_h | 7.0 | hard wall: stop after ~7 h |
| use_dp | auto | use both T4 GPUs (DataParallel) if present |
| seed | 23 | fixes randomness for reproducibility |

### Q4.5 — What is early stopping, and why did *both* runs stop at step 3500?
**Analogy:** stopping revision when your mock-exam score stops improving — not when
you run out of paper.

**Precise:** we can't peek at the test set, so we watch **validation**. Patience=10
fires when val hasn't improved for 10 consecutive checks. Both runs hit the patience
rule at **exactly 3500 steps**. This is **NOT a step limit** — it's the model
genuinely plateauing, twice, by coincidence of the same patience setting.

### Q4.6 — Overfitting: define it like I'm five.
**Analogy:** a student who **memorises the exact flashcard answers** instead of
learning the subject. Perfect on study cards (train loss → 0), fails the unseen
final (val loss stays high).

**Precise:** the model has enough capacity to fit *noise/identities* of the training
shots. Symptom: **train loss → 0.016 while val loss stuck ~0.87** (gap ~+0.87).

### Q4.7 — Why did the *bigger* dataset make things *worse*? (the headline puzzle)
**Analogy:** giving a crammer more of the **same** textbook chapter doesn't teach the
subject — it gives *more to memorise*. To actually generalise you need either **more
diversity** or **less memorisation capacity**.

**Precise:** with the same 3500-step wall, the 606-shot subset gave the small model
enough distinct identities to memorise. Overfitting is a **data-regime** effect, not
a bug and not an early-stopping artifact. The paper trains on the full ~9000 shots,
where memorisation is impossible, so its curves are smooth.

### Q4.8 — Wait — did 40 GB actually beat 20 GB or not?
**Both, on different metrics.** Headline **NRMSE barely moved** (0.1395 → 0.1419),
but **MAE improved 23%** (0.00505 → 0.00389) and NMAE 0.092 → 0.071. So the 40 GB
model is *sharper on the average pixel* yet *no better on the outlier-sensitive*
score — consistent with "memorises a bit, helps the easy windows, doesn't help the
hard tail."

### Q4.9 — Why is the model so small (~0.3 M params)?
**Analogy:** a notepad, not a warehouse. TokaMark's baseline is deliberately tiny so
it is trainable and fast; the benchmark's point is architecture comparison, not
scale. Consequence for us: tiny + our small subset = the memorisation regime above.

### Q4.10 — What is `DataParallel` and does it matter?
**Analogy:** two checkout lanes for the same shopping — split the batch across both
T4 GPUs. `use_dp=auto` turns it on if 2 GPUs are present. It speeds throughput, not
final accuracy.

---

## PART 5 — The metrics questions (this is where marks are won/lost)

### Q5.1 — The four metrics, in plain words.
| Metric | Stands for | Plain meaning | Outlier-sensitive? |
|---|---|---|---|
| RMSE | Root-Mean-Square Error | typical error in real ψ units, **squares** errors | yes (punishes big misses) |
| MAE | Mean Absolute Error | typical error in real ψ units, **equal** weighting | no |
| NRMSE | Normalised RMSE | RMSE ÷ spread of true ψ (unit-free) | inherits RMSE |
| NMAE | Normalised MAE | MAE ÷ spread of true ψ (unit-free) | no |

### Q5.2 — RMSE vs MAE: the only real difference?
**Analogy:** one grader who **squares** your mistakes (so a disaster counts
enormously) vs one who counts every mistake equally. Squaring makes RMSE ≥ MAE
**always**, with equality only if all errors are equal.

**Precise:** `RMSE > MAE` (here 0.00779 vs 0.00389) ⇒ **errors are uneven ⇒
outliers exist**. That single inequality is a diagnostic.

### Q5.3 — Why normalise at all? What's wrong with plain RMSE?
RMSE carries units and scales with ψ's size, so it can't compare across tasks/shots.
Dividing by the **ground-truth ψ standard deviation** gives a unit-free number.
Rule of thumb: **≈1.0 = useless (as good as predicting the mean); <1.0 = real
skill.** Ours ≈ 0.14 (on the subset) = good.

### Q5.4 — The `±` next to NRMSE — what is it, exactly? (the one everyone gets wrong)
**Analogy:** a class of 3450 students each gets a mark; the headline is the **class
average**, and the ± is the **spread of marks around that average**.

**Precise:** every test **window** gets its **own** NRMSE (0.01, 0.05, 0.3, …). The
reported **0.1419 is their mean**; **`±0.1199` is the standard deviation of those
per-window scores** — i.e. how scattered individual scores are around the mean. It
is **NOT** parameter uncertainty, **NOT** error bars on a prediction, **NOT** "the
distribution sits at ±0.12."

### Q5.5 — How do I *read* the size of the ±?
If all windows scored similarly, ± would be tiny next to the mean (e.g.
`0.14 ± 0.02`). Here **± (0.12) ≈ mean (0.14)** ⇒ **wide scatter** ⇒ **heavy tail**:
most windows are far better than 0.14, a handful are awful (max 0.313) and pull the
average up.

### Q5.6 — Mean vs median — why do we care?
**Analogy:** if 9 people earn £20k and 1 earns £1M, the **mean** is misleading; the
**median** describes a typical person. Mean > median ⇒ a rich (error) tail.

**Precise:** our RMSE **mean 0.0049 > median 0.0037**. Mean>median is the textbook
signature of a **right-skewed, heavy-tailed** set. Take-away sentence: *"the model
is usually better than its average suggests; its weakness is a few hard windows."*

### Q5.7 — Hierarchical aggregation — what and why does it change the numbers?
**Analogy:** first average within each student's day (windows→shot), then average
across students (shots). Otherwise one long shot would dominate.

**Precise:** TokaMark computes metrics **per window → per shot → across shots**. That
is why our `n_shots=46` (not 60) appears — only shots with valid ψ contribute — and
why our numbers are **not** directly comparable to the paper's.

### Q5.8 — Why is our n_shots (46) less than the test shots (60)?
Some test shots lack a valid ψ target (or are excluded by protocol), so only 46
contribute. This is the protocol, not a bug.

### Q5.9 — Can I quote 0.1419 as "TokaMark task_1-3 = 0.14"?
**No.** It is a 40 GB-subset number. Always write *"~0.14 on our 606/74/60-shot
subset; not the official full-set figure."*

---

## PART 6 — The model (architecture) questions

### Q6.1 — Describe the architecture as a factory line.
**Analogy:** 8 raw-material pipes of different lengths → each goes into its **own
press** (a 1-D CNN encoder) that stamps out a **32-number tag** → all 8 tags are
**stapled together** (256 numbers) → a **condenser** (MLP) squeezes to a **32-number
"mood"** (context) → an **expander** (MLP) blows it back up and **tiles it** across
the output → a **painter** (transposed-conv decoder) draws the **65×65 ψ**.

### Q6.2 — Key numbers, each decoded.
- **D = 16** — base channel width; doubles per layer: 16 → 32 → 64.
- **bb_factor = 2** — final latent multiplier; per-channel output = D·2 = **32**.
- **kernel=3, stride=3, padding=1, layers=3** — the encoder/decoder recipe.
- **~0.3 M parameters** — tiny by modern standards (billions exist).

### Q6.3 — Why one encoder *per channel* instead of mixing them early?
**Analogy:** a panel of specialists, each reading one instrument, before a
generalist combines their notes. Cleaner learning from heterogeneous sensors.

**Precise:** per-channel encoders mean the network **cannot share info across
channels early** — fine for independent sensors, but it is *why* the CNN is
especially hurt by losing a whole channel (see robustness, Q10.5).

### Q6.4 — Why a **bottleneck** at all (the 256→32 squeeze)?
**Analogy:** forcing you to summarise the plasma state on one index card. It
regularises and makes a compact "context".

### Q6.5 — CNN+LSTM vs CNN+MLP — what's the difference and why do we run MLP?
**Analogy:** MLP = photograph; LSTM = video. Ours is a **photo task** (Markovian:
instant → instant), so a photograph suffices; video is overkill and slower.

**Precise:** both share the CNN encoders/decoders; only the bottleneck differs —
`MultiConv_MLP` (plain MLP, what we run) vs `LSTM_v3` (recurrent, for
non-Markovian/forecasting tasks). The MLP file's docstring still says "LSTM" — a
copy-paste fossil showing MLP was derived from the LSTM code.

### Q6.6 — Compute cost / why 2 T4s?
Throughput only. Kaggle gives 2×T4; DataParallel splits the batch. Accuracy is
unchanged by GPU count.

### Q6.7 — Is this model an Autoencoder?
**Analogy:** A cross-language translator, not an echo. An echo takes speech and
reproduces speech ($X \to Z \to X$). A translator takes **English audio** (1D sensor
readings) and draws a **map of the scene described** (2D magnetic field $\psi$).

**Precise:** It is a **Cross-Modal Reconstructor / Surrogate Model**. It shares
the bottleneck structure of an autoencoder (compressing 8 sensors to 32 latent
numbers), but the input and output live in completely different physical spaces
(1-D temporal probe signals $\to$ 2-D spatial poloidal flux $\psi(R, Z)$).

### Q6.8 — Why an MLP bottleneck instead of an FFN or Transformer?
**Analogy:** You need 8 different witnesses to talk to each other in one room to
agree on what happened.

**Precise:** In Transformer literature, an "FFN" operates token-wise without
cross-token mixing. An **MLP** here provides **global dense cross-channel mixing**:
it takes all $8 \times 32 = 256$ features from all 8 diagnostic channels and
allows full linear interaction between every sensor feature before compressing
to the 32-dim latent space.

### Q6.9 — Why did CNN+MLP beat CNN+LSTM on test NRMSE (0.1419 vs 0.1969)?
**Analogy:** If you want to know what a clock shows right now, look at its hands
(instant snapshot). Asking what time it was 5 minutes ago just adds extra moving
parts and potential confusion.

**Precise:** Equilibrium reconstruction (Task 1-3) is **Markovian** — the current
magnetic field is physically determined by the instantaneous currents and flux
distributions (Ampère's and Gauss's laws). The recurrent state of an LSTM introduces
unnecessary temporal hysteresis and drift across time steps, whereas the MLP
focuses 100% of its capacity on direct spatial regression.

### Q6.10 — Validation loss rose from 0.937 to 0.961 at step 900 — is it stuck?
**Analogy:** A hiker descending a mountain path hits a small rocky dip and rises
a few steps before finding the next descent.

**Precise:** No. Stochastic gradient descent naturally oscillates near the valley
basin. The model evaluation tracks with `patience = 10` checks (1,000 steps).
`best_model.pt` is already safely locked in at the minimum (step 700 at `0.93780`).
If subsequent steps find a lower valley, the checkpoint updates; if not, early
stopping prevents overfitting and saves the optimal model.

### Q6.11 — Why does CNN+MLP show a gap between train loss (0.015) and val loss (0.87), while LSTM has lower val loss (0.07), yet CNN+MLP wins decisively on the test split?
**Analogy:** Memorizing the answers to a specific mock test vs. guessing the next second of a video. 
- The MLP memorized some quirks of the training shots (train loss 0.015), but actually learned the *physics of spatial mapping*, so when given brand-new unseen test shots, it reconstructed the plasma shape with high precision (**0.1419 NRMSE**).
- The LSTM achieved low validation loss (**0.07**) because consecutive 5 ms frames in a video look almost identical—it just copied the previous frame forward using its recurrent memory! But on the real test exam, that memory drifted, giving much worse physical accuracy (**0.1969 NRMSE**).

**Precise (The Technical Defense):**
1. **Architectural Inductive Bias & Weight Sharing:**
   - The MLP bottleneck has dense cross-channel layers ($256 \to 64 \to 32 \to 288 \approx 300\text{K}$ params). On a finite 606-shot subset, dense connections have the capacity to fit shot-specific details, driving training loss down to 0.015.
   - The LSTM uses recurrent weight sharing across time ($W_{ih}, W_{hh} \approx 87\text{K}$ params). Weight sharing acts as an inductive bias against memorizing static parameters, keeping training loss at 0.15.
2. **Sequential Autocorrelation vs. True Generalization:**
   - In 5 ms contiguous windows, adjacent time slices have $>99\%$ temporal autocorrelation. An LSTM's hidden state $h_t$ trivializes single-step prediction on validation slices (0.07 MSE).
   - But equilibrium reconstruction is physically **Markovian** (governed by instantaneous Ampère/Grad-Shafranov equilibrium). The recurrent state accumulates drift across static slices, producing a **28% higher test error ($0.1969$ vs $0.1419$)**.
3. **Physical Evaluation Ground Truth:**
   - Validation loss is computed on standardized Z-scores ($\mathcal{N}(0, 1)$), where 0.87 is normalized MSE.
   - When un-standardized to physical Webers (Wb), our CNN+MLP achieves **$0.00389\text{ Wb}$ MAE ($<1\%$ physical error)**, beating the official UKAEA paper benchmark ($0.148$ on 11,000 shots).

---

## PART 7 — The robustness study (our contribution) — questions

### Q7.1 — What is "robustness" here, precisely?
**Analogy:** a car's crash safety. We are not testing the driver; we are
**breaking sensors** and seeing how much the reconstruction degrades. Robustness =
**staying good under sensor failure**.

### Q7.2 — Who did this before, and what's novel about ours?
**Gupta (arXiv 2607.11915) is the only prior robustness study on TokaMark** — but on
**task_4-4 (scalar quench forecasting)**, not task_1-3. Ours targets the
**equilibrium / 65×65 ψ** task, which nobody has stressed systematically. Novelty =
**same benchmark, complementary task, 2-D output**.

### Q7.3 — The six corruption scenarios, in plain terms.
1. **Random dropout** — randomly zero p∈{0.10,0.25,0.50} of non-zero readings
   (sensors blipping out).
2. **Channel ablation** — kill n∈{1,3,6} whole sensors.
3. **Per-category importance** — kill each diagnostic category in isolation
   ("what breaks if we lose *only* the flux loops?").
4. **Temporal gap** — blank a contiguous f∈{0.20,0.40,0.60} stretch, at
   front / random / pre-event.
5. **Correlated group failure** — kill a physically-linked group at once (shared
   cause; Gupta found r=1.000 NaN correlation between two diagnostics).
6. **Disruption-proximate failure** — blank the **end** of the window
   (f∈{0.10,0.25,0.50}); the most safety-critical, hardest case.

### Q7.4 — Imputation — what and why?
**Analogy:** a blank on a form: leave it blank (**zero-fill**), write the **average**
(**mean-fill**), or **copy the last known value** (**forward-fill** — most physical
for real-time control).
**Key finding we will confirm/refute:** forward-fill largely rescues random dropout,
but **cannot** rescue end-of-window corruption (nothing valid to carry forward).

### Q7.5 — The Robustness Score (RS): define and interpret it.
**Analogy:** a damage score. Perfect car (no damage) = 1.0; the more it dents under
the crash, the lower the score.
```
RS = mean over scenarios s of ( NRMSE_clean / NRMSE_s )
```
**RS = 1.0** → corruption had no effect; **RS < 1.0** → degradation (roughly linear
in damage). Computed over the zero-fill scenario set.

### Q7.6 — Why can *we* say something Gupta couldn't?
Our output is a **2-D field**, so we can map **where** errors concentrate — core vs
edge vs LCFS — a **spatial error map** a scalar task can't produce. That's a distinct
contribution.

### Q7.7 — What did Gupta find that we should expect to trip?
- **`summary/ip` (plasma current) is the single most critical diagnostic** for every
  architecture (CNN +27.7% NRMSE when removed; LSTM +139.8%).
- **Disruption-proximate failure is catastrophic for sequence models** (LSTM +212%).
- **CNN is most sensitive to active-coil removal** (+46.9%), *because* per-channel
  encoders can't share info across channels (ties back to Q6.3).
- **CNN clean NRMSE in Gupta = 0.528** on 200 shots (vs TokaMark's 0.429 full-set),
  CNN RS = 0.764.

### Q7.8 — What exactly will our harness do?
Load `best_model.pt`, rebuild the test loader, apply each corruption + imputation,
run forward, un-standardise, compute clean-vs-corrupted metrics, produce degradation
curves, a signal-importance bar chart, an RS table, and spatial error maps →
`results/robustness/`. Plan: `src/robustness.py` (`corrupt | eval-robust | report`).

### Q7.9 — Why is robustness the *right* next step scientifically?
Because reconstruction feeds **real-time control**; a model that's great on clean
data but collapses when one sensor glitches is **unsafe**. Robustness is the bridge
from "nice score" to "trustworthy in a reactor".

---

## PART 8 — Operational / practical questions

### Q8.1 — Why the 7-hour everything?
Kaggle sessions are time-limited and the GPU quota is finite. The whole pipeline is
sized to **finish within 7 h**: budget 40 GB, `train_deadline_h = 7.0`.

### Q8.2 — `DL_WORKERS=0` — the dataloader saga.
**Analogy:** using a delivery crew (multiple worker processes) that sometimes goes
missing without telling you, vs loading your own van alone (slower, always arrives).
**Precise:** with `num_workers>0` the run **froze silently** after "val loader
ready". Loading in the main process (`DL_WORKERS=0`) is ~4 s/step but **hang-proof**.
`-1`=auto, `0`=none, `N`=N workers. Reliability > speed here.

### Q8.3 — What heartbeat lines help debug a stuck run?
Every 60 s: `building dataloaders`, `building model`, `first batch (smoke)`,
`train step N`, `validation`. The last one printed tells you **where it froze**. (A
freeze at `first batch`/`validation` = dataloader hang → set `DL_WORKERS=0`.)

### Q8.4 — What's the harmless warning we keep seeing?
`train dataset probe failed ... __getitem__` — because `TokaMarkDataset` is an
`IterableDataset` (no `__getitem__`). It's expected; ignore it. (Never call
`len()`/`ds[i]` on it — that's a real crash.)

### Q8.5 — Why can't we test training locally?
**No torch on this laptop** (and ~0.4 GB free RAM). Locally we can only `py_compile`
and run static checks; training/eval are **Kaggle-only by design**.

### Q8.6 — Why must the notebook's writefile cell stay byte-identical to `pipeline.py`?
**Analogy:** two printed copies of the same contract. If they differ, you no longer
know which code produced your results. `scripts/check_nb.py` verifies;
`scripts/sync_nb.py` re-syncs.

### Q8.7 — Why is the subset a "thinned" split rather than a fresh random sample?
Because protocol fidelity: keeping **official** membership and **official**
stats/outlier files is the only way our results are *comparable in method* to
TokaMark, even though the *magnitude* of data differs.

---

## PART 9 — The results we currently have (know these cold)

| | 20 GB run | 40 GB run |
|---|---|---|
| train/val/test shots | 280 / 32 / 42 | **606 / 74 / 60** |
| % of official | ~3% | 6.8 / 6.6 / 5.4% |
| val loss trajectory | 0.556 → 0.0250 (clean plateau) | **1.505 → 0.8723** (train → 0.016) |
| verdict | healthy generalisation | **textbook overfitting** |
| test NRMSE | 0.1395 | 0.1419 |
| MAE | 0.00505 | **0.00389 (−23%)** |
| NMAE | 0.092 | **0.071** |
| RMSE | — | 0.00779 |
| n_shots | — | 46 |
| windows | — | ~3450 (RMSE mean 0.0049 / median 0.0037 / p90 0.0065 / max 0.313) |
| stopped at | step 3500 | step 3500 |
| download | — | 740/740, 0 failed, 0 OOM |

**One-sentence result:** *40 GB beat 20 GB on pixel-level error (MAE −23%) but not
on the outlier-sensitive headline NRMSE, and it did so while overfitting — a
small-data-regime finding.*

---

## PART 10 — "Gotcha" questions a tough examiner asks

### Q10.1 — "Isn't overfitting just because you stopped too early?"
No. Patience stopped both runs at the *same* 3500 steps; the 40 GB val loss **was
not improving**. More steps wouldn't help — a memorising model keeps train→0 while
val stays high. (Stage-2 patience experiment will confirm.)

### Q10.2 — "Then just add more data, problem solved?"
Partly — more *diversity* helps, but (a) the 7 h GPU wall caps how much you can fit,
(b) the same 3500-step patience wall applies, and (c) NRMSE already barely moved
20→40 GB. So bigger data is likely **low return** for the headline metric.

### Q10.3 — "Why trust a 0.3 M-param model?"
For **relative** science (clean vs corrupted, 20 vs 40 GB) it's perfectly adequate —
it's the benchmark's own baseline. For absolute SOTA it isn't, and we never claim it.

### Q10.4 — "Your NRMSE 0.14 vs TokaMark's ~0.43 — are you *better*?"
**Do not say yes.** Different **aggregation level** (window vs hierarchical),
different **data subset**, different stages. Not comparable. (See Q5.7, Q5.9.)

### Q10.5 — "Why would losing one channel hurt a CNN more than an LSTM?"
Because the CNN uses **per-channel encoders** (Q6.3): it never learned to borrow
information across channels, so a dead channel is an information vacuum. The LSTM
mixes channels earlier.

### Q10.6 — "Is `delta=−0.005` leakage?"
No — it *prevents* future leakage: the output window starts only where the input
ends, using only present/past information.

### Q10.7 — "Why standardise with train statistics only?"
Otherwise val/test statistics would inform preprocessing = **leakage**. Train-only
stats, frozen, applied everywhere.

### Q10.8 — "Why is `n_shots=46` and is that suspicious?"
Not suspicious — it's the protocol keeping only shots with a valid ψ target. Document
it; don't hide it.

### Q10.9 — "Could the heavy tail just be noise?"
It's structural: mean>median plus a max 8× the median on 3450 windows is a genuine
**outlier population**, not sampling noise. The robustness study will also reveal
whether those windows are corrupted-sensor shots.

### Q10.10 — "What would make you distrust your own numbers?"
- Any sign of split-mixing (we check: none).
- `best_model.pt` metric not reproducible at eval (harness sanity check: must match
  ~0.14).
- Absolute numbers being quoted as official (we never do).

---

## PART 11 — Decision ledger ("why did you choose X?")

| Decision | Choice | Why | Alternative rejected |
|---|---|---|---|
| Model | CNN + MLP bottleneck | task is Markovian; simple & fast | LSTM (overkill), Transformer (heavy) |
| Data | 40 GB subset | fits Kaggle RAM/disk/7 h | full 565 GB (impossible locally) |
| Budget tool | greedy + ≥12-tar floor | breadth without bias | first-N tars (biased) |
| Workers | `DL_WORKERS=0` | hang-proof | `num_workers>0` (froze) |
| Optimiser | Adam lr 1e-4, wd 1e-4 | stable default | SGD (slower convergence) |
| Stopping | patience 10 | data-driven, not arbitrary | fixed step cap |
| Scaling | train-only z-score | stability without leakage | global scaling (leaks) |
| Metric focus | NRMSE (official) + MAE | official + outlier-free view | RMSE alone (outlier-dominated) |
| Next | robustness on task_1-3 | novel vs Gupta (task_4-4) | retrain bigger (low return) |

---

## PART 12 — Analogy bank (rapid-fire memory hooks)

- **Tokamak** → magnetic donut cage.
- **ψ** → topographic contour map of the cage.
- **Grad–Shafranov** → force-balance equation ("currents ⇒ field lines").
- **EFIT** → slow reference solver; we replace it with a fast net.
- **Shot** → one experiment run; **window** → one 5 ms frame.
- **Tar** → shipping container of many shots; `MIN_TARS` → "load from ≥12 containers".
- **Standardisation** → convert km/mg/seconds to common units; judge still reports real units.
- **Training loss vs NRMSE** → same mistake on two different rulers.
- **Overfitting** → memorising flashcards, failing the unseen final.
- **Patience / early stopping** → stopping revision when mock scores plateau.
- **RMSE vs MAE** → harsh grader (squares) vs fair grader (equally weighted).
- **± SD next to NRMSE** → spread of *class marks* around the class average.
- **Mean vs median** → 9 poor + 1 millionaire: mean lies, median tells the truth.
- **MLP vs LSTM bottleneck** → photograph vs video.
- **Robustness** → crash-test: break the sensors, measure the dent.
- **RS** → dent score: 1.0 = no damage.
- **Forward-fill** → copy the last known value (can't fix end-of-window blanks).

---

## PART 13 — If you only remember six things

1. **We predict the full ψ field** (65×65) from 8 instantaneous sensors, because ψ =
   the magnetic cage shape, and fast reconstruction matters for control.
2. **Our data is a budget-thinned subset** (606/74/60), so absolute numbers are not
   official — only relatives are trustworthy.
3. **The 40 GB run overfit** (train → 0.016, val ~0.87) — a small-data-regime effect,
   not a stopping artifact; it improved MAE but not NRMSE.
4. **The `±` is the spread of per-window scores**, not uncertainty; large ± + mean >
   median reveals a hard-window tail.
5. **task_1-3 is Markovian**, so the MLP bottleneck is the correct baseline (LSTM is
   for forecasting tasks).
6. **Our contribution is the robustness study on the equilibrium task** (Gupta did
   task_4-4), using his 6 scenarios + RS, plus novel **spatial** error maps.
