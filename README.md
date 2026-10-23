# PATU Forestry Crane: Neural Dynamics Modelling and Predictive Control

This repository documents a reproducible research workflow for a PATU forestry crane: **four neural dynamics models**, a separately trained **Compact control predictor**, conventional **model predictive control (MPC)**, and a final **neural-assisted MPC** controller. The results and commands below refer to the simplified 17-script project layout described in the supplied project README. Raw recordings and large intermediate datasets are stored outside the repository.

> **Scope of the control results:** The controller's prediction model and the separately advanced virtual crane plant use the **same frozen learned Compact checkpoint**, with separate state histories. Consequently, this is a *shared-surrogate closed-loop simulation*, not independent multibody-physics or experimental crane validation.

## 1. Repository layout and setup

```text
PATU_Crane_Research/
├── README.md
├── Results/
│   ├── GRU/                 preprocess.py  train.py  test.py  checkpoints/  results/
│   ├── LSTM/                preprocess.py  train.py  test.py  checkpoints/  results/
│   ├── Transformer/         preprocess.py  train.py  test.py  checkpoints/  results/
│   ├── Compact/             preprocess.py  train.py  test.py  checkpoints/  results/
│   └── Model_Comparison/    compare.py  results/
├── MPC/                     mpc.py  run_mpc.py  requirements.txt  configs/  checkpoints/  results/
└── Neural_MPC/              train.py  test.py  checkpoints/  results/
```

There are **17 main Python scripts** in this intended layout: 12 across the four model folders, one comparison script, two MPC scripts, and two neural-MPC scripts. Run commands from the project root. The simplified-package README identifies **64-bit Python 3.12** on Windows as its verified environment.

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r MPC/requirements.txt
```

Set `--data-root` to the original CSV location and `--prepared-root` to an external working/cache directory. The expected recordings are:

```text
D:\PATU_Data\
├── Train\Simulation_Output_00.csv ... Simulation_Output_42.csv  (43 recordings)
└── Test\Simulation_Output_43.csv ... Simulation_Output_53.csv   (11 recordings)
```

The 54 recordings contain **22 measured/simulated physical-state signals** and **two joystick commands**, `ulift[V]` and `ujib[V]`. The 22 predicted physical states, in the source-defined order, are:

```text
speedref[rad/s], speedpmsm[rad/s], iqref[A], udref[V], uqref[V],
id[A], iq[A], psid[Wb], psiq[Wb], pumppressure[Pa],
plinepistonlift[Pa], plinerodlift[Pa], ppistonsidelift[Pa], prodsidelift[Pa],
thetalift[rad], plinepistonjib[Pa], plinerodjib[Pa], ppistonjib[Pa],
prodjib[Pa], thetajib[rad], thetaliftdot[rad/s], thetajibdot[rad/s]
```

Two different data partitions serve different purposes. For **dynamics training**, the first **90% of each of the 43 Train recordings** supplies training samples, and the last **10%** supplies validation samples. For **neural-controller training**, **34 complete Train recordings** supply MPC demonstrations and **nine other Train recordings** supply development-validation cases (IDs `05, 08, 10, 13, 18, 22, 37, 39, 40`). The **11 Test recordings** are not used to fit scalers or train/select checkpoints in the described reproduction. 
## 2. Stage I — Four neural dynamics models

### 2.1 Preprocessing and the 48 input features

All four candidate architectures learn the same task: given the recent crane condition and joystick commands, **predict changes in the next 22 physical states**. The original sampling period is **10 µs** (100,000 samples/s), *not* the 10 ms used later in the controller.

Preprocessing removes the first **18 s** (`1,800,000` samples) of every recording; aligns the physical states and lift/jib commands; smooths the original-stage commands using a **Savitzky–Golay filter (51 samples, polynomial order 3)**; and normalizes the 24 signals using statistics fitted from the training portions only. The same saved scaler is used for validation or Test evaluation. The preprocessing cache holds the 24 aligned signals; the 48-feature inputs are constructed as needed, rather than stored as a second large dataset.

| Prepared input | Features | Interpretation |
|---|---:|---|
| Current 22 physical states | 22 | Electrical, hydraulic and mechanical condition |
| Current lift and jib joystick commands | 2 | Control signals applied at the current step |
| Change in the 22 states since the previous step | 22 | Recent state evolution |
| Change in the two joystick commands | 2 | Recent command variation |
| **Total** | **48** | Current 24 signals plus their 24 first differences |

Thus, each input contains the *current condition and recent changes*. The output consists of **22 physical-state increments**, not joystick predictions. These increments are multiplied by the saved training-derived state-increment scales and added to the current normalized physical states to obtain the next prediction. The original 24-column ordering, including the positions of the joystick columns, is defined by the preprocessing code.

Prepare the shared training cache once:

```powershell
python Results/GRU/preprocess.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --training-only
```

Any of the other model-folder `preprocess.py` scripts implements the same preparation. Omit `--training-only` when preparing the Test recordings with the saved Train scaler. To evaluate an original checkpoint, specify its **original frozen scaler** (for example `--frozen-scaler Results/GRU/checkpoints/original_scaler.json`); do not refit normalization on Test data. The model `test.py` entry points can prepare missing arrays using the selected checkpoint's saved scaler.

### 2.2 Neural network architectures

The **encoder** compresses the incoming measurements into smaller learned features, and the recurrent or attention mechanism processes temporal information before the decoder predicts state increments. These are *internal learned representations*, not separately trained latent-state autoencoders.

| Component | GRU | LSTM | Causal Transformer | Original Compact |
|---|---|---|---|---|
| Input features | 48 | 48 | 48 | 48 |
| Input/physical-state encoder | 48 → 12 | 48 → 12 | 48 → 16 | 22 physical states → 8 |
| Temporal mechanism | 16-unit GRU cell | 16-unit LSTM cell | 1 attention layer, 2 heads; 64-step causal history | Custom 8-feature recurrent transition |
| Additional processing | 16-unit post-recurrent layer | 16-unit post-recurrent layer | Feedforward width 32 and 16-unit post-attention layer | 32-unit hidden layer |
| Output | 22 state increments | 22 state increments | 22 state increments | 22 state increments |
| Direct input-to-output skip | Yes | Yes | Yes | Yes |
| **Trainable parameters** | **4,098** | **4,946** | **4,710** | **4,246** |

**GRU:** A 48-to-12 feature encoder feeds a 16-unit GRU memory cell. A 16-unit post-processing layer and output layer predict 22 state increments. An additional initialization encoder reads the 22 physical states to initialize memory, and a direct 48-to-22 linear skip contributes to the output.

**LSTM:** The same 48-to-12 feature path feeds a 16-unit LSTM with separate hidden and cell memory. A physical-state encoder initializes both memory components. A 16-unit output-processing layer and direct skip produce the 22 increments.

**Causal Transformer:** The 48 inputs are embedded in width 16, processed with one causal self-attention layer (two heads, feedforward width 32, dropout 0), and decoded through a 16-unit layer. Its context covers at most 64 previous steps. Causality prevents attending to future states.

**Original Compact:** A 22-to-8 encoder initializes its learned internal memory. At every step, a custom 8-dimensional transition combines current 48-feature input with prior memory; a 32-unit hidden layer and output decoder predict 22 increments, also using a direct linear skip. This is a compact *recurrent dynamics model*, not a separately trained latent-space control system.

**Important:** The Compact candidate in this section has **4,246 parameters**, which is slightly *more* than the **4,098-parameter GRU**, but fewer than the LSTM and Transformer. It is **not** the later 9,916-parameter Compact model used for control.

### 2.3 Training method and settings

Models are trained on **recursive rollouts** as well as short next-step predictions: their own predicted physical states feed subsequent steps so the loss captures error accumulation. Recorded joystick input continues to provide the control sequence. A short two-epoch warm-up precedes longer training rollouts.

| Training setting | Recovered value |
|---|---:|
| Original dynamics seed | 42 |
| Maximum epochs | 60 |
| Batch size / steps per epoch | 16 / 20 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.001 / 1e-6 |
| Gradient clipping / gradient block | 1.0 / 50 steps |
| Warm-up | First 2 epochs, 5-step rollouts |
| Main rollout choices | 100 / 500 / 1,000 steps |
| Rollout sampling probabilities | 20% / 60% / 20% |
| Corresponding physical durations | 1 / 5 / 10 ms |
| Recursive burn-in | Up to 10,000 steps; 25% zero-prefix cases |
| Mechanical angle and angular-velocity loss weights | 4 each |
| Other physical-state loss weights | 1 each |
| Checkpoint criterion | Weighted 500-step validation over four fixed windows |
| Learning-rate scheduler / early stopping | None / None |

The four upweighted mechanical states are the **lift angle, jib angle, lift angular velocity and jib angular velocity**. This emphasizes mechanical movement during learning; it does not mean the other 18 physical states are omitted. The increment-scaling routine uses **10,000 sampled Train-prefix pairs per recording**, with scale clipping to **[1e-5, 1]**. The checkpoint selection procedure is separate from Test evaluation.

Train independently:

```powershell
python Results/GRU/train.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --seed 42
python Results/LSTM/train.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --seed 42
python Results/Transformer/train.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --seed 42
python Results/Compact/train.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --seed 42
```

Training writes a **new checkpoint**, its associated scaler, configuration, epoch history, `training_loss.csv`, and loss figure to the corresponding folder. Original frozen checkpoints are preserved. A different random seed may lead to different weights and metrics. For a reduced end-to-end training check, add `--smoke-test` (three reduced epochs with forward/backward passes, validation, checkpoint save and reload); this **does not establish a full 60-epoch reproduction**.

### 2.4 Frozen-checkpoint testing and comparison

The four original checkpoints were evaluated on **11 Test recordings** (43–53) using the same sampling and evaluation procedure. The protocol uses **7,831 rollout origins**, selected at **10,000-raw-sample spacing**, plus the last valid origin of each recording. A **one-step** prediction scores the next physical state; a **50-step recursive rollout** feeds predicted states back while supplying the recorded joystick commands, without injecting future measured physical states. **50 × 10 µs = 0.5 ms**, *not* 500 ms.

The reported NRMSE divides errors by the *training-data standard deviation* for each physical state, averages recordings equally per state, and then averages the 22 state scores equally. Lower NRMSE is better; these tiny short-horizon errors are **not accuracy percentages**. R² can be undefined for effectively constant states.

| Model | Parameters | One-step NRMSE ↓ | 50-step NRMSE ↓ | One-step inference | 50-step inference |
|---|---:|---:|---:|---:|---:|
| GRU | 4,098 | 5.419649e-6 | 1.503523e-4 | 4.303 µs | 73.786 µs |
| LSTM | 4,946 | 5.413840e-6 | 1.531134e-4 | 4.470 µs | 79.741 µs |
| Transformer | 4,710 | 5.786726e-6 | 1.598183e-4 | 6.409 µs | 380.631 µs |
| **Original Compact** | **4,246** | **5.334929e-6** | **1.497528e-4** | **4.002 µs** | **67.180 µs** |

The original Compact achieved the lowest **mean** one-step and 50-step NRMSE and the lowest measured native inference time. The differences in errors are small and it does not win every individual signal. Timing values are warmed, single-thread native inference benchmarks, not complete controller-loop times.

#### Per-state 50-step recursive NRMSE

The following table includes **all 22 physical-state outputs**, averaged over the 11 Test recordings. Values are rounded for presentation; bold indicates the lowest value in a row. For nearly tied values, refer to full-precision exported CSVs.

| Physical State | GRU | LSTM | Transformer | Compact |
|---|---:|---:|---:|---:|
| `speedref[rad/s]` | 8.78e-7 | 4.08e-6 | **3.50e-7** | 2.88e-6 |
| `speedpmsm[rad/s]` | **6.60e-6** | 7.22e-6 | 6.69e-6 | 7.66e-6 |
| `iqref[A]` | 2.05e-6 | 1.33e-6 | **1.22e-6** | 3.17e-6 |
| `udref[V]` | **1.06e-6** | 1.44e-6 | 2.07e-6 | 2.12e-6 |
| `uqref[V]` | **1.10e-6** | 1.66e-6 | 1.16e-6 | 4.29e-6 |
| `id[A]` | **3.24e-6** | 4.43e-6 | 3.52e-6 | 5.14e-6 |
| `iq[A]` | 1.26e-6 | **1.02e-6** | 1.66e-6 | 3.59e-6 |
| `psid[Wb]` | 1.33e-6 | 1.15e-6 | **4.71e-7** | 4.58e-6 |
| `psiq[Wb]` | 1.54e-6 | **1.40e-6** | 1.57e-6 | 3.63e-6 |
| `pumppressure[Pa]` | 1.75e-6 | **1.09e-6** | 1.67e-6 | 3.43e-6 |
| `plinepistonlift[Pa]` | 3.31e-4 | 3.37e-4 | 3.46e-4 | **3.28e-4** |
| `plinerodlift[Pa]` | 1.14e-4 | 1.18e-4 | 1.14e-4 | **1.10e-4** |
| `ppistonsidelift[Pa]` | 3.31e-4 | 3.37e-4 | 3.46e-4 | **3.28e-4** |
| `prodsidelift[Pa]` | 1.14e-4 | 1.18e-4 | 1.14e-4 | **1.10e-4** |
| `thetalift[rad]` | **1.62e-6** | 2.18e-6 | 1.64e-6 | 3.10e-6 |
| `plinepistonjib[Pa]` | 1.15e-4 | 1.26e-4 | 1.37e-4 | **1.02e-4** |
| `plinerodjib[Pa]` | 1.28e-4 | 1.28e-4 | 1.27e-4 | **1.26e-4** |
| `ppistonjib[Pa]` | 1.15e-4 | 1.26e-4 | 1.36e-4 | **1.02e-4** |
| `prodjib[Pa]` | 1.28e-4 | 1.28e-4 | 1.27e-4 | **1.26e-4** |
| `thetajib[rad]` | **1.49e-6** | 1.65e-6 | 1.66e-6 | 1.88e-6 |
| `thetaliftdot[rad/s]` | **7.35e-4** | 7.48e-4 | 7.45e-4 | 7.41e-4 |
| `thetajibdot[rad/s]` | 1.174e-3 | **1.172e-3** | 1.299e-3 | 1.177e-3 |

The overall advantage of the Compact model is most evident in its average and its hydraulic-pressure predictions; GRU leads on the lift/jib angles and lift angular velocity, and LSTM on jib angular velocity. The angular-velocity states have the largest normalized short-horizon errors. **These 0.5 ms forecast scores cannot by themselves establish long-term physical-model accuracy.** The 10 ms controller predictor introduced below is a separately trained model, so its control results should not be treated as a direct extension of these scores.

Run the original frozen evaluation in each model folder (use `--checkpoint original` if that is the selector in the current release); to explicitly evaluate a newly trained checkpoint:

```powershell
python Results/GRU/test.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --checkpoint Results/GRU/checkpoints/gru_retrained_s42.pt
python Results/Compact/test.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --checkpoint Results/Compact/checkpoints/compact_retrained_s42.pt
python Results/Model_Comparison/compare.py
```

Each model's test stage prints and saves summary scores, per-state and per-recording metrics, inference timing, and prediction figures. Newly trained checkpoint results must stay separate from original frozen-checkpoint results. The comparison script reads compatible saved results rather than mixing normalization or evaluation protocols.

![Four-model accuracy](Results/Model_Comparison/results/accuracy.png)
![Four-model inference timing](Results/Model_Comparison/results/latency.png)
![Accuracy versus native inference time](Results/Model_Comparison/results/accuracy_computation.png)

## 3. Stage II — Compact predictor selected for control and crane geometry

The final MPC does **not** operate with the original 4,246-parameter, 10 µs Compact candidate. It uses a **separate 9,916-parameter frozen Compact predictor** trained for **10 ms** state advances (`1,000 × 10 µs`). This reduces the forecast depth needed for practical MPC. The checkpoint is:

```text
MPC/checkpoints/compact_norm_integrated_s44_dt1000.pt
SHA256: 590028f9aa7c1e660f4a09ec4b7563619ddcb9ebcee34eb694b589d33dc942f5
```

### 3.1 How the improved model handles mechanical motion

The model predicts all **22 physical states** given current states, recent changes and candidate lift/jib commands. Relative state differences are separately scaled. For the **lift and jib angles**, the improved model uses the current and predicted angular velocities and the **10 ms step duration** to integrate angles using their average angular velocity (a trapezoidal update). This keeps predicted angle change connected to predicted motion rather than using unconstrained angle increments alone. The original saved normalization and state-range handling are retained. The control-stage forecast and virtual-plant advancement use **raw, unsmoothed command histories**; they do not apply the four-model preprocessing smoother to newly optimized commands.

### 3.2 Converting joint angles to crane-tip position

MPC tracks the crane tip in the **horizontal–vertical (Cartesian X–Z) plane**. The geometry uses:

| Component | Calibrated value |
|---|---|
| Fixed base (X, Z) | (−0.09, 1.4261) m |
| Lift boom local vector | (2.886080943, 0.019892554) m |
| Jib-to-tip local vector | (2.035, 0.245) m |

Forward kinematics separately rotates the lift and jib vectors by their corresponding measured or predicted angles and adds both to the fixed base. **Both angles are world-frame absolute angles**: the jib angle is not added to the lift angle. The reference and predicted trajectories use the *same* geometric conversion.

### 3.3 Where the reference trajectory comes from

The **recorded CSVs** provide the desired future lift and jib angles and angular velocities. Their angles are converted to a desired **X–Z tip trajectory**. MPC uses this future trajectory as a **reference**, but **does not replay the recorded joystick commands as its control output**. Instead it generates new bounded lift and jib command pairs using the current virtual-plant state and a frozen neural forecast of their consequences. Future reference information is available to the planner; **future actual virtual-plant states are not**.

## 4. Stage III — Conventional model predictive control

MPC repeatedly selects lift/jib joystick commands by predicting future crane motion and minimizing a cost measuring reference-tracking error and undesirable command behavior. Every **100 ms**, it observes the latest **22 virtual-plant states**, optimizes a command plan over its longer forecast horizon, applies **only the first 100 ms command pair**, then advances the plant and replans. This is *receding-horizon control*.

### 4.1 MPC200 versus MPC400

| Configuration | MPC200 | MPC400 (selected) |
|---|---:|---:|
| Internal model step | 10 ms | 10 ms |
| Prediction horizon | 200 ms | 400 ms |
| Forecast steps | 20 | 40 |
| Replanning/control interval | 100 ms | 100 ms |
| Move block 1 | 100 ms | 100 ms |
| Move block 2 | 100 ms | 300 ms |
| Optimized quantities | Lift and jib for each block = **4 scalars** | Lift and jib for each block = **4 scalars** |
| Optimizer | SLSQP | SLSQP |
| Max solver iterations | 5 | 5 |
| Solver `ftol` | 1e-8 | 1e-8 |
| Finite-difference `eps` | 0.03 | 0.03 |
| Prediction/virtual-plant model | Frozen 9,916-param Compact | Frozen 9,916-param Compact |

The **400 ms horizon** gives a longer view of the intended motion without increasing the number of decision variables: it still adjusts only two joystick pairs. The second pair is held over the remainder of the forecast, but it is *not* blindly sent to the plant; optimization runs again after the first 100 ms.

**Why not use a 50 ms prediction horizon?** With the selected **100 ms control interval**, a 50 ms forecast ends *before the currently chosen command stops being applied*. Such an objective cannot assess the full held-command effect and supplies less anticipation. A **50 ms control interval** would be a different, more frequently optimized design; it was **not** part of the reported selected comparison. Do not interpret this as an experimentally demonstrated failure of a 50 ms controller.

### 4.2 MPC objective and its actual weights

Every candidate joystick plan is propagated through the frozen neural dynamics and converted to predicted X–Z tip positions. The numerical objective is a **weighted sum of mean squared normalized tracking and command terms** (plus a terminal error), not a raw unnormalized sum over 40 steps.

| Cost term | Weight | How it contributes |
|---|---:|---|
| **Tip-position tracking** | **1.0** | Mean squared difference between predicted and recorded X–Z tip coordinates, each divided by **0.01 m (10 mm)** |
| Joint-angle tracking | **0.0** | Disabled in the selected run; angle information still affects tip position via geometry |
| **Angular-velocity tracking** | **0.1** | Mean squared lift/jib velocity error after division by the saved corresponding standard deviations |
| Joystick effort | **0.0001** | Penalizes squared command magnitude after normalization to the command bounds |
| Joystick changes | **0.001** | Penalizes changes from the previous applied command to block 1 and from block 1 to block 2 |
| **Terminal tip-position tracking** | **5.0** | Additional squared normalized X–Z tip error at the **last** forecast step |

The **tip-position scale of 10 mm is a cost normalization choice**, not a guaranteed allowable tracking error. The angular-velocity term adds information about *how* the booms should move; the terminal term adds emphasis on *where the tip should end the planned horizon*. The joystick penalties discourage unnecessarily large or abruptly changing commands. They are soft penalties, **not certified physical safety limits**.

The bounded joystick values are approximately **−1 V to +1 V**, with original observed, slightly asymmetric channel bounds preserved in code. Command-rate and hydraulic-pressure constraints are not independently enforced in the selected configuration. SLSQP starts from a shifted prior plan, evaluates neural rollouts and cost values, and keeps the best feasible candidate found even when the numerical solver reports non-success. The setup uses a numerical gradient step of `0.03` and an iteration budget of five, which does **not** imply only five forward model evaluations.

### 4.3 MPC tracking results and selected controller

MPC200 and MPC400 were compared on **nine development-validation recordings** (IDs `05, 08, 10, 13, 18, 22, 37, 39, 40`). Tip RMSE measures the Euclidean Cartesian difference between the virtual crane tip and its recorded reference, with each recording contributing equally to the reported mean.

| Controller | Prediction horizon | Mean tip-position RMSE ↓ |
|---|---:|---:|
| MPC200 | 200 ms | 521.21 mm |
| **MPC400** | **400 ms** | **54.75 mm** |

MPC400 reduced mean tip-position RMSE by **approximately 89.5%** relative to MPC200 in this development comparison. It was selected for the neural-assisted stage. Its separate **136-context matched, warmed, single-thread** benchmark showed **43.05 ms mean** controller computation and **79.59 ms p99**, against a **100 ms replanning interval**. This timing excludes unrelated whole-simulation overhead and is machine-dependent; it is not a formal hard real-time certification.

### 4.4 Run MPC

The `MPC/mpc.py` file contains the selected Compact dynamics implementation, kinematics, MPC cost and solver, and shared-model virtual plant. `MPC/run_mpc.py` runs comparisons and saves per-recording trajectories, tracking errors, joystick histories, solver diagnostics and timing tables/plots under `MPC/results/`.

```powershell
python MPC/run_mpc.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --controller both
python MPC/run_mpc.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --controller mpc400
python MPC/run_mpc.py --verify
```

Use `--recompute` to rerun saved full experiments; `--smoke-test` is a limited functional check. Cached runs should be accepted only after validation of recording durations and trajectory/configuration hashes.

![MPC Cartesian trajectories](MPC/results/comparison/trajectory.png)
![MPC mechanical states](MPC/results/comparison/mechanical_states.png)

## 5. Stage IV — Offline neural policy and neural-assisted MPC

### 5.1 What is learned and why

The final system uses a **frozen neural command predictor** to provide a quick *starting guess* for MPC; it does **not** replace the MPC optimizer. A supervised network learns from original **MPC400 joystick decisions** collected offline, then proposes one lift/jib pair when the online controller is called.

| Neural-policy input | Number of features |
|---|---:|
| Current 22 physical states | 22 |
| Previous 22 physical states | 22 |
| Previously applied lift/jib commands | 2 |
| Four current mechanical reference-tracking errors | 4 |
| 40 future reference points × four mechanical values | 160 |
| **Total** | **210** |

The four future reference variables are lift angle, jib angle, lift angular velocity and jib angular velocity. The **wide_s44** policy architecture is **210 → 128 → 64 → 2**, with ReLU hidden layers and tanh outputs, and **35,394 trainable parameters**. The two outputs are candidate lift and jib joystick commands. This controller's 210 features are separate from the 48 features used in Stage I dynamics modelling.

### 5.2 Collect MPC teacher demonstrations and train the policy

The training entry point `Neural_MPC/train.py` contains **both** teacher generation and policy training. The original MPC400 teacher uses the **same selected 9,916-parameter Compact dynamics, 400 ms cost, joystick limits, five-iteration SLSQP budget and historical warm-start convention**. At each 100 ms MPC update, it records the 210 features and the teacher's optimized **first** lift/jib command pair. The resulting demonstrations are split according to **34 Train recordings for neural fitting** and **nine other Train recordings for validation**; the distinct 11 Test recordings are not used in this process.

| Demonstration subset | Samples |
|---|---:|
| MPC teacher training examples | 25,644 |
| Teacher development-validation examples | 6,644 |
| **Total teacher examples** | **32,288** |

Teacher collection requires tens of thousands of optimizations and can be expensive. The recovered pipeline preserves the historical **previous-second-move warm start** rather than replacing the teacher's decisions with a new zero/hold-initialization convention. The model input/scaler construction also preserves the historical float64/float32 conversion sequence when replaying the original checkpoint.

| Neural-policy training setting | Value |
|---|---:|
| Model / parameters | 210 → 128 → 64 → 2 / 35,394 |
| Loss | Mean squared error of normalized joystick commands |
| Optimizer / learning rate | Adam / 0.001 |
| Weight decay | 1e-5 |
| Batch size | 512 |
| Max epochs / early-stopping patience | 240 / 40 |
| Minimum validation improvement | 1e-7 |
| Gradient clipping | 5 |
| Selected seed | 44 |
| Reported completed fitting run | 72 epochs |

The supplied reproduction report describes the seed-44 retrained policy as **bit-identical** to the original frozen policy in its verified software/numerical environment, including saved scaler tensors. That assertion is a **reported result for that environment**, not a guarantee that new data, another implementation or hardware produces identical weights. The original `wide_s44` checkpoint remains the default; newly fitted files must not overwrite it.

```powershell
# Generate MPC teacher decisions and train the 210-input neural policy
python Neural_MPC/train.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --seed 44

# Or train with independently stored, validated teacher demonstrations
python Neural_MPC/train.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --reuse-demonstrations "D:\PATU_Teacher_Demonstrations" --seed 44
```

Teacher demonstrations are stored **outside the ZIP**; the reuse path must contain compatible protocol metadata and records. A verified historical-demo fit and a bounded teacher-generation smoke test are not the same as demonstrating a complete fresh 32,288-optimization teacher regeneration.

### 5.3 Final online neural-assisted MPC loop

At each **100 ms control update**:

1. Read the virtual crane's current/previous states, previous joystick pair, current tracking errors, and **400 ms future mechanical reference**.
2. The neural command predictor outputs **one lift/jib command pair**.
3. Copy this pair to initialize **both**  MPC move blocks (the first 100 ms, then the remaining 300 ms).
4. Use the **same MPC400 cost and frozen Compact model** to refine the four planned joystick values with SLSQP, now limited to **three iterations**.
5. Apply **only the first optimized joystick pair for 100 ms** to the separately advancing learned virtual crane plant.
6. Observe the updated state and repeat; no future actual plant states are supplied to the controller.

The conventional three-iteration MPC is an essential comparison: it gets the same reduced solver budget but starts from the traditional shifted MPC plan rather than the neural suggestion. This separates the benefit of **neural initialization** from the saving due merely to **fewer solver iterations**.

#### Conventional Three-Iteration MPC

To evaluate the contribution of neural initialization, a conventional three-iteration MPC controller was included as a comparison. It uses the same 400 ms prediction horizon, Compact dynamics model, cost function, and joystick constraints as the original MPC400, but reduces the maximum SLSQP iterations from five to three. Unlike neural-assisted MPC, it initializes optimization using the previous MPC command plan rather than neural-network predictions. This comparison helps determine whether the improvement comes from neural initialization or simply reducing the number of optimization iterations.

### 5.4 Final controller comparison

All three controllers were evaluated on the **nine full development-validation recordings** using the same learned-model virtual plant. **Mean tip RMSE** averages the nine per-recording errors. Controller timing was measured **separately**, on **136 identical matched input contexts**, after warm-up, using one CPU thread. The timing columns should not be read as whole-run wall time.

| Controller | Mean tip RMSE ↓ | Mean controller time ↓ | p99 controller time ↓ | RMSE improvement vs original | Mean-time saving vs original |
|---|---:|---:|---:|---:|---:|
| Original MPC400 (**5 iterations**) | 54.75 mm | 43.05 ms | 79.59 ms | Reference (0%) | Reference (0%) |
| Conventional MPC400 (**3 iterations**) | 131.20 mm | 27.70 ms | **49.87 ms** | **−139.6%** | +35.7% |
| **Neural-assisted MPC (3 iterations)** | **43.31 mm** | **27.59 ms** | 51.10 ms | **+20.9%** | **+35.9%** |

**Interpretation:** Neural-assisted MPC achieved **20.9% lower mean tip-position RMSE** and **35.9% less matched average controller time** than original five-iteration MPC400. It improved six of nine recording-level RMSEs. Against *conventional three-iteration MPC*, neural assistance reduced mean RMSE by about **67.0%** with **essentially the same mean computation time** (27.59 versus 27.70 ms). Conventional three-iteration MPC had a marginally better **p99** time (49.87 versus 51.10 ms), and the hybrid's mean final tip-position error was **5.81 mm**, versus **1.72 mm** for original MPC400. The hybrid observed **zero deadline misses among 6,644 closed-loop updates**, but this does not certify hard real-time performance.

The speed gain relative to five-iteration MPC comes **primarily from reducing SLSQP's iteration budget**; the neural network demonstrably improves tracking accuracy under that smaller budget, but its starting guess was *not* proven to consistently lower the initial objective or accelerate optimizer convergence. **The main contribution is a better accuracy–computation trade-off in a shared-surrogate control study**, not a verified standalone neural controller or physical-crane deployment.

### 5.5 Test the hybrid and save results

`Neural_MPC/test.py` evaluates original five-iteration MPC400, conventional three-iteration MPC, and three-iteration neural-assisted MPC. The historical frozen `wide_s44` policy is the default. To evaluate a separately retrained policy, pass its checkpoint path explicitly.

```powershell
python Neural_MPC/test.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --checkpoint original
python Neural_MPC/test.py --data-root "D:\PATU_Data" --prepared-root "D:\PATU_Cache" --checkpoint Neural_MPC/checkpoints/wide_retrained_s44.pt
```

The test stage prints and saves per-recording and summary tip-position errors, reference-versus-tracked Cartesian paths, angles and angular velocities, joystick time series, solver and saturation information, controller runtime distributions, deadline statistics, and actual numerical output tables. Files are written under `Neural_MPC/results/`. Where applicable the original MPC400 run can be read from the verified cache; use `--recompute` to repeat the optimized runs.

![Neural-assisted Cartesian trajectories](Neural_MPC/results/comparison/trajectory.png)
![Neural-assisted tip error](Neural_MPC/results/comparison/tip_error.png)
![Controller accuracy and computation](Neural_MPC/results/comparison/accuracy_computation.png)

## 6. Reproducibility and interpretation

- **Two Compact models are different:** the **4,246-parameter** model is the original **10 µs** four-model candidate; the **9,916-parameter** improved Compact is the selected **10 ms** MPC predictor and shared virtual-plant dynamics model.
- **Training versus evaluation:** frozen-model testing reproduces existing checkpoints' performance; from-scratch training tests whether a new checkpoint can be produced under the original protocol. A successful three-epoch smoke test does not demonstrate a full historical training reproduction.
- **Train/validation/Test:** the dynamics and controller training splits use different rules. The nine controller cases are **development-validation**, not an untouched independently held-out final controller Test set. The 11 dynamics Test files were previously examined in the research history.
- **Short-horizon versus control claims:** the four original models' 50-step rollouts cover **0.5 ms**; the improved control predictor takes **10 ms** steps in 200/400 ms MPC horizons. Their results answer different questions.
- **Simulation limitation:** both MPC optimization and virtual plant use the same frozen neural dynamics checkpoint. Separate histories do not make them independent models. Generalization to different physical dynamics, loads, sensor noise and real hardware remains to be evaluated.
- **Timing:** native model kernels and matched controller calculations are measured under particular thread, warm-up, software and hardware conditions. The reported 100 ms control interval and zero observed deadline misses do not establish worst-case hard real-time guarantees.

## 7. Verification status of this document

This README has been edited for clarity, non-repetition, numerically consistent tables and clear separation of experimental stages. **It does not claim an end-to-end rerun was performed during this edit.** Its training configuration, 17-script commands, full-run controller hashes and identical-policy fitting report come from the **supplied newer README**, while the **attached `PATU_Crane_Final_Research.zip` is an older, different layout** containing `src/patu_crane/` and root-level `run_*.py` scripts. That archive provides source-level support for the original model architectures, 48-feature input construction, 10 ms integrated-angle controller model and mean-squared MPC objective, but **cannot validate that the newer 17-script commands, new training entry points and final exported plots currently exist and run**. Verify those against the updated ZIP before publishing or distributing this README.

For questions: Sohaib.Saeed@lut.fi



For questions contact:
Sohaib.Saeed@lut.fi
