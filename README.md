# PATU Forestry Crane: 22-State Neural Dynamics and Neural-Assisted Predictive Control


This repository presents a three-stage investigation of a planar, electro-hydraulic PATU forestry crane: 

(1) comparing four neural networks that predict **22 physical states** from historical crane states and two joystick commands; 

(2) using a separately trained Compact recurrent dynamics model within **model predictive control (MPC)** to track a recorded Cartesian tip trajectory; and 

(3) using a frozen neural control policy to initialize a faster **neural-assisted MPC** optimizer. It is an inference and evaluation package, not a complete end-to-end retraining distribution.

> **Central result:** On nine reused Train-derived development-validation recordings, the selected neural-assisted MPC achieved **43.31 mm mean Cartesian tip RMSE** with **27.59 ms mean matched controller time**, compared with **54.75 mm** and **43.05 ms** for the original 400-ms MPC. This is a **20.9% reduction in mean tracking RMSE** and **35.9% reduction in matched computation time**. These results use a *shared learned neural-surrogate virtual plant*; physical-crane performance has not been established.

## 1. Repository layout

```text
PATU_Crane_Research/
├── README.md
├── Results/
│   ├── Four_Models/          # GRU, LSTM, Transformer, original Compact; checkpoints and per-model outputs
│   └── Model_Comparison/     # shared four-model tables and figures
├── MPC/                      # shared code; selected 9,916-parameter Compact; MPC200/MPC400
└── Neural_MPC/               # wide neural policy; conventional3; neural-assisted MPC; results
```


## 2. Crane state, inputs and dataset

The machine is represented by an ordered vector of **22 physical states**, and **two input commands**. The controller changes the lift and jib through bounded joystick values. The two commands are *not* among the 22 predicted output states. The original 24-column order contains joystick inputs at indices 0 and 16 (zero-based); the other columns are predicted states.

**22 predicted signals, in model order:**

| Index | Signal | Meaning |
|---:|---|---|
| 1 | `speedref[rad/s]` | electrical-machine speed reference |
| 2 | `speedpmsm[rad/s]` | electrical-machine rotational speed |
| 3 | `iqref[A]` | q-axis current reference |
| 4 | `udref[V]` | d-axis voltage reference |
| 5 | `uqref[V]` | q-axis voltage reference |
| 6 | `id[A]` | d-axis current |
| 7 | `iq[A]` | q-axis current |
| 8 | `psid[Wb]` | d-axis flux linkage |
| 9 | `psiq[Wb]` | q-axis flux linkage |
| 10 | `pumppressure[Pa]` | pump pressure |
| 11–14 | `plinepistonlift`, `plinerodlift`, `ppistonsidelift`, `prodsidelift` | lift hydraulic pressure signals, Pa |
| 15 | `thetalift[rad]` | lift angle |
| 16–19 | `plinepistonjib`, `plinerodjib`, `ppistonjib`, `prodjib` | jib hydraulic pressure signals, Pb |
| 20 | `thetajib[rad]` | absolute jib angle |
| 21 | `thetaliftdot[rad/s]` | lift angular velocity |
| 22 | `thetajibdot[rad/s]` | jib angular velocity |

The 54 original CSV files are organized as **43 Train files** (`Simulation_Output_00`–`42`) and **11 Test files** (`43`–`53`). The nine controller development-validation recordings are **05, 08, 10, 13, 18, 22, 37, 39 and 40**. The remaining 34 Train recordings were used for controller training. The Test recordings have been examined historically and are not an untouched independent test set.

### 2.1 Preprocessing

The original recording interval is **10 µs** (`0.00001 s`), corresponding to 100,000 raw samples per second. The preprocessing/configuration applies as:

1. Remove the first **18 seconds** of each recording as startup trim.
2. Use a **51-sample, third-order Savitzky–Golay** filter on the commands for the original four-model preparation.
3. Apply the **frozen Train-derived normalization/scaling** accompanying the trained models. Reproduction must not refit a scaler on validation/Test.
4. Preserve the original command alignment: previous state, current state and appropriately aligned commands inform the next-state prediction.
5. Keep an **unsmoothed raw command stream** for the later control experiments; do not substitute the smoothed four-model inputs into MPC without checking the original implementation.

#### Input feature preparation: 
Each model receives 48 input features, created from two groups:

1. 24 current values: The 22 physical crane states and 2 joystick commands at the current time step.

2. 24 changes: The difference between each of these 24 values at the current and previous time steps.

These two groups are combined into a single 48-feature input vector. This gives the models information about both the crane's current condition and how its states and commands have recently changed. The models then use these features to predict the changes in the 22 physical states at the next time step.

## 3. Stage I — The four 22-state neural dynamics models

These four models learn how the crane changes after receiving joystick commands. They all predict 22 physical-state values, not new joystick decisions.

### 3.1 Frozen architecture summary

| Model | Parameter count | Main saved settings | Selected checkpoint epoch |
|---|---:|---|---:|
| GRU | **4,098** | hidden 16; latent 12; decoder 16; skip connection | 44 |
| LSTM | **4,946** | hidden 16; latent 12; decoder 16; skip connection | 32 |
| Causal Transformer | **4,710** | model width 16; 2 heads; 1 layer; feedforward 32; history 64; dropout 0 | 44 |
| Original Compact recurrent | **4,246** | latent 8; hidden 32 | 56 |

The archived training metadata gives a 60-epoch maximum, batch size 16, learning rate 0.001, seed 42, and validation horizons of 100, 500 and 1,000 **raw 10-µs steps**. It records weighted normalized MSE with mechanical-angle and angular-velocity weights of 4 for the original-model training setup. These are **archived training settings**.

#### GRU (4,098 parameters)

The GRU receives the current 24 signals (22 crane states and 2 joystick commands) and their changes from the previous time step. These 48 features are compressed into 12 encoded features and processed by a 16-unit GRU cell. The resulting information passes through a 16-unit decoding layer to predict changes in all 22 physical states. These changes are scaled and added to the current states to obtain the next prediction.

#### LSTM (4,946 parameters)

The LSTM uses the same 48 input features and 12-feature encoder as the GRU. Instead of a GRU cell, it uses a 16-unit LSTM cell with hidden and cell memory to capture previous crane behaviour. Its output passes through a 16-unit decoding layer to predict scaled changes in the 22 physical states, which are added to the current states.

#### Causal Transformer (4,710 parameters)

The Transformer converts the same 48 input features into 16-dimensional encoded representations. It maintains a history of up to 64 time steps and processes it using one Transformer layer with two attention heads. Positional encoding preserves time order, while causal attention prevents access to future states. The processed representation passes through a 16-unit decoding layer to predict scaled changes in the 22 physical states, which are added to their current values.

#### Original Compact model
The Compact model uses 48 input features, consisting of the current 22 crane states, 2 joystick commands, and their changes from the previous time step. Unlike the GRU and LSTM, it uses a custom recurrent mechanism. First, an encoder compresses the 22 physical states into 8 internal features. At each prediction step, these internal features are updated using the current 48 inputs and the previous internal memory. The updated information is combined with the inputs and passed through a 32-unit hidden layer. A decoder then predicts the changes in all 22 physical states. A direct linear skip connection from the 48 input features also contributes to these predictions. The predicted changes are scaled and added to the current states to calculate the next crane state.


### 3.2 Prediction Evaluation

The four neural models were evaluated using **11 Test recordings (Simulation_Output_43 to Simulation_Output_53)**. All models used their original trained checkpoints and frozen training-data normalization parameters without retraining.

Two prediction evaluations were performed:

- **One-step prediction:** Predicting the next 22 physical states using the available recorded information.
- **Recursive prediction:** Predicting 50 consecutive steps by feeding previously predicted states back into the model without using future measured physical states.

With a sampling interval of **10 µs**, the 50-step prediction represents **0.5 ms** of crane operation.

The evaluation included **7,831 sampled starting points** across the 11 Test recordings. NRMSE was calculated using the training-set standard deviation, averaged equally across the Test recordings and physical states.

#### Overall Prediction Accuracy

| Model | Parameters | One-step NRMSE | 50-step NRMSE |
|---|---:|---:|---:|
| GRU | 4,098 | 0.000005420 | 0.000150352 |
| LSTM | 4,946 | 0.000005414 | 0.000153113 |
| Transformer | 4,710 | 0.000005787 | 0.000159818 |
| **Original Compact** | **4,246** | **0.000005335** | **0.000149753** |

The original Compact model achieved the lowest overall NRMSE for both one-step and 50-step recursive predictions. It also achieved the lowest 50-step error on 11 Test recordings**.

#### Prediction Performance Across All 22 Physical States

The following table presents the average **50-step recursive prediction NRMSE** for all 22 physical states, evaluated across 11 Test recordings.

NRMSE measures prediction error relative to the standard deviation of each state in the training data. A lower value indicates better prediction accuracy. Values are averaged equally across the 11 recordings.

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

**Bold values** indicate the lowest NRMSE for each physical state.

All four models achieved low normalized prediction errors over the evaluated **0.5 ms horizon**. The Compact model achieved the lowest overall average NRMSE and performed particularly well on several hydraulic-pressure states. However, GRU achieved the lowest error for both lift and jib angles, while GRU and LSTM performed slightly better on the corresponding angular velocities.

Angular velocities showed the largest normalized prediction errors, particularly jib angular velocity. This suggests that accurately predicting changes in mechanical motion is more challenging than predicting slowly varying states over this short horizon.

These results describe short-horizon prediction performance and should not be interpreted as evidence of accurate long-duration physical simulation.


#### Inference Time Comparison

The models were benchmarked using one CPU thread with warmed execution. The measurements exclude preprocessing, data loading, initial memory encoding, buffer preparation and compilation.

| Model | Mean One-step Time | Mean 50-step Time |
|---|---:|---:|
| GRU | 4.303 µs | 73.786 µs |
| LSTM | 4.470 µs | 79.741 µs |
| Transformer | 6.409 µs | 380.631 µs |
| **Original Compact** | **4.002 µs** | **67.180 µs** |

The Compact model achieved the lowest average inference time for both evaluation settings.

Its 50-step rollout required **67.18 µs** on average to predict 500 µs of simulated crane movement, corresponding to approximately **7.4 times faster-than-real-time prediction** for the measured computation kernel.

Individual model results are available in `Results/Four_Models/results/`, while comparative figures are stored in `Results/Model_Comparison/figures/`.


---

## 4. Stage II — Selected Neural Surrogate and Crane Geometry

Following the four-model comparison, a separate improved **Compact recurrent model with 9,916 parameters** was selected for the MPC experiments. This differs from the original 4,246-parameter Compact model used in the initial comparison.

Unlike the original four models operating at 10 µs, the improved Compact model predicts at **10 ms intervals**, using a 1,000-raw-sample stride.

The model receives the current crane states, joystick commands and recent changes, and predicts changes in all 22 physical states.

It uses separate normalization for state differences and updates the lift and jib angles by integrating the predicted angular velocities over each 10 ms interval.

### 4.1 Role of the Neural Surrogate

The improved Compact model is used in two parts of the control experiments:

1. **MPC predictor:** Predicts how the crane responds to candidate joystick commands over the prediction horizon.
2. **Virtual crane plant:** Simulates the crane's evolving state under the commands applied by the controller.

Both use the same frozen trained model but maintain independent state histories. The virtual plant advances using the controller's generated commands and is never reset to future recorded measurements.

This arrangement allows MPC and neural-assisted MPC to be evaluated under identical simulated dynamics. However, the results represent **shared neural-surrogate simulation**, not independently validated physical crane performance.


### 4.1 Crane Tip Position Calculation

To control the crane, we need to know the position of its tip. However, the neural model predicts the **lift and jib angles**, not the Cartesian tip position directly. Therefore, we use **forward kinematics** to calculate the tip's horizontal (X) and vertical (Z) coordinates from these angles.

The calculation uses the crane's actual geometry:

- **Base position:** X = -0.09 m and Z = 1.4261 m.
- **Main lift boom:** Calibrated local vector of (2.8861, 0.0199) m.
- **Jib boom:** Calibrated local vector of (2.035, 0.245) m.

The tip position is calculated in three steps:

1. **Start from the crane base:** The calculation begins at the fixed base position (-0.09, 1.4261) m.
2. **Calculate the lift boom position:** The predicted lift angle is used to rotate the main boom's calibrated vector. This determines the horizontal and vertical displacement contributed by the lift boom.
3. **Calculate the jib boom position:** The predicted jib angle is used to rotate the jib boom's calibrated vector. Its displacement is added to the base position and lift boom displacement to obtain the final crane tip position.

Both angles are measured relative to the world coordinate system. Therefore, the jib angle is used directly and is **not added to the lift angle**.

This calculation is performed for the predicted crane states throughout the MPC prediction horizon. The resulting tip positions are compared with the reference trajectory to calculate the tracking error.

The MPC optimizer then uses this error to find joystick commands that move the crane tip closer to its desired position.


## 5. Stage III — Model predictive control (MPC)

### 5.1 Model Predictive Control (MPC)

Model Predictive Control (MPC) is used to automatically control the crane's lift and jib movements so that its tip follows a prerecorded reference trajectory.

The desired trajectory comes from the original CSV recordings. At each control update, MPC reads the current simulated crane states and the upcoming reference trajectory. It then tests possible joystick commands using the frozen Compact neural dynamics model to predict how the crane will move.

The predicted tip positions are compared with the reference positions, and SLSQP optimization selects the joystick commands that minimize the tracking error.

Only the first control command is applied to the virtual crane. After **100 ms**, the controller receives the updated crane states and repeats the optimization using a new future reference. This process is called **receding-horizon control**.

#### Prediction Horizon and Control Interval

Two MPC configurations were investigated: **MPC200 and MPC400**. The numbers 200 and 400 represent how far the controller predicts into the future, in milliseconds.

- **MPC200:** Predicts crane movement 200 ms into the future using 20 prediction steps.
- **MPC400:** Predicts crane movement 400 ms into the future using 40 prediction steps.

Both controllers use a **10 ms prediction step** and update the joystick commands every **100 ms**.

These three time settings have different purposes:

- **10 ms prediction step:** The time interval between consecutive predicted crane states.
- **100 ms control interval:** How frequently MPC calculates and applies a new joystick command.
- **200 or 400 ms prediction horizon:** How far ahead MPC looks when evaluating possible commands.

For example, MPC400 predicts 400 ms of future crane movement but only applies the selected joystick command for the next 100 ms. It then observes the new simulated state and calculates a new 400 ms prediction.

#### Why Not Use a 50 ms Prediction Horizon?

The control interval was set to **100 ms**, meaning the controller applies each selected command for 100 ms before optimizing again.

A 50 ms prediction horizon would only cover half of this control interval. Therefore, the optimizer would not evaluate the complete movement resulting from the command before the next control update.

For this reason, the evaluated prediction horizons were longer than the 100 ms control interval. MPC200 initially provided 200 ms of future information, while MPC400 was investigated to determine whether looking further ahead could improve trajectory tracking.

The longer horizon gives MPC more opportunity to anticipate upcoming movements, changes in direction, and the crane's delayed response to joystick commands.

A **50 ms control interval** was a separate possibility: but during the test it was found that the results were not good with shorter horizon because of the many optimization steps inside the preceding horizon.

#### MPC Configuration

| Parameter | MPC200 | MPC400 |
|---|---|---|
| Control update interval | 100 ms | 100 ms |
| Neural prediction step | 10 ms | 10 ms |
| Prediction horizon | 200 ms | 400 ms |
| Number of prediction steps | 20 | 40 |
| First joystick move | 100 ms | 100 ms |
| Second planned joystick move | 100 ms | 300 ms |
| Optimized joystick values | 4 | 4 |
| Optimization algorithm | SLSQP | SLSQP |
| Maximum solver iterations | 5 | 5 |
| Solver tolerance (`ftol`) | 1e-8 | 1e-8 |
| Finite-difference step (`eps`) | 0.03 | 0.03 |

#### How the Joystick Commands Are Optimized

Instead of calculating a separate joystick command for every 10 ms prediction step, MPC divides the prediction horizon into two control blocks.

For **MPC200**:

- First block: Lift and jib commands applied for the initial 100 ms.
- Second block: Lift and jib commands planned for the following 100 ms.

For **MPC400**:

- First block: Lift and jib commands applied for the initial 100 ms.
- Second block: Lift and jib commands planned for the following 300 ms.

Each block contains two joystick values: one for the lift boom and one for the jib boom. Therefore, both MPC configurations optimize only **four joystick values**, despite having different prediction horizons.

The second block allows MPC to consider future movement, but it is not automatically applied. Only the first 100 ms command is sent to the virtual crane before the next optimization.

This approach keeps the number of optimization variables small while allowing the controller to predict further into the future.

The 400 ms configuration was selected because it achieved substantially better trajectory tracking in the evaluated development-validation recordings, reducing mean tip-position RMSE from approximately **521.21 mm with MPC200** to **54.75 mm with MPC400**.



### 5.2 MPC Objective Function and Optimization

The MPC controller selects lift and jib joystick commands by minimizing a cost function. This cost function evaluates how closely the predicted crane movement follows the recorded reference trajectory while also considering movement speed, joystick effort, and command changes.

For every candidate joystick sequence, the Compact neural model predicts the crane's future states. The predicted lift and jib angles are converted into Cartesian tip positions using forward kinematics.

The controller then calculates the following costs:

| Cost Component | Weight | Purpose |
|---|---:|---|
| Tip position tracking | 1.0 | Minimizes the difference between predicted and reference tip positions |
| Joint angle tracking | 0.0 | Disabled because Cartesian tip tracking is the primary objective |
| Angular velocity tracking | 0.1 | Reduces differences between predicted and reference lift/jib angular velocities |
| Joystick effort | 0.0001 | Discourages unnecessarily large joystick commands |
| Joystick command changes | 0.001 | Discourages large changes from the previous command and between planned moves |
| Terminal tip position | 5.0 | Gives additional importance to the predicted tip position at the end of the horizon |

#### How the Cost Is Calculated

1. **Tip position error:** At each predicted time step, the controller calculates the horizontal and vertical distance between the predicted crane tip and the recorded reference tip. These differences are scaled using a 0.01 m (10 mm) reference scale, squared, and averaged across both coordinates and the prediction horizon.

2. **Angular velocity error:** The predicted lift and jib angular velocities are compared with their recorded reference values. The differences are normalized using the saved dynamics-scaler standard deviations, squared, and averaged.

3. **Joystick effort:** The planned lift and jib joystick commands are converted into normalized values using their allowed command ranges. Their squared values are averaged to discourage excessive control effort.

4. **Joystick changes:** The controller compares the first planned command with the previously applied command and the second planned command with the first. These changes are normalized, squared, and averaged to discourage abrupt control changes.

5. **Terminal position error:** The predicted tip position at the final prediction step is compared with the corresponding reference position. This error is scaled, squared, and averaged across the horizontal and vertical coordinates. Its higher weight encourages the controller to consider where the crane will be at the end of the prediction horizon.

Each component is multiplied by its specified weight, and the weighted components are added together to obtain the final MPC cost.

The optimizer searches for the joystick commands that produce the lowest cost.

#### SLSQP Optimization

The controller uses **Sequential Least Squares Programming (SLSQP)** to optimize four joystick values: lift and jib commands for each of the two planned control blocks.

The joystick commands are bounded approximately between **-1 V and +1 V**, based on the observed command range in the dataset.

The optimizer begins with a feasible initial command sequence and improves it by repeatedly evaluating the predicted crane movement and its cost.

The original MPC uses a maximum of **five SLSQP iterations**. Each iteration may require multiple neural-model evaluations, so five optimization iterations do not mean only five predictions.

If the solver reaches its iteration limit, the controller uses the best feasible candidate identified during optimization.

The same objective function, normalization, joystick bounds, and prediction procedure are maintained for MPC200, MPC400, and the later neural-assisted MPC experiments. The neural-assisted MPC changes the initial optimization guess and reduces the maximum iteration count to three.



### 5.2 Horizon comparison

The 11 development-validation recordings produced the following results:

| Controller | Mean-per-recording Cartesian tip RMSE |
|---|---:|
| MPC200 | 521.21 mm |
| MPC300 (historical horizon study) | 161.40 mm |
| **MPC400** | **54.75 mm** |

Moving from 200 ms to 400 ms reduced mean tip RMSE by approximately **89.5%** in this surrogate test. The 300-ms result is historical evidence; the simplified share package need not retain its experimental code.


## 6. Stage IV — Neural-Assisted Model Predictive Control

The final proposed controller combines a trained neural network with Model Predictive Control (MPC) to improve crane trajectory tracking while reducing online computation.

Instead of relying entirely on numerical optimization to determine the joystick commands, the neural network first provides an initial estimate. MPC then improves this estimate using the frozen Compact dynamics model and a reduced optimization budget.

This approach combines the fast prediction capabilities of neural networks with MPC's ability to evaluate future crane movements and correct control decisions.

### 6.1 Neural-Assisted MPC Architecture

The controller combines two trained neural models:

1. **Compact dynamics model (9,916 parameters):** Predicts how the crane's 22 physical states change under different joystick commands. It is used inside MPC to evaluate future movements.
2. **Neural command predictor (35,394 parameters):** Estimates suitable lift and jib joystick commands from the current crane conditions and desired trajectory.

The neural command predictor uses a fully connected network with **210 inputs, two hidden layers containing 128 and 64 neurons, and two joystick outputs**.

Its input information includes:

- Current crane states (22 features).
- Previous crane states (22 features).
- Previously applied lift and jib commands (2 features).
- Current mechanical tracking errors (4 features).
- Desired mechanical reference over the next 400 ms (160 features).

The network outputs two values: an initial command for the lift boom and an initial command for the jib boom.

These predicted commands are then passed to MPC for further optimization.

### 6.2 How Neural-Assisted MPC Works

The neural-assisted controller operates every **100 ms** using the following procedure:

1. **Read the current crane states:** Obtain the latest 22 physical states from the virtual crane and identify the upcoming reference trajectory from the recorded CSV file.

2. **Generate an initial control estimate:** The neural network processes the current states, previous states, joystick history, tracking errors, and future reference information to predict lift and jib joystick commands.

3. **Initialize the MPC optimizer:** The predicted joystick commands are used as the initial guess for both control blocks of the 400 ms prediction horizon.

4. **Predict future crane movement:** The Compact dynamics model predicts how the crane will respond to the proposed joystick commands over the next 400 ms, using 40 prediction steps of 10 ms.

5. **Optimize the joystick commands:** SLSQP refines the initial commands using the same MPC cost function described in Section 5. The optimization is limited to **three iterations**, compared with five iterations in the original MPC400.

6. **Apply the optimized command:** Only the first optimized lift and jib commands are applied to the virtual crane for 100 ms.

7. **Repeat the process:** After 100 ms, the crane states are updated and the controller repeats the procedure using a new 400 ms reference preview.

The neural network provides an initial control estimate, while MPC performs the final optimization. Therefore, the proposed approach retains online optimization but reduces the number of solver iterations.

### 6.3 Final Controller Results

The controllers were evaluated using **nine development-validation recordings**, covering their complete usable durations.

Trajectory-tracking performance was measured using the mean Cartesian tip-position RMSE. Computation time was measured separately using 136 identical controller input conditions on a warmed, single-thread CPU benchmark.

| Controller | Mean Tip RMSE | Mean Computation Time | 99th Percentile Time |
|---|---:|---:|---:|
| Original MPC400 (5 iterations) | 54.75 mm | 43.05 ms | 79.59 ms |
| Conventional MPC (3 iterations) | 131.20 mm | 27.70 ms | 49.87 ms |
| **Neural-Assisted MPC (3 iterations)** | **43.31 mm** | **27.59 ms** | **51.10 ms** |

Compared with the original MPC400 controller, the proposed neural-assisted MPC achieved:

- **20.9% reduction in mean tip-position RMSE**, improving trajectory-tracking accuracy.
- **35.9% reduction in mean controller computation time**, reducing online computational demand.
- **Improved tracking on six of nine recordings.**
- **Zero observed deadline misses across 6,644 closed-loop control updates.**
- **A worst-recording RMSE of 96.58 mm**, compared with 144.60 mm for the original MPC400.

The neural-assisted controller achieved a mean computation time of **27.59 ms**, well within its 100 ms control update interval. This demonstrates real-time-capable computational performance under the evaluated simulation conditions.

Although the hybrid improved average tracking performance, the mean final tip-position error increased from 1.72 mm for MPC400 to 5.81 mm for neural-assisted MPC. Therefore, the improvement is not uniform across every performance metric.

### 6.4 Main Achievement

The proposed neural-assisted MPC demonstrated that combining a fast neural control estimate with limited numerical optimization can improve the accuracy–computation trade-off of predictive crane control.

The neural network provides the initial joystick commands, while MPC evaluates their predicted effects and refines the commands using the crane dynamics model.

Unlike simply reducing conventional MPC from five to three SLSQP iterations, which increased mean tracking RMSE to 131.20 mm, the neural-assisted three-iteration MPC achieved **43.31 mm RMSE**.

The results demonstrate that neural-based initialization can improve closed-loop tracking under a reduced optimization budget, while retaining MPC's predictive correction mechanism.

The computation reduction is primarily associated with the smaller iteration budget, rather than a demonstrated reduction in optimizer iterations due to the neural initialization alone.

These results were obtained using a shared learned virtual crane model. Independent physics-based simulation or hardware validation is required before extending the conclusions to physical crane operation.

---

## 7. Figures


## 7. Figures

The cleaned package should put the generated figures at the locations below. Each figure must be produced from real result arrays; do not copy a figure without its source data or invent missing data.

### Four-model mechanical and hydraulic predictions

![Mechanical-state predictions](Results/Model_Comparison/figures/mechanical_prediction.png)

![Hydraulic-state predictions](Results/Model_Comparison/figures/hydraulic_prediction.png)

### Original MPC and horizon results

![MPC horizon trajectories](MPC/results/comparison/figures/mpc_horizons_trajectory.png)

![MPC horizon tip errors](MPC/results/comparison/figures/mpc_horizons_error.png)

### Final MPC versus neural-assisted MPC

![Original MPC versus neural-assisted MPC trajectories](Neural_MPC/results/comparison/figures/mpc_vs_hybrid_trajectory.png)

![MPC and hybrid error over time](Neural_MPC/results/comparison/figures/mpc_vs_hybrid_error.png)

![MPC and hybrid lift/jib commands](Neural_MPC/results/comparison/figures/mpc_vs_hybrid_commands.png)

![Accuracy-computation trade-off](Neural_MPC/results/comparison/figures/computation_tradeoff.png)

## 8. Installation and execution (after reorganization is verified)

The supplied original package reports a verified **64-bit Python 3.12.10 on Windows 11** environment, with pinned CPU versions in its original `requirements.txt`. The simplified package places dependencies in `MPC/requirements.txt`. Example PowerShell workflow:

```powershell
cd PATU_Crane_Research
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r MPC/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Point to your local raw dataset directory containing Train/
& ".\.venv\Scripts\python.exe" Results/Four_Models/run_models.py --data-root "D:\PATU_Data"
& ".\.venv\Scripts\python.exe" Results/Model_Comparison/compare_models.py
& ".\.venv\Scripts\python.exe" MPC/run_mpc.py --data-root "D:\PATU_Data"
& ".\.venv\Scripts\python.exe" Neural_MPC/run_neural_mpc.py --data-root "D:\PATU_Data"
```

**Do not run these commands until Codex has implemented and verified the reorganized entry points.** In the original ZIP, the already implemented equivalent scripts were `run_models.py`, `run_mpc.py`, `run_hybrid.py`, and `run_all.py` at the old root. Numerical thread variables must be set *before* importing numerical libraries for the regression-sensitive SLSQP setup. Four-model validation data and full controller recordings are not included as raw CSVs in the ZIP.

## 9. Experimental interpretation and limitations

- **Prediction versus control:** GRU/LSTM/Transformer/original Compact predict the **22 physical states**. The wide neural policy predicts **two joystick commands**. They are different networks with different targets.
- **Selected Compact is separate:** The MPC's 9,916-parameter, 10-ms Compact differs from the 4,246-parameter, 10-µs original comparison Compact.
- **Shared-model simulation:** Both controller predictor and virtual plant use the same learned dynamics model; their histories are separate but independent physical-model mismatch is not tested.
- **Dataset reuse:** The nine controller development-validation recordings were repeatedly inspected. Historically examined Test recordings are not untouched external validation.
- **No certified physical deployment:** Real hydraulic limits, noisy sensing, delays, payload variation and independently validated physics have not been demonstrated sufficiently for deployment claims.
- **Timing is empirical:** Thread count, JIT warm-up, hardware and load affect SLSQP wall time. Recorded deadline performance is not a hard guarantee.
- **Standalone neural imitation was unsuccessful:** Its large mean closed-loop error must be reported alongside its much lower execution time.
- **Hybrid is not optimization-free:** The result is a measured neural-initialized *reduced-iteration MPC*, not a pure neural control replacement.

**Reproducibility rule:** preserve checkpoint hashes, signal ordering, scaler parameters, reference construction, kinematics, solver cost addition order, warm-start/clipping conventions, and matched recording durations. Report rather than conceal numerical mismatches after packaging.
