# MatuRL

This is the official repository for **MatuRL**, a DRL framework for CAR-T maturation that simulates B cell–like 
antibody evolution *in silico*, as described in *"AI-guided scFv engineering enhances T cell-specific cytotoxicity toward cancer cells"*.

## Prerequisites

Install the required packages with the following command.

```bash
pip install torch antiberty scikit-learn matplotlib pandas
```

## Usage

Follow the steps below to optimize a specified scFv.

### Step 1. Dataset Preparation

To train the binder classifier, prepare a CSV file containing four columns: `v_Lc`, `link`, `v_Rc`, and `label`. Place the file in the `dataset` directory.

- The values for `v_Lc`, `link`, and `v_Rc` must be uppercase protein sequences, using only the one-letter codes of the 20 standard amino acids.

- The `label` field accepts only integer values `0` or `1`, where `0` indicates the scFv (`v_Lc` + `link` + `v_Rc`) is not a binder, and `1` indicates it is a binder.

### Step 2. Parameter Configuration

In `config.py`, four parameters must be configured manually: `TARGET_V_LC`, `TARGET_LINK`, `TARGET_V_RC`, and `DATASET_FILENAME`.

- `TARGET_V_LC`, `TARGET_LINK`, and `TARGET_V_RC` specify the scFv to be optimized.

- `DATASET_FILENAME` refers to the dataset folder name mentioned in the previous step (omit the `.csv`).

The remaining parameters are related to DRL settings. They have default values and can be left unchanged.

### Step 3. Framework Execution

Run `main.py` to execute the MatuRL framework.

The program first trains the binder classifier (unless a trained model already exists), and then trains the DRL agents that constitute the scFv optimizer.

The `scFv_optimization/logs` directory stores the results generated during agent training, including the binding probability growth curve for the target scFv (`bind_prob_curve.svg`) and the optimization result sequence for each episode (`exp_results.csv`).
