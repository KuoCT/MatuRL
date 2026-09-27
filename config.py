import json
from pathlib import Path
from typing import Any


JOB_PATH: Path | None = None
JOB_NAME: str | None = None
OUTPUT_DIR: Path | None = None
JOB: dict[str, Any] = {}


def load_job(job_path: str | Path) -> dict[str, Any]:
    """Load job configuration from a JSON file."""

    job_path = Path(job_path)

    if not job_path.exists():
        raise FileNotFoundError(
            f"Job configuration file not found: {job_path}"
        )

    if job_path.suffix.lower() != ".json":
        raise ValueError(
            f"Job configuration must be a JSON file: {job_path}"
        )

    with job_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def configure(job_path: str | Path) -> None:
    """Load a job file and expose its settings as config variables."""

    global JOB_PATH, JOB
    global JOB_NAME
    global OUTPUT_DIR

    global PPO_BETA
    global PPO_GAMMA
    global PPO_EPSILON
    global PPO_LAMBDA
    global PPO_TIME_HORIZON
    global PPO_N_EPISODES
    global PPO_BUFFER_SIZE
    global PPO_BATCH_SIZE
    global PPO_N_EPOCHS

    global AGENTS_LR
    global AGENTS_GAMMA
    global AGENTS_STEP_SIZE
    global AGENTS_HIDDEN_DIM
    global AGENTS_DROPOUT_RATE

    global N_PARALLELS
    global CHECKPOINT_INTERVAL
    global RANDOM_SEED

    global TARGET_V_LC
    global TARGET_LINK
    global TARGET_V_RC

    global DATASET_FILENAME

    global BINDER_MODEL_NAME
    global BINDER_MODEL_PATH
    global RUN_RL

    JOB_PATH = Path(job_path)
    JOB = load_job(JOB_PATH)

    JOB_NAME = JOB_PATH.stem
    OUTPUT_DIR = Path("output") / JOB_NAME

    # PPO settings
    PPO_BETA = JOB["PPO_BETA"]
    PPO_GAMMA = JOB["PPO_GAMMA"]
    PPO_EPSILON = JOB["PPO_EPSILON"]
    PPO_LAMBDA = JOB["PPO_LAMBDA"]

    PPO_TIME_HORIZON = JOB["PPO_TIME_HORIZON"]
    PPO_N_EPISODES = JOB["PPO_N_EPISODES"]

    PPO_BUFFER_SIZE = 2048 * PPO_TIME_HORIZON
    PPO_BATCH_SIZE = JOB["PPO_BATCH_SIZE"]
    PPO_N_EPOCHS = JOB["PPO_N_EPOCHS"]

    # Agent settings
    AGENTS_LR = JOB["AGENTS_LR"]
    AGENTS_GAMMA = JOB["AGENTS_GAMMA"]
    AGENTS_STEP_SIZE = JOB["AGENTS_STEP_SIZE"]
    AGENTS_HIDDEN_DIM = JOB["AGENTS_HIDDEN_DIM"]
    AGENTS_DROPOUT_RATE = JOB["AGENTS_DROPOUT_RATE"]

    # Runtime settings
    N_PARALLELS = JOB["N_PARALLELS"]
    CHECKPOINT_INTERVAL = JOB["CHECKPOINT_INTERVAL"]
    RANDOM_SEED = JOB["RANDOM_SEED"]

    # Target scFv
    TARGET_V_LC = JOB["TARGET_V_LC"]
    TARGET_LINK = JOB["TARGET_LINK"]
    TARGET_V_RC = JOB["TARGET_V_RC"]

    # Binder classifier dataset
    DATASET_FILENAME = JOB["DATASET_FILENAME"]

    # Binder classifier model
    BINDER_MODEL_NAME = JOB["BINDER_MODEL_NAME"]
    BINDER_MODEL_PATH = Path("models") / f"{BINDER_MODEL_NAME}.pt"

    # Reinforcement learning
    RUN_RL = JOB.get("RUN_RL", True)