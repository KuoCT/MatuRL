import argparse
import config
import shutil
from time import perf_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MatuRL with a job configuration file.")
    parser.add_argument("--job", required=True, help="Path to the job JSON configuration file.")
    return parser.parse_args()


def main() -> None:
    total_start = perf_counter()

    setup_start = perf_counter()
    args = parse_args()
    config.configure(args.job)

    # Each job owns one immutable output directory.
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config.JOB_PATH, config.OUTPUT_DIR / "job.json")
    from scFv_optimization._utils import save_config
    save_config(config, config.OUTPUT_DIR)

    print(f"[Timing] Job setup: {perf_counter() - setup_start:.2f} s")

    binder_start = perf_counter()
    binder_action = "reused"
    if not config.BINDER_MODEL_PATH.exists():
        from binder_classification.train import train
        train()
        binder_action = "trained"

    print(
        f"[Timing] Binder classifier ({binder_action}): "
        f"{perf_counter() - binder_start:.2f} s"
    )

    if not config.RUN_RL:
        print(f"[Timing] Total: {perf_counter() - total_start:.2f} s")
        return

    framework_start = perf_counter()
    from scFv_optimization.framework import Framework

    framework = Framework()
    print(
        f"[Timing] Framework initialization: "
        f"{perf_counter() - framework_start:.2f} s"
    )

    training_start = perf_counter()
    framework.train()
    print(f"[Timing] RL training: {perf_counter() - training_start:.2f} s")
    print(f"[Timing] Total: {perf_counter() - total_start:.2f} s")


if __name__ == "__main__":
    main()
