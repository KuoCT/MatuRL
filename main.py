import argparse
import config
import shutil
from contextlib import contextmanager
from time import perf_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MatuRL with a job configuration file.")
    parser.add_argument("--job", required=True, help="Path to the job JSON configuration file.")
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    days, remainder = divmod(round(seconds), 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)

    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


@contextmanager
def print_timing(label: str):
    start = perf_counter()
    try:
        yield
    finally:
        print(f"[Timing] {label}: {format_duration(perf_counter() - start)}")


def main() -> None:
    with print_timing("Total"):
        with print_timing("Job setup"):
            config.configure(parse_args().job)
            print(f"\nExperiment: {config.JOB_NAME}\n")

            # Each job owns one immutable output directory.
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
            shutil.copy2(config.JOB_PATH, config.OUTPUT_DIR / "job.json")

            from scFv_optimization._utils import save_config
            save_config(config, config.OUTPUT_DIR)

        model_exists = config.BINDER_MODEL_PATH.exists()
        binder_action = "reused" if model_exists else "trained"
        with print_timing(f"Binder classifier ({binder_action})"):
            if not model_exists:
                from binder_classification.train import train
                train()

        if config.RUN_RL:
            with print_timing("Framework initialization"):
                from scFv_optimization.framework import Framework
                framework = Framework()

            with print_timing("RL training"):
                framework.train()

    print(f"\nCompleted: {config.JOB_NAME}\n")


if __name__ == "__main__":
    main()
