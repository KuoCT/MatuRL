from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as T

from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve

if TYPE_CHECKING:
    from binder_classification.classifier import BinderClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed MatuRL runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to completed run output directories.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="CSV dataset paths used for ROC evaluation.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["opti", "roc", "roc-ds"],
        default=["opti", "roc"],
        help="Plot types to generate (default: opti roc).",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the evaluation output directory.",
    )
    return parser.parse_args()


def apply_plot_style(
    ax: plt.Axes,
    tick_fontsize: int = 6,
    tick_fontweight: str = "normal",
) -> None:
    """Apply a clean GraphPad Prism-like style to the plot."""

    # Keep only the left and bottom axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Draw ticks outward
    ax.tick_params(axis="both", direction="out", width=1.2, length=5, labelsize=tick_fontsize)

    # Apply tick label font weight
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight(tick_fontweight)


def set_legend(
    ax: plt.Axes,
    loc: str = "best",
    bbox_to_anchor: tuple[float, float] | None = None,
    ncol: int = 1,
    fontsize: int = 8,
    fontweight: str = "normal",
) -> None:
    """Configure the legend using native matplotlib positioning options."""

    ax.legend(
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        ncol=ncol,
        prop={"size": fontsize, "weight": fontweight},
    )


def add_title(
    ax: plt.Axes,
    title: str = "",
    subtitle: str = "",
    title_fontsize: int = 10,
    title_fontweight: str = "bold",
    subtitle_fontsize: int = 8,
    subtitle_fontweight: str = "normal",
) -> None:
    """Add left-aligned title and subtitle text."""

    if title: ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=18, loc="left")
    if subtitle: ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=subtitle_fontsize, fontweight=subtitle_fontweight, ha="left", va="bottom")


def predict(
    model: BinderClassifier,
    dataset: str | Path,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the classifier and return true labels and binding probabilities."""

    from binder_classification.dataset import get_batches

    y_true_batches: list[np.ndarray] = []
    y_prob_batches: list[np.ndarray] = []

    # Use evaluation mode and disable gradient calculation
    model.eval()
    with T.no_grad():
        for v_Lc_embs, link_embs, v_Hc_embs, labels in get_batches(dataset, batch_size):
            logits: T.Tensor = model(v_Lc_embs, link_embs, v_Hc_embs)
            probs = T.sigmoid(logits.squeeze(1))

            y_true_batches.append(labels.cpu().numpy())
            y_prob_batches.append(probs.cpu().numpy())

    # Combine all mini-batches
    y_trues = np.concatenate(y_true_batches)
    y_probs = np.concatenate(y_prob_batches)

    return y_trues, y_probs


def evaluate_dataset(
    dataset: str | Path | list[str | Path],
    dataset_label: str | list[str],
    model_path: str | Path,
    output_dir: str | Path = "./out",
    title: str = "",
    subtitle: str = "",
    legend_loc: str = "best",
    legend_bbox: tuple[float, float] | None = None,
    legend_ncol: int = 1,
    colors: list[str] | None = None,
    batch_size: int = 128,
    figsize: tuple[float, float] = (4, 4),
    title_fontsize: int = 10,
    title_fontweight: str = "bold",
    subtitle_fontsize: int = 8,
    subtitle_fontweight: str = "normal",
    axis_title_fontsize: int = 8,
    axis_title_fontweight: str = "bold",
    tick_fontsize: int = 6,
    tick_fontweight: str = "normal",
    legend_fontsize: int = 8,
    legend_fontweight: str = "normal",
) -> None:
    """Evaluate one or more datasets and draw their ROC curves on one axis."""

    from binder_classification.classifier import BinderClassifier

    # Convert single inputs to lists
    if isinstance(dataset, (str, Path)): dataset = [dataset]
    if isinstance(dataset_label, str): dataset_label = [dataset_label]

    # Check dataset settings
    if len(dataset) != len(dataset_label):
        raise ValueError("dataset and dataset_label must have the same length")

    if colors is None:
        colors = [None] * len(dataset)
    elif len(colors) != len(dataset):
        raise ValueError("colors must have the same length as dataset")

    # Prepare paths
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained classifier
    model = BinderClassifier(str(model_path))
    model.load_model()

    # Create the shared ROC axis
    fig, ax = plt.subplots(figsize=figsize)

    # Evaluate each dataset and add its ROC curve
    for dataset_path, label, color in zip(dataset, dataset_label, colors):
        y_trues, y_probs = predict(
            model=model,
            dataset=dataset_path,
            batch_size=batch_size,
        )

        auc = roc_auc_score(y_trues, y_probs)
        fpr, tpr, _ = roc_curve(y_trues, y_probs)

        # Save sample-level predictions
        prediction_df = pd.DataFrame({
            "label": y_trues.astype(np.int32),
            "probability": y_probs,
        })

        file_label = label.replace(" ", "_")
        prediction_df.to_csv(output_dir / f"{file_label}_predictions.csv", index=False)

        # Add ROC curve
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{label} (AUC = {auc:.3f})",
        )

        print(f"{label}: n = {len(y_trues)}, ROC-AUC = {auc:.4f}")

    # Add random-classifier reference line
    ax.plot(
        [0, 1],
        [0, 1],
        color="0.6",
        linewidth=1,
        linestyle="--",
        label="_nolegend_",
    )

    # Configure ROC axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("False Positive Rate", fontsize=axis_title_fontsize, fontweight=axis_title_fontweight, labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=axis_title_fontsize, fontweight=axis_title_fontweight, labelpad=8)

    # Apply title and plot formatting
    add_title(
        ax,
        title=title,
        subtitle=subtitle,
        title_fontsize=title_fontsize,
        title_fontweight=title_fontweight,
        subtitle_fontsize=subtitle_fontsize,
        subtitle_fontweight=subtitle_fontweight,
    )

    apply_plot_style(
        ax,
        tick_fontsize=tick_fontsize,
        tick_fontweight=tick_fontweight,
    )

    set_legend(
        ax,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        fontweight=legend_fontweight,
    )

    # Save ROC figure
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def compare_models(
    dataset: str | Path,
    model_paths: list[str | Path],
    model_labels: list[str],
    output_dir: str | Path,
    colors: list[str] | None = None,
    batch_size: int = 128,
    figsize: tuple[float, float] = (4, 4),
) -> None:
    """Evaluate multiple binder classifiers on one common dataset."""

    from binder_classification.classifier import BinderClassifier

    if len(model_paths) != len(model_labels):
        raise ValueError("model_paths and model_labels must have the same length")

    if colors is None:
        colors = [None] * len(model_paths)
    elif len(colors) != len(model_paths):
        raise ValueError("colors must have the same length as model_paths")

    output_dir = Path(output_dir)
    metrics = []
    fig, ax = plt.subplots(figsize=figsize)

    for model_path, label, color in zip(model_paths, model_labels, colors):
        model = BinderClassifier(str(model_path))
        model.load_model()

        y_trues, y_probs = predict(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
        )

        auc = roc_auc_score(y_trues, y_probs)
        fpr, tpr, _ = roc_curve(y_trues, y_probs)

        safe_label = label.replace(" ", "_")
        pd.DataFrame({
            "label": y_trues.astype(np.int32),
            "probability": y_probs,
        }).to_csv(
            output_dir / f"{safe_label}_predictions.csv",
            index=False,
        )

        metrics.append({
            "run": label,
            "model_path": str(model_path),
            "n": len(y_trues),
            "roc_auc": auc,
        })

        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{label} (AUC = {auc:.3f})",
        )

        print(f"{label}: n = {len(y_trues)}, ROC-AUC = {auc:.4f}")

    pd.DataFrame(metrics).to_csv(
        output_dir / "roc_metrics.csv",
        index=False,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color="0.6",
        linewidth=1,
        linestyle="--",
        label="_nolegend_",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("False Positive Rate", fontsize=8, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=8, fontweight="bold", labelpad=8)

    add_title(
        ax,
        title="Binder Classifier Comparison",
        subtitle="ROC Analysis",
    )
    apply_plot_style(ax)
    set_legend(ax, loc="lower right")

    fig.tight_layout()
    fig.savefig(
        output_dir / "roc_curve.svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def plot_bind_prob_curve(
    dataset: str | Path | list[str | Path],
    dataset_label: str | list[str],
    output_path: str | Path,
    sigma: int = 200,
    colors: list[str] | None = None,
    linewidth: float = 1.5,
    title: str = "Binding Probability Across Episodes",
    subtitle: str = "",
    legend_loc: str = "best",
    legend_bbox: tuple[float, float] | None = None,
    legend_ncol: int = 1,
    figsize: tuple[float, float] = (4, 4),
    title_fontsize: int = 10,
    title_fontweight: str = "bold",
    subtitle_fontsize: int = 8,
    subtitle_fontweight: str = "normal",
    axis_title_fontsize: int = 8,
    axis_title_fontweight: str = "bold",
    tick_fontsize: int = 6,
    tick_fontweight: str = "normal",
    legend_fontsize: int = 8,
    legend_fontweight: str = "normal",
) -> None:
    """Replot one or more smoothed binding probability curves from exp_results.csv."""

    # Convert single inputs to lists
    if isinstance(dataset, (str, Path)): dataset = [dataset]
    if isinstance(dataset_label, str): dataset_label = [dataset_label]

    # Check dataset settings
    if len(dataset) != len(dataset_label):
        raise ValueError("dataset and dataset_label must have the same length")

    if colors is None:
        colors = [None] * len(dataset)
    elif len(colors) != len(dataset):
        raise ValueError("colors must have the same length as dataset")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the shared binding probability axis
    fig, ax = plt.subplots(figsize=figsize)

    # Read each experiment and add its smoothed curve
    for dataset_path, label, color in zip(dataset, dataset_label, colors):
        df = pd.read_csv(dataset_path)

        if "Bind-Prob_T" not in df.columns:
            raise ValueError(f"Cannot find column 'Bind-Prob_T' in: {dataset_path}")

        # Apply the same Gaussian smoothing used by the original framework
        y_data = df["Bind-Prob_T"].astype(float).to_numpy()
        x_data = np.arange(1, len(y_data) + 1)
        y_data_smooth = gaussian_filter1d(y_data, sigma=sigma)

        # Add smoothed binding probability curve
        ax.plot(
            x_data,
            y_data_smooth,
            color=color,
            linewidth=linewidth,
            label=label,
        )

    # Configure axes
    ax.set_xlabel("Episodes", fontsize=axis_title_fontsize, fontweight=axis_title_fontweight, labelpad=8)
    ax.set_ylabel("Binding Probability", fontsize=axis_title_fontsize, fontweight=axis_title_fontweight, labelpad=8)

    # Apply title and plot formatting
    add_title(
        ax,
        title=title,
        subtitle=subtitle,
        title_fontsize=title_fontsize,
        title_fontweight=title_fontweight,
        subtitle_fontsize=subtitle_fontsize,
        subtitle_fontweight=subtitle_fontweight,
    )

    apply_plot_style(
        ax,
        tick_fontsize=tick_fontsize,
        tick_fontweight=tick_fontweight,
    )

    set_legend(
        ax,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        fontweight=legend_fontweight,
    )

    # Save styled figure
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    run_dirs = [Path(path) for path in args.runs]
    plot_types = set(args.plots)

    if "roc" in plot_types and "roc-ds" in plot_types:
        raise ValueError("roc and roc-ds cannot be generated together")

    dataset_paths = []
    if "roc" in plot_types or "roc-ds" in plot_types:
        if args.dataset is None:
            raise ValueError("--dataset is required when generating ROC plots")

        dataset_paths = [Path(path) for path in args.dataset]
        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Evaluation dataset not found: {dataset_path}"
                )

    if "roc" in plot_types and len(dataset_paths) != 1:
        raise ValueError("roc requires exactly one common dataset")

    if "roc-ds" in plot_types and len(run_dirs) != 1:
        raise ValueError("roc-ds requires exactly one run directory")

    run_labels = []
    model_paths = []
    result_paths = []

    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        result_path = run_dir / "exp_results.csv"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Run config not found: {config_path}"
            )

        if "opti" in plot_types and not result_path.exists():
            raise FileNotFoundError(
                f"Optimization result not found: {result_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            run_config = json.load(file)

        run_labels.append(run_config.get("JOB_NAME", run_dir.name))

        if "opti" in plot_types:
            result_paths.append(result_path)

        if "roc" in plot_types or "roc-ds" in plot_types:
            try:
                model_path = Path(run_config["BINDER_MODEL_PATH"])
            except KeyError as error:
                raise ValueError(
                    f"BINDER_MODEL_PATH is missing from: {config_path}"
                ) from error

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Binder model not found: {model_path}"
                )

            model_paths.append(model_path)

    evaluation_dir = Path("evaluate") / args.name
    evaluation_dir.mkdir(parents=True, exist_ok=False)

    if "opti" in plot_types:
        plot_bind_prob_curve(
            dataset=result_paths,
            dataset_label=run_labels,
            output_path=evaluation_dir / "bind_prob_curve.svg",
            sigma=200,
            title="Binding Probability Across Episodes",
            subtitle="scFv Optimization Comparison",
            legend_loc="lower right",
        )

    if "roc" in plot_types:
        compare_models(
            dataset=dataset_paths[0],
            model_paths=model_paths,
            model_labels=run_labels,
            output_dir=evaluation_dir,
        )

    if "roc-ds" in plot_types:
        evaluate_dataset(
            dataset=dataset_paths,
            dataset_label=[path.stem for path in dataset_paths],
            model_path=model_paths[0],
            output_dir=evaluation_dir,
            title=run_labels[0],
            subtitle="ROC Analysis Across Datasets",
            legend_loc="lower right",
        )


if __name__ == "__main__":
    main()
