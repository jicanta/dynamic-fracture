from __future__ import annotations

from pathlib import Path
from os import PathLike
from typing import Optional, Tuple, Sequence, Union


from matplotlib.ticker import MaxNLocator

def plot_loss_curves_to_metrics(
    log_paths: Union[PathLike, Sequence[PathLike]],
    *,
    model_root: PathLike = "/scratch/gilbreth/kmartinu/MICROSTRUCTURE_ML/MODEL",
    metrics_root: PathLike = "/scratch/gilbreth/kmartinu/MICROSTRUCTURE_ML/ALL_OUTPUTS/METRICS",
    figure_name: str = "train_val_loss",
    title: str = "Training vs Validation Loss",
    
    title_fontsize: int = 18,
    label_fontsize: int = 16,
    tick_fontsize: int = 14,
    legend_fontsize: int = 14,
    
    xlim: Optional[Tuple[float, float]] = None,
    xtick_step: int = 1,
    n_xticks: Optional[int] = None,     # set ~how many integer ticks to show
    max_epoch: Optional[float] = None,  # plot only up to this epoch
    yscale: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
    show: bool = False,
    append_csvs: bool = False,
    epoch_mode: str = "auto",  # "auto" | "offset" | "raw"
) -> Path:
    # Normalize to list of Paths
    if isinstance(log_paths, (str, Path)):
        log_paths_list = [Path(log_paths)]
    else:
        log_paths_list = [Path(p) for p in log_paths]

    for p in log_paths_list:
        if not p.exists():
            raise FileNotFoundError(f"CSV log not found: {p}")

    model_root = Path(model_root)
    metrics_root = Path(metrics_root)

    # Use first CSV to determine run_name
    run_name = _infer_run_name_from_model(log_paths_list[0], model_root)

    out_dir = metrics_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{figure_name}.png"

    # Load data
    if append_csvs and len(log_paths_list) > 1:
        df = _load_and_append_logs(log_paths_list, epoch_mode=epoch_mode)
    else:
        df = pd.read_csv(log_paths_list[0])
        required = {"epoch", "loss", "val_loss"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. Found: {list(df.columns)}"
            )
        df = df.copy()
        df["epoch"] = pd.to_numeric(df["epoch"], errors="raise")

    # Truncate to max_epoch if requested
    if max_epoch is not None:
        df = df[df["epoch"] <= max_epoch].copy()

    if df.empty:
        raise ValueError("No rows to plot after applying max_epoch/filtering.")

    # Sort by epoch (nice for appended logs) + drop duplicate epochs if any
    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")

    epochs = df["epoch"].to_numpy()
    loss = df["loss"].to_numpy()
    val_loss = df["val_loss"].to_numpy()

    # --- plot ---
    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.plot(epochs, loss, label="train loss")
    ax.plot(epochs, val_loss, label="val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    
    ax.set_xlabel("Epoch", fontsize=label_fontsize)
    ax.set_ylabel("Loss", fontsize=label_fontsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    ax.legend(fontsize=legend_fontsize)

    e_min = float(np.min(epochs))
    e_max = float(np.max(epochs))

    # x-limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif max_epoch is not None:
        ax.set_xlim(e_min, float(max_epoch))

    # ---- integer x-ticks ----
    if n_xticks is not None:
        if n_xticks < 2:
            raise ValueError("n_xticks must be >= 2")
        # MaxNLocator picks "nice" integer ticks, ~n_xticks total
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=max(n_xticks - 1, 1), integer=True)
        )
    else:
        # exact integer ticks every xtick_step
        if xtick_step and xtick_step > 0:
            i_min = int(np.floor(e_min))
            i_max = int(np.ceil(e_max))
            ax.set_xticks(np.arange(i_min, i_max + 1, xtick_step))
            # ensure integer labels (avoids 0.0 style)
            ax.set_xticklabels([str(x) for x in range(i_min, i_max + 1, xtick_step)])

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if yscale is not None:
        ax.set_yscale(yscale)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return out_path
