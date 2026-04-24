from typing import Optional, Sequence
DEFAULT_METRICS: Sequence[str] = ("precision", "recall", "f1", "accuracy")

def plot_per_frame_metrics(
    csv_path: str = "per_frame_metrics.csv",
    outdir: str = "metric_plots",
    metrics: Sequence[str] = DEFAULT_METRICS,
    *,
    frame_col: str = "frame_id",
    time_divisor: float = 10.0,
    x_label: Optional[str] = None,
    show: bool = True,
    f1_positive_only: bool = True,
    # confusion matrix plotting
    plot_confusion_counts: bool = True,
    confusion_cols: Sequence[str] = ("tp", "tn", "fp", "fn"),
    confusion_grid_name: str = "confusion_counts_2x2_grid.png",
    confusion_ylabel: str = "Pixel count",
    confusion_logy: bool = False,
) -> dict:
    """
    Reads a per-frame metrics CSV and saves:
      1) Individual scatter plots for each metric
      2) A 2x2 grid scatter plot of the metrics
      3) (optional) TP/TN/FP/FN over time

    Notes:
      - If f1_positive_only=True, ALL plots only use rows where F1 > 0.
        (This is useful to ignore early frames before any fracture appears.)
      - Confusion counts require the CSV to contain columns named in `confusion_cols`
        (default: tp, tn, fp, fn).
    """
    os.makedirs(outdir, exist_ok=True)

    # ---------- load CSV ----------
    df = pd.read_csv(csv_path)

    # Validate columns for metrics
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"CSV missing metric columns: {missing}")
    if frame_col not in df.columns:
        raise ValueError(f"CSV missing '{frame_col}'.")

    # Clean + sort frame column
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
    df = df.dropna(subset=[frame_col]).copy()
    df[frame_col] = df[frame_col].astype(int)
    df = df.sort_values(frame_col).reset_index(drop=True)

    # Keep a copy of the full DF for info stats
    df_full = df.copy()

    # ---------- optional F1>0 filtering for ALL plots ----------
    if f1_positive_only:
        if "f1" not in df.columns:
            raise ValueError("f1_positive_only=True, but CSV has no 'f1' column.")
        df = df[df["f1"] > 0].copy()
        if len(df) == 0:
            print("[plot_metrics] WARNING: f1_positive_only=True but no rows where f1>0. "
                  "Nothing will be plotted.")
    # At this point, df is the dataframe actually being plotted

    if x_label is None:
        x_label = "Time (ns)"

    # X axis from (possibly filtered) df
    x_all = df[frame_col].to_numpy(dtype=float) / float(time_divisor)

    saved_paths = []

    # -----------------------------
    # 1) Individual scatter plots (metrics)
    # -----------------------------
    for m in metrics:
        x_m = x_all
        y_m = df[m].to_numpy()

        plt.figure(figsize=(7, 4))
        plt.scatter(x_m, y_m, s=8)
        plt.xlabel(x_label)
        plt.ylabel(m.capitalize())

        if f1_positive_only:
            title_suffix = " (F1>0 frames only)"
        else:
            title_suffix = ""

        plt.title(f"{m.capitalize()} vs {x_label}{title_suffix}")
        plt.tight_layout()

        fname_suffix = "_f1_gt0" if f1_positive_only else ""
        out_path = os.path.join(outdir, f"{m}_scatter{fname_suffix}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths.append(out_path)

    # -----------------------------
    # 2) 2x2 grid scatter (metrics)
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = np.asarray(axes).ravel()

    for ax, m in zip(axes, metrics):
        x_m = x_all
        y_m = df[m].to_numpy()
        title = f"{m.capitalize()} vs Time"
        if f1_positive_only:
            title += " (F1>0 frames only)"

        ax.scatter(x_m, y_m, s=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(m.capitalize())
        ax.set_title(title)

    fig.tight_layout()
    grid_path = os.path.join(
        outdir,
        "metrics_2x2_grid_scatter_f1_gt0.png" if f1_positive_only else "metrics_2x2_grid_scatter.png",
    )
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    saved_paths.append(grid_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    # -----------------------------
    # 3) Confusion counts over time (TP/TN/FP/FN)
    # -----------------------------
    confusion_plotted = False
    missing_conf = []
    if plot_confusion_counts:
        missing_conf = [c for c in confusion_cols if c not in df.columns]
        if missing_conf:
            print(
                f"[plot_metrics] WARNING: confusion count columns missing: {missing_conf}. "
                f"Skipping TP/TN/FP/FN plots."
            )
        else:
            # Ensure numeric
            for c in confusion_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            # Individual plots
            for c in confusion_cols:
                plt.figure(figsize=(7, 4))
                plt.scatter(x_all, df[c].to_numpy(), s=8)
                plt.xlabel(x_label)
                plt.ylabel(confusion_ylabel)

                title = f"{c.upper()} vs {x_label}"
                if f1_positive_only:
                    title += " (F1>0 frames only)"
                plt.title(title)

                if confusion_logy:
                    plt.yscale("log")
                plt.tight_layout()

                fname_suffix = "_f1_gt0" if f1_positive_only else ""
                out_path = os.path.join(outdir, f"{c}_scatter{fname_suffix}.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                saved_paths.append(out_path)

            # 2x2 grid
            fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
            axes2 = np.asarray(axes2).ravel()

            for ax, c in zip(axes2, confusion_cols):
                ax.scatter(x_all, df[c].to_numpy(), s=8)
                ax.set_xlabel(x_label)
                ax.set_ylabel(confusion_ylabel)
                title = f"{c.upper()} vs {x_label}"
                if f1_positive_only:
                    title += " (F1>0 frames only)"
                ax.set_title(title)
                if confusion_logy:
                    ax.set_yscale("log")

            fig2.tight_layout()
            conf_grid_path = os.path.join(
                outdir,
                confusion_grid_name.replace(".png", "_f1_gt0.png") if f1_positive_only else confusion_grid_name,
            )
            fig2.savefig(conf_grid_path, dpi=300, bbox_inches="tight")
            saved_paths.append(conf_grid_path)

            if show:
                plt.show()
            else:
                plt.close(fig2)

            confusion_plotted = True

    # -----------------------------
    # Info dict (use full DF for totals)
    # -----------------------------
    info = {
        "outdir": outdir,
        "saved_paths": saved_paths,
        "rows_plotted_total": int(len(df)),
        "rows_total_original": int(len(df_full)),
        "x_min_total": float(np.min(x_all)) if len(x_all) else None,
        "x_max_total": float(np.max(x_all)) if len(x_all) else None,
        "x_label": x_label,
        "csv_path": csv_path,
        "f1_positive_only": f1_positive_only,
        "plot_confusion_counts": plot_confusion_counts,
        "confusion_cols": tuple(confusion_cols),
        "confusion_plotted": confusion_plotted,
        "missing_confusion_cols": missing_conf,
    }

    return info
