import matplotlib.pyplot as plt
import pandas as pd
from tbparse import SummaryReader
from pathlib import Path
import argparse

def plot_logs(log_dir: str, metrics: list[str]):
    """Reads TensorBoard event files and plots specified metrics."""
    log_path = Path(log_dir)
    if not log_path.is_dir():
        print(f"Error: Log directory not found: {log_dir}")
        return

    # Find all event files recursively
    event_files = list(log_path.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    print(f"Found {len(event_files)} event files. Reading data...")

    # Use tbparse to read all event files into a single DataFrame
    # Group by run directory to separate different experiments if needed
    reader = SummaryReader(str(log_path), pivot=True)
    df = reader.scalars

    if df.empty:
        print("No scalar data found in the event files.")
        return

    print("Data loaded successfully. Plotting...")

    # Determine the number of unique runs/directories
    run_dirs = df['dir_name'].unique()
    num_runs = len(run_dirs)
    print(f"Found data for {num_runs} run(s): {', '.join(run_dirs)}")

    # Plot each specified metric
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes] # Ensure axes is always iterable

    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in the data. Skipping.")
            ax.set_title(f"Metric '{metric}' (Not Found)")
            ax.text(0.5, 0.5, 'Metric not found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue

        # Plot data for each run
        for run_dir in run_dirs:
            run_df = df[df['dir_name'] == run_dir].dropna(subset=[metric])
            if not run_df.empty:
                ax.plot(run_df['step'], run_df[metric], label=f"{run_dir} - {metric}")

        ax.set_ylabel(metric)
        ax.set_title(f"Metric: {metric}")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Step")
    fig.suptitle("Training Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    # plt.show()
    # You can also save the figure
    plt.savefig("training_metrics.png")
    print("Plot saved to training_metrics.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from TensorBoard logs.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing TensorBoard event files.")
    parser.add_argument("--metrics", nargs='+', default=["loss", "mean_reward", "success_rate/train", "success_rate/eval", "grad_norm"], help="List of metrics to plot.")
    args = parser.parse_args()

    plot_logs(args.log_dir, args.metrics)