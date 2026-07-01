from pathlib import Path
import pandas as pd

DATA_DIR = Path("processed")


def main():
    metrics_path = DATA_DIR / "test_scene_metrics_pairwise_gradient_boosting_classifier.csv"
    metrics = pd.read_csv(metrics_path)

    summary = (
        metrics.groupby("n_images")
        .agg(
            n_scenes=("scene_id", "count"),
            mean_pairwise_acc=("pairwise_acc", "mean"),
            mean_top1_acc=("top1_acc", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_kendall=("kendall", "mean"),
        )
        .reset_index()
        .sort_values("n_images")
    )

    print(summary.to_string(index=False))

    output_path = DATA_DIR / "performance_by_scene_size.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()