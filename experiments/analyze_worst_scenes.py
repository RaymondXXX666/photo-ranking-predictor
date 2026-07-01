from pathlib import Path
import pandas as pd

DATA_DIR = Path("processed")


def main():
    metrics_path = DATA_DIR / "test_scene_metrics_pairwise_gradient_boosting_classifier.csv"
    preds_path = DATA_DIR / "test_predictions_pairwise_gradient_boosting_classifier.csv"
    features_path = DATA_DIR / "test_features.csv"

    metrics = pd.read_csv(metrics_path)
    preds = pd.read_csv(preds_path)
    features = pd.read_csv(features_path)

    worst = metrics.sort_values("pairwise_acc").head(20)
    best = metrics.sort_values("pairwise_acc", ascending=False).head(20)

    print("Worst 20 scenes by pairwise accuracy:")
    print(worst.to_string(index=False))

    print("\nBest 20 scenes by pairwise accuracy:")
    print(best.to_string(index=False))

    selected_features = [
        "shoot_id", "scene_id", "image_id", "rank",
        "num_faces", "num_selected_faces",
        "focus_mean", "focus_min", "focus_max",
        "main_focus_mean", "main_focus_min",
        "eye_open_mean", "eye_open_max",
        "eye_midblink_mean", "eye_midblink_max",
        "eye_closed_mean", "eye_closed_max",
        "yaw_abs_mean", "pitch_abs_mean",
        "face_center_x_mean", "face_center_y_mean",
        "total_face_area_ratio",
    ]

    available = [c for c in selected_features if c in features.columns]

    merged = preds.merge(
        features[available],
        on=["shoot_id", "scene_id", "image_id", "rank"],
        how="left",
    )

    worst_keys = set(zip(worst["shoot_id"], worst["scene_id"]))

    worst_detail = merged[
        merged.apply(lambda r: (r["shoot_id"], r["scene_id"]) in worst_keys, axis=1)
    ].sort_values(["scene_id", "rank"])

    output_path = DATA_DIR / "worst_scene_details.csv"
    worst_detail.to_csv(output_path, index=False)

    print(f"\nSaved worst scene details to {output_path}")


if __name__ == "__main__":
    main()