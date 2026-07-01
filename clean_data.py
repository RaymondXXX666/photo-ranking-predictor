import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")

TRAIN_SHOOTS = [f"wedding_shoot_{i:02d}" for i in range(1, 9)]
VALID_SHOOTS = ["wedding_shoot_09"]
TEST_SHOOTS = ["wedding_shoot_10"]


def load_all_detections(data_dir: Path) -> pd.DataFrame:
    rows = []

    for shoot_dir in sorted(data_dir.glob("wedding_shoot_*")):
        json_path = shoot_dir / "face_detections.json"

        if not json_path.exists():
            print(f"Skipping {shoot_dir}: face_detections.json not found")
            continue

        shoot_id = shoot_dir.name

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        detections = data.get("detections", [])

        for item in detections:
            row = {
                "shoot_id": shoot_id,
                "image_id": item.get("image_id"),
                "scene_id": item.get("scene_id"),
                "rank": item.get("rank"),
                "num_faces": item.get("num_faces"),
                "category": item.get("category"),
                "is_group_pose": item.get("is_group_pose"),
                "image_width": item.get("image_width"),
                "image_height": item.get("image_height"),
                "faces": item.get("faces", []),
                "group_metrics": item.get("group_metrics", {}),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def split_dataset(df: pd.DataFrame):
    # 1. 排除 rank=null 的样本
    ranked_df = df[df["rank"].notna()].copy()

    # 2. rank 转成 int，后面训练更方便
    ranked_df["rank"] = ranked_df["rank"].astype(int)

    # 3. 计算每个 scene 里有多少张 ranked image
    ranked_df["scene_size"] = ranked_df.groupby(
        ["shoot_id", "scene_id"]
    )["image_id"].transform("count")

    # 4. 排除只有一张图的 scene
    ranked_df = ranked_df[ranked_df["scene_size"] >= 2].copy()

    train_df = ranked_df[ranked_df["shoot_id"].isin(TRAIN_SHOOTS)].copy()
    valid_df = ranked_df[ranked_df["shoot_id"].isin(VALID_SHOOTS)].copy()
    test_df = ranked_df[ranked_df["shoot_id"].isin(TEST_SHOOTS)].copy()

    return train_df, valid_df, test_df


def main():
    df = load_all_detections(DATA_DIR)

    print("All images:", len(df))
    print("Images with rank:", df["rank"].notna().sum())
    print("Images with rank=null:", df["rank"].isna().sum())
    print()

    ranked_df = df[df["rank"].notna()].copy()
    ranked_df["scene_size"] = ranked_df.groupby(
        ["shoot_id", "scene_id"]
    )["image_id"].transform("count")

    print("Ranked images before removing single-image scenes:", len(ranked_df))
    print("Single-image scene samples:", (ranked_df["scene_size"] < 2).sum())
    print("Ranked images after removing single-image scenes:", (ranked_df["scene_size"] >= 2).sum())
    print()

    train_df, valid_df, test_df = split_dataset(df)

    print("Train shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)
    print("Test shape:", test_df.shape)
    print()

    print("Train shoots:", sorted(train_df["shoot_id"].unique()))
    print("Valid shoots:", sorted(valid_df["shoot_id"].unique()))
    print("Test shoots:", sorted(test_df["shoot_id"].unique()))
    print()

    print("Train rank null count:", train_df["rank"].isna().sum())
    print("Valid rank null count:", valid_df["rank"].isna().sum())
    print("Test rank null count:", test_df["rank"].isna().sum())
    print()

    print("Train min scene size:", train_df["scene_size"].min())
    print("Valid min scene size:", valid_df["scene_size"].min())
    print("Test min scene size:", test_df["scene_size"].min())

    output_dir = Path("processed")
    output_dir.mkdir(exist_ok=True)

    train_df.to_pickle(output_dir / "train_raw.pkl")
    valid_df.to_pickle(output_dir / "valid_raw.pkl")
    test_df.to_pickle(output_dir / "test_raw.pkl")

    print()
    print("Saved:")
    print(output_dir / "train_raw.pkl")
    print(output_dir / "valid_raw.pkl")
    print(output_dir / "test_raw.pkl")


if __name__ == "__main__":
    main()