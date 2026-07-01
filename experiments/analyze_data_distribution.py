from pathlib import Path
import pandas as pd

DATA_DIR = Path("processed")


def describe_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}_features.csv")

    scene_stats = (
        df.groupby(["shoot_id", "scene_id"])
        .size()
        .reset_index(name="scene_size")
    )

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    print("Images:", len(df))
    print("Scenes:", len(scene_stats))
    print("\nScene size distribution:")
    print(scene_stats["scene_size"].describe())

    print("\nScene size counts:")
    print(scene_stats["scene_size"].value_counts().sort_index())

    print("\nCategory counts:")
    print(df["category_single_face"].sum(), "single_face")
    print(df["category_two_faces"].sum(), "two_faces")
    print(df["category_small_group"].sum(), "small_group")
    print(df["category_large_group"].sum(), "large_group")
    print(df["category_pair"].sum(), "pair")
    print(df["category_no_faces"].sum(), "no_faces")
    print(df["category_unknown"].sum(), "unknown")

    print("\nRank distribution:")
    print(df["rank"].value_counts().sort_index())


def main():
    for split in ["train", "valid", "test"]:
        describe_split(split)


if __name__ == "__main__":
    main()