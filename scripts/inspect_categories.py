from pathlib import Path
import pandas as pd

DATA_DIR = Path("processed")


def inspect_split(name):
    df = pd.read_pickle(DATA_DIR / f"{name}_raw.pkl")

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    print("Total images:", len(df))
    print("\nCategory value counts:")
    print(df["category"].value_counts(dropna=False))

    print("\nRanked no_faces samples:")
    no_faces = df[df["category"] == "no_faces"]
    print(no_faces[["shoot_id", "scene_id", "image_id", "rank", "num_faces", "category"]].head(20).to_string(index=False))


def main():
    for split in ["train", "valid", "test"]:
        inspect_split(split)


if __name__ == "__main__":
    main()