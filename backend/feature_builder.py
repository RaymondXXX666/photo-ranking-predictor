from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("processed")
OUTPUT_DIR = Path("processed")
MAX_FACES = 20


EYE_KEYS = [
    "midblink",
    "closed",
    "partiallyopen",
    "open",
    "glasses",
    "covered",
    "barelyopen",
    "sideFaceAppearsClosed",
    "sideFaceAppearsOpen",
]


def safe_mean(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else 0.0


def safe_max(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.max(values)) if values else 0.0


def safe_min(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.min(values)) if values else 0.0


def select_faces(faces, max_faces=MAX_FACES):
    """
    如果一张图超过 20 张脸，只保留最重要的 20 张。
    这里按 main_subject_score、confidence、bbox area 粗略排序。
    """
    if not isinstance(faces, list):
        return []

    def face_priority(face):
        bbox = face.get("bbox", {}) or {}
        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        area = max(0, x2 - x1) * max(0, y2 - y1)

        return (
            face.get("main_subject_score", 0) or 0,
            face.get("confidence", 0) or 0,
            area,
        )

    return sorted(faces, key=face_priority, reverse=True)[:max_faces]


def extract_image_features(row):
    faces = select_faces(row["faces"])
    group_metrics = row["group_metrics"] if isinstance(row["group_metrics"], dict) else {}

    image_width = row.get("image_width") or 1
    image_height = row.get("image_height") or 1
    image_area = image_width * image_height

    features = {
        "shoot_id": row["shoot_id"],
        "scene_id": row["scene_id"],
        "image_id": row["image_id"],
        "rank": row["rank"],

        "num_faces": row.get("num_faces", 0) or 0,
        "num_selected_faces": len(faces),
        "is_group_pose": int(bool(row.get("is_group_pose", False))),
    }

    # category one-hot
    # category one-hot
    category = row.get("category", "unknown")
    num_faces = row.get("num_faces", 0) or 0

    # Fix noisy category labels:
    # Some ranked samples are marked as "no_faces" even though num_faces > 0.
    if category == "no_faces" and num_faces > 0:
        if num_faces == 1:
            category = "single_face"
        elif num_faces == 2:
            category = "two_faces"
        else:
            category = "other"

    KNOWN_CATEGORIES = [
        "single_face",
        "two_faces",
        "casual_group",
        "posed_group_shot",
        "other",
        "no_faces",
        "unknown",
    ]

    if category not in KNOWN_CATEGORIES:
        category = "unknown"

    for cat in KNOWN_CATEGORIES:
        features[f"category_{cat}"] = int(category == cat)

    # group metrics
    features["group_num_primary_faces"] = group_metrics.get("num_primary_faces", 0) or 0
    features["group_faces_looking_at_camera"] = group_metrics.get("faces_looking_at_camera", 0) or 0
    features["group_camera_facing_percentage"] = group_metrics.get("camera_facing_percentage", 0.0) or 0.0
    features["group_has_overlaps"] = int(bool(group_metrics.get("has_overlaps", False)))
    features["group_max_overlap"] = group_metrics.get("max_overlap", 0.0) or 0.0
    features["group_faces_in_rows"] = group_metrics.get("faces_in_rows", 0) or 0
    features["group_row_percentage"] = group_metrics.get("row_percentage", 0.0) or 0.0

    # face-level containers
    focus_scores = []
    user_facing_scores = []
    confidences = []
    main_subject_scores = []
    subject_prob_main = []
    subject_prob_secondary = []
    subject_prob_unimportant = []

    yaw_abs = []
    pitch_abs = []
    roll_abs = []

    face_area_ratios = []
    face_center_x = []
    face_center_y = []
    
    #global eye state flags
    main_focus_scores = []
    main_midblink_probs = []
    main_open_probs = []
    main_closed_probs = []
    main_barelyopen_probs = []
    main_covered_probs = []
    main_yaw_abs = []
    main_pitch_abs = []
    main_face_area_ratios = []
    
    # main eye state flags
    main_bad_eye_flags = []
    nonmain_bad_eye_flags = []

    main_midblink_flags = []
    main_closed_flags = []
    main_barelyopen_flags = []

    nonmain_midblink_flags = []
    nonmain_closed_flags = []
    nonmain_barelyopen_flags = []

    #face ratio
    main_quality_penalties = []
    all_quality_penalties = []

    main_focus_penalties = []
    main_eye_penalties = []
    main_pose_penalties = []

    open_probs = []
    closed_probs = []
    covered_probs = []

    main_open_good_flags = []
    main_bad_quality_flags = []
    

    eye_probs = {key: [] for key in EYE_KEYS}

    for face in faces:
        focus_scores.append(face.get("focus_score", 0.0))
        user_facing_scores.append(face.get("user_facing_score", 0.0))
        confidences.append(face.get("confidence", 0.0))

        main_subject_scores.append(face.get("main_subject_score", 0.0))
        subject_prob_main.append(face.get("subject_prob_main", 0.0))
        subject_prob_secondary.append(face.get("subject_prob_secondary", 0.0))
        subject_prob_unimportant.append(face.get("subject_prob_unimportant", 0.0))

        orientation = face.get("orientation", {}) or {}
        yaw_abs.append(abs(orientation.get("yaw", 0) or 0))
        pitch_abs.append(abs(orientation.get("pitch", 0) or 0))
        roll_abs.append(abs(orientation.get("roll", 0) or 0))

        bbox = face.get("bbox", {}) or {}
        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area_ratio = (w * h) / image_area if image_area > 0 else 0.0

        cx = ((x1 + x2) / 2) / image_width if image_width > 0 else 0.0
        cy = ((y1 + y2) / 2) / image_height if image_height > 0 else 0.0

        face_area_ratios.append(area_ratio)
        face_center_x.append(cx)
        face_center_y.append(cy)

        eye_state = face.get("eye_state", {}) or {}
        probabilities = eye_state.get("probabilities", {}) or {}

        midblink_prob = probabilities.get("midblink", 0.0)
        closed_prob = probabilities.get("closed", 0.0)
        partiallyopen_prob = probabilities.get("partiallyopen", 0.0)
        open_prob = probabilities.get("open", 0.0)
        covered_prob = probabilities.get("covered", 0.0)
        barelyopen_prob = probabilities.get("barelyopen", 0.0)


        open_probs.append(open_prob)
        closed_probs.append(closed_prob)
        covered_probs.append(covered_prob)

        focus_score = face.get("focus_score", 0.0) or 0.0
        subject_weight = face.get("subject_prob_main", 0.0) or 0.0

        yaw = abs(orientation.get("yaw", 0) or 0)
        pitch = abs(orientation.get("pitch", 0) or 0)

        focus_penalty = max(0.0, 70.0 - focus_score) / 70.0

        eye_penalty = max(
            closed_prob,
            covered_prob,
            midblink_prob,
            barelyopen_prob,
            0.5 * partiallyopen_prob,
        )

        pose_penalty = min(1.0, (yaw + pitch) / 90.0)

        quality_penalty = (
            0.45 * focus_penalty
            + 0.40 * eye_penalty
            + 0.15 * pose_penalty
        )

        all_quality_penalties.append(quality_penalty * max(subject_weight, 0.1))

        if face.get("is_main_subject", False):
            main_focus_penalties.append(focus_penalty)
            main_eye_penalties.append(eye_penalty)
            main_pose_penalties.append(pose_penalty)
            main_quality_penalties.append(quality_penalty)
            main_open_good_flags.append(int(open_prob > 0.8))
            main_bad_quality_flags.append(int(quality_penalty > 0.5))

        midblink_prob = probabilities.get("midblink", 0.0)
        closed_prob = probabilities.get("closed", 0.0)
        barelyopen_prob = probabilities.get("barelyopen", 0.0)

        midblink_flag = int(midblink_prob > 0.3)
        closed_flag = int(closed_prob > 0.3)
        barelyopen_flag = int(barelyopen_prob > 0.5)

        bad_eye_flag = int(
         midblink_flag == 1
         or closed_flag == 1
         or barelyopen_flag == 1
         )

        for key in EYE_KEYS:
            eye_probs[key].append(probabilities.get(key, 0.0))

        if face.get("is_main_subject", False):
            main_focus_scores.append(face.get("focus_score", 0.0))
            main_midblink_probs.append(probabilities.get("midblink", 0.0))
            main_open_probs.append(probabilities.get("open", 0.0))
            main_closed_probs.append(probabilities.get("closed", 0.0))
            main_barelyopen_probs.append(probabilities.get("barelyopen", 0.0))
            main_covered_probs.append(probabilities.get("covered", 0.0))
            main_yaw_abs.append(abs(orientation.get("yaw", 0) or 0))
            main_pitch_abs.append(abs(orientation.get("pitch", 0) or 0))
            main_face_area_ratios.append(area_ratio)

            main_bad_eye_flags.append(bad_eye_flag)
            main_midblink_flags.append(midblink_flag)
            main_closed_flags.append(closed_flag)
            main_barelyopen_flags.append(barelyopen_flag)
        else:
            nonmain_bad_eye_flags.append(bad_eye_flag)
            nonmain_midblink_flags.append(midblink_flag)
            nonmain_closed_flags.append(closed_flag)
            nonmain_barelyopen_flags.append(barelyopen_flag)

    
    
    low_focus_threshold = 50.0

    num_faces_selected = len(faces)
    num_main_faces = len(main_bad_eye_flags)
    num_nonmain_faces = len(nonmain_bad_eye_flags)

    # Low-focus features can still be global, because blur affects overall quality.
    features["num_low_focus_faces"] = sum(v < low_focus_threshold for v in focus_scores)
    features["low_focus_face_ratio"] = (
    features["num_low_focus_faces"] / num_faces_selected
    if num_faces_selected > 0 else 0.0
    )

    features["all_faces_good_focus"] = int(
    num_faces_selected > 0 and features["num_low_focus_faces"] == 0
    )

    # Main-subject bad-eye features: stronger quality signal.
    features["num_main_faces"] = num_main_faces
    features["num_main_bad_eye_faces"] = int(np.sum(main_bad_eye_flags)) if main_bad_eye_flags else 0
    features["main_bad_eye_ratio"] = (
    features["num_main_bad_eye_faces"] / num_main_faces
    if num_main_faces > 0 else 0.0
    )

    features["num_main_midblink_faces"] = int(np.sum(main_midblink_flags)) if main_midblink_flags else 0
    features["num_main_closed_faces"] = int(np.sum(main_closed_flags)) if main_closed_flags else 0
    features["num_main_barelyopen_faces"] = int(np.sum(main_barelyopen_flags)) if main_barelyopen_flags else 0

    features["any_main_bad_eye"] = int(features["num_main_bad_eye_faces"] > 0)
    features["any_main_midblink"] = int(features["num_main_midblink_faces"] > 0)
    features["any_main_closed"] = int(features["num_main_closed_faces"] > 0)

    # Non-main bad-eye features: weaker / contextual signal.
    features["num_nonmain_faces"] = num_nonmain_faces
    features["num_nonmain_bad_eye_faces"] = int(np.sum(nonmain_bad_eye_flags)) if nonmain_bad_eye_flags else 0
    features["nonmain_bad_eye_ratio"] = (
    features["num_nonmain_bad_eye_faces"] / num_nonmain_faces
    if num_nonmain_faces > 0 else 0.0
    )

    features["num_nonmain_midblink_faces"] = int(np.sum(nonmain_midblink_flags)) if nonmain_midblink_flags else 0
    features["num_nonmain_closed_faces"] = int(np.sum(nonmain_closed_flags)) if nonmain_closed_flags else 0
    features["num_nonmain_barelyopen_faces"] = int(np.sum(nonmain_barelyopen_flags)) if nonmain_barelyopen_flags else 0

    features["any_nonmain_bad_eye"] = int(features["num_nonmain_bad_eye_faces"] > 0)

    # aggregate numeric features
    for name, values in {
        "focus": focus_scores,
        "user_facing": user_facing_scores,
        "confidence": confidences,
        "main_subject_score": main_subject_scores,
        "subject_prob_main": subject_prob_main,
        "subject_prob_secondary": subject_prob_secondary,
        "subject_prob_unimportant": subject_prob_unimportant,
        "yaw_abs": yaw_abs,
        "pitch_abs": pitch_abs,
        "roll_abs": roll_abs,
        "face_area_ratio": face_area_ratios,
        "face_center_x": face_center_x,
        "face_center_y": face_center_y,
    }.items():
        features[f"{name}_mean"] = safe_mean(values)
        features[f"{name}_max"] = safe_max(values)
        features[f"{name}_min"] = safe_min(values)

    # eye features
    for key, values in eye_probs.items():
        features[f"eye_{key}_mean"] = safe_mean(values)
        features[f"eye_{key}_max"] = safe_max(values)

    # main subject specific features
    features["main_focus_mean"] = safe_mean(main_focus_scores)
    features["main_focus_min"] = safe_min(main_focus_scores)
    features["main_focus_max"] = safe_max(main_focus_scores)

    features["main_midblink_mean"] = safe_mean(main_midblink_probs)
    features["main_midblink_max"] = safe_max(main_midblink_probs)

    features["main_open_mean"] = safe_mean(main_open_probs)
    features["main_open_min"] = safe_min(main_open_probs)
    features["main_open_max"] = safe_max(main_open_probs)

    features["main_closed_mean"] = safe_mean(main_closed_probs)
    features["main_closed_max"] = safe_max(main_closed_probs)

    features["main_barelyopen_mean"] = safe_mean(main_barelyopen_probs)
    features["main_barelyopen_max"] = safe_max(main_barelyopen_probs)

    features["main_covered_mean"] = safe_mean(main_covered_probs)
    features["main_covered_max"] = safe_max(main_covered_probs)

    features["main_yaw_abs_mean"] = safe_mean(main_yaw_abs)
    features["main_yaw_abs_max"] = safe_max(main_yaw_abs)

    features["main_pitch_abs_mean"] = safe_mean(main_pitch_abs)
    features["main_pitch_abs_max"] = safe_max(main_pitch_abs)

    features["main_face_area_ratio_mean"] = safe_mean(main_face_area_ratios)
    features["main_face_area_ratio_max"] = safe_max(main_face_area_ratios)
    # composition rough features
    features["largest_face_area_ratio"] = safe_max(face_area_ratios)
    features["total_face_area_ratio"] = float(np.sum(face_area_ratios)) if face_area_ratios else 0.0

    #features["main_quality_penalty_mean"] = safe_mean(main_quality_penalties)
    #features["main_quality_penalty_max"] = safe_max(main_quality_penalties)
    #features["main_quality_penalty_min"] = safe_min(main_quality_penalties)

    features["main_focus_penalty_max"] = safe_max(main_focus_penalties)
    features["main_eye_penalty_max"] = safe_max(main_eye_penalties)
    features["main_pose_penalty_max"] = safe_max(main_pose_penalties)

    #features["all_weighted_quality_penalty_mean"] = safe_mean(all_quality_penalties)
    #features["all_weighted_quality_penalty_max"] = safe_max(all_quality_penalties)

    features["main_all_open_ratio"] = (
    float(np.mean(main_open_good_flags)) if main_open_good_flags else 0.0
    )

    features["num_main_bad_quality_faces"] = (
    int(np.sum(main_bad_quality_flags)) if main_bad_quality_flags else 0
    )

    features["any_main_bad_quality"] = int(features["num_main_bad_quality_faces"] > 0)

    features["group_open_prob_min"] = safe_min(open_probs)
    features["group_closed_prob_max"] = safe_max(closed_probs)
    features["group_covered_prob_max"] = safe_max(covered_probs)

    return features


def build_features(raw_path: Path, output_path: Path):
    df = pd.read_pickle(raw_path)

    feature_rows = []
    for _, row in df.iterrows():
        feature_rows.append(extract_image_features(row))

    feature_df = pd.DataFrame(feature_rows)

    # insurance: fill any remaining missing values
    feature_df = feature_df.fillna(0)

    feature_df.to_csv(output_path, index=False)

    print(f"Saved {output_path}")
    print("Shape:", feature_df.shape)
    print("Columns:", len(feature_df.columns))
    print()


def main():
    build_features(RAW_DIR / "train_raw.pkl", OUTPUT_DIR / "train_features.csv")
    build_features(RAW_DIR / "valid_raw.pkl", OUTPUT_DIR / "valid_features.csv")
    build_features(RAW_DIR / "test_raw.pkl", OUTPUT_DIR / "test_features.csv")

def _looks_like_image_record(item):
    if not isinstance(item, dict):
        return False

    return (
        "image_id" in item
        or "faces" in item
        or "scene_id" in item
    )


def _extract_records_from_json(data):
    """
    Robustly extract image records from different possible JSON structures.
    Supports:
    1. [ {...}, {...} ]
    2. {"images": [ ... ]}
    3. {"detections": [ ... ]}
    4. {"records": [ ... ]}
    5. {"some_id": {...}, "some_id2": {...}}
    """

    if isinstance(data, list):
        return [item for item in data if _looks_like_image_record(item)]

    if isinstance(data, dict):
        for key in ["images", "detections", "records", "data", "items"]:
            if key in data and isinstance(data[key], list):
                return [item for item in data[key] if _looks_like_image_record(item)]

        if _looks_like_image_record(data):
            return [data]

        records = []
        for value in data.values():
            if _looks_like_image_record(value):
                records.append(value)
            elif isinstance(value, list):
                records.extend([item for item in value if _looks_like_image_record(item)])

        return records

    return []


def build_features_from_uploaded_json(data, default_shoot_id="uploaded"):
    """
    Convert uploaded face_detections.json content into the same feature dataframe
    used during training.
    """

    records = _extract_records_from_json(data)

    rows = []

    for idx, image in enumerate(records):
        faces = image.get("faces", [])

        raw_rank = image.get("rank", None)

        try:
              rank = float(raw_rank) if raw_rank is not None else None
        except (TypeError, ValueError):
              rank = None

           # In this dataset, valid ranks start from 1.
           # rank <= 0 should be treated as missing / invalid.
        if rank is not None and rank <= 0:
             rank = None

        row = {
            "shoot_id": image.get("shoot_id", default_shoot_id),
            "image_id": image.get("image_id", f"uploaded_image_{idx}"),
            "scene_id": image.get("scene_id", "uploaded_scene"),
            "rank": rank,
            

            "num_faces": image.get("num_faces", len(faces) if isinstance(faces, list) else 0),
            "category": image.get("category", "unknown"),
            "is_group_pose": image.get("is_group_pose", False),

            "image_width": image.get("image_width", image.get("width", 1)),
            "image_height": image.get("image_height", image.get("height", 1)),

            "faces": faces,
            "group_metrics": image.get("group_metrics", {}),
        }

        rows.append(extract_image_features(row))

    feature_df = pd.DataFrame(rows)

    if len(feature_df) == 0:
     return pd.DataFrame()

# Do not fill missing rank with 0.
# rank=NaN means invalid / missing label and should not be evaluated.
    metadata_cols = ["shoot_id", "scene_id", "image_id", "rank"]
    feature_cols = [c for c in feature_df.columns if c not in metadata_cols]

    feature_df[feature_cols] = feature_df[feature_cols].fillna(0)

    return feature_df

#if __name__ == "__main__":
#    main()

