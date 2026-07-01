# Photo Ranking Dataset - Data Structure Documentation

## Live Demo Link: https://photo-ranking-predictor.vercel.app

## Overview

This dataset contains face detection data from 10 wedding photo shoots with ground truth rankings. All data is provided in a single JSON file per shoot for simplicity.

**Dataset Statistics:**
- **10 wedding shoots** (33-478 images each)
- **2,581 total images**
- **2,064 images (80%)** with valid rankings
- **517 images (20%)** without rankings (no faces detected)

## Directory Structure

```
data/
├── wedding_shoot_01/        # 33 images, 100% ranked
│   └── face_detections.json
├── wedding_shoot_02/        # 93 images, 95% ranked
│   └── face_detections.json
├── wedding_shoot_03/        # 124 images, 68% ranked
│   └── face_detections.json
├── wedding_shoot_04/        # 164 images, 72% ranked
│   └── face_detections.json
├── wedding_shoot_05/        # 222 images, 90% ranked
│   └── face_detections.json
├── wedding_shoot_06/        # 284 images, 95% ranked
│   └── face_detections.json
├── wedding_shoot_07/        # 387 images, 67% ranked
│   └── face_detections.json
├── wedding_shoot_08/        # 396 images, 46% ranked
│   └── face_detections.json
├── wedding_shoot_09/        # 400 images, 95% ranked
│   └── face_detections.json
└── wedding_shoot_10/        # 478 images, 94% ranked
    └── face_detections.json
```

## JSON Data Format

Each `face_detections.json` file contains all necessary data for that shoot:

### Complete Example

```json
{
  "detections": [
    {
      "image_id": "6404aa8d-dbd8-4e8a-a53f-39f5e6e5a0be_private-image-never-share",
      "image_path": "<removed_for_privacy>",
      
      // Image dimensions
      "image_width": 6000,
      "image_height": 4000,
      
      // Scene grouping - images in same scene should be ranked together
      "scene_id": "scene_0001",
      
      // Ground truth ranking (1=best within scene, null if no faces)
      "rank": 1,
      
      // Face detections
      "num_faces": 2,
      "is_group_pose": false,
      "category": "pair",
      
      "faces": [
        {
          "face_id": 1,
          
          // Bounding box (pixel coordinates)
          "bbox": {
            "x1": 2007,
            "y1": 2559,
            "x2": 2207,
            "y2": 2795
          },
          
          // Detection confidence
          "confidence": 0.9832,
          
          // Head orientation (degrees)
          "orientation": {
            "pitch": -19,   // Up/down: -90 to +90
            "yaw": -66,     // Left/right: -180 to +180
            "roll": 0       // Tilt: -180 to +180
          },
          "orientation_label": "left",
          
          // Focus/sharpness score
          "focus_score": 85.5,              // Range: 0-100
          "user_facing_score": 4.5,         // Range: 1-5
          "focus_prediction": "perfect",
          
          // Subject importance
          "main_subject_score": 0.75,           // Range: 0-1
          "subject_prob_main": 0.85,
          "subject_prob_secondary": 0.12,
          "subject_prob_unimportant": 0.03,
          "is_main_subject": true,
          "subject_class": 0,                   // 0=main, 2=secondary
          
          // Eye state probabilities
          "eye_state": {
            "prediction": "open",
            "probabilities": {
              "open": 0.9001,
              "closed": 0.0299,
              "midblink": 0.0034,
              "partiallyopen": 0.0439,
              "glasses": 0.0,
              "covered": 0.0001,
              "barelyopen": 0.0185,
              "sideFaceAppearsClosed": 0.0,
              "sideFaceAppearsOpen": 0.0
            }
          }
        }
      ]
    }
  ]
}
```

## Field Descriptions

### Image-Level Fields

| Field | Type | Description | Range/Format |
|-------|------|-------------|--------------|
| `image_id` | string | Unique image identifier | UUID format |
| `image_path` | string | Original path (privacy removed) | `<removed_for_privacy>` |
| `image_width` | integer | Image width in pixels | > 0 |
| `image_height` | integer | Image height in pixels | > 0 |
| `scene_id` | string | Scene grouping identifier | `scene_XXXX` |
| `rank` | integer/null | Learn2rank ground truth ranking | 1=best, null if no faces |
| `num_faces` | integer | Number of faces detected | ≥ 0 |
| `is_group_pose` | boolean | Whether image is a group photo | true/false |
| `category` | string | Photo category | "single", "pair", "group" |

### Face-Level Fields

| Field | Type | Description | Range/Format |
|-------|------|-------------|--------------|
| `face_id` | integer | Face index within image | 1, 2, 3, ... |
| `bbox` | object | Face bounding box in pixels | `{x1, y1, x2, y2}` |
| `confidence` | float | Face detection confidence | 0.0 - 1.0 |
| `orientation.pitch` | float | Head tilt up/down | -90° to +90° |
| `orientation.yaw` | float | Head rotation left/right | -180° to +180° |
| `orientation.roll` | float | Head tilt angle | -180° to +180° |
| `orientation_label` | string | Direction label | "frontal", "left", "right" |
| `focus_score` | float | Image sharpness/focus quality | 0 - 100 |
| `user_facing_score` | float | User-friendly focus score | 1 - 5 |
| `focus_prediction` | string | Focus quality label | "terrible", "bad", "soft", "ok", "good", "perfect" |
| `main_subject_score` | float | Primary subject probability | 0.0 - 1.0 |
| `subject_prob_main` | float | Main subject probability | 0.0 - 1.0 |
| `subject_prob_secondary` | float | Secondary subject probability | 0.0 - 1.0 |
| `subject_prob_unimportant` | float | Unimportant subject probability | 0.0 - 1.0 |
| `is_main_subject` | boolean | Is this the main subject? | true/false |
| `subject_class` | integer | Subject classification | 0=main, 2=secondary |
| `eye_state.prediction` | string | Top eye state | "open", "closed", "midblink", etc. |
| `eye_state.probabilities` | object | Eye state probabilities | 9 states, each 0.0-1.0 |

## Important Notes

### Scene Grouping
Images with the same `scene_id` belong to the same scene and should be ranked together:
- Rankings are **within-scene**, not global
- Each scene has its own ranking: 1, 2, 3, ...
- Different scenes can have images with the same rank number

### Null Rankings
Some images have `rank: null`. Approximately 20% of the dataset contains null rankings.

### Multiple Faces Per Image
Images can have 0 to N faces. The `num_faces` field indicates how many faces were detected. For ranking:
- Images with multiple faces may need aggregation (mean, max, weighted average)
- Each face has its own features (bbox, orientation, focus, etc.)
- The ranking is for the entire image, not individual faces

## Data Statistics by Shoot

| Shoot | Total Images | Ranked | Null | Scenes | Quality |
|-------|-------------|--------|------|--------|---------|
| wedding_shoot_01 | 33 | 33 (100%) | 0 | 11 | Excellent |
| wedding_shoot_02 | 93 | 88 (95%) | 5 | 58 | Excellent |
| wedding_shoot_03 | 124 | 84 (68%) | 40 | 45 | Fair |
| wedding_shoot_04 | 164 | 118 (72%) | 46 | 57 | Fair |
| wedding_shoot_05 | 222 | 199 (90%) | 23 | 62 | Good |
| wedding_shoot_06 | 284 | 270 (95%) | 14 | 43 | Excellent |
| wedding_shoot_07 | 387 | 260 (67%) | 127 | 88 | Fair |
| wedding_shoot_08 | 396 | 183 (46%) | 213 | 71 | Poor |
| wedding_shoot_09 | 400 | 379 (95%) | 21 | 204 | Excellent |
| wedding_shoot_10 | 478 | 450 (94%) | 28 | 261 | Excellent |
| **TOTAL** | **2,581** | **2,064 (80%)** | **517** | - | **Good** |

## Loading Data

```python
import json

# Load a shoot
with open('data/wedding_shoot_01/face_detections.json') as f:
    data = json.load(f)

detections = data['detections']

# Access fields
for det in detections:
    image_id = det['image_id']
    width = det['image_width']
    height = det['image_height']
    scene = det['scene_id']
    rank = det['rank']  # Can be null!
    
    # Process faces
    for face in det['faces']:
        bbox = face['bbox']
        orientation = face['orientation']
        focus = face['focus_score']
        # ... etc
```
