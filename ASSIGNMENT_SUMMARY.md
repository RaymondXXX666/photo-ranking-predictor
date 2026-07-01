# Photo Ranking Dataset - Assignment Overview

## Objective

Build a system to predict photo rankings from face detection features.

## Dataset

**10 wedding photo shoots** with ground truth rankings:
- 2,581 total images
- Rankings are **within-scene** (not global)
- Real-world data with realistic noise

## The Task

Given face detection data (bounding boxes, orientation, focus, eye state, etc.), predict which photos should be ranked higher within each scene.

**Use the `rank` field as your traget for training.**

## Approach

Your solution should include these three steps:

### 1. Data Cleaning & Feature Vector Building
- The dataset is noisy - clean it appropriately
- Extract and engineer features from the raw face detection data
- Handle images with multiple faces per image
- **Constraint**: For images with more than 20 faces, keep only 20 faces (selection strategy is up to you)

### 2. Train the Model
- Build a ranking model using the cleaned features
- Use `rank` as your target label
- Remember: rankings are **within-scene**, not global

### 3. Validation
- Evaluate your model's performance
- Measure how well it predicts rankings

## What's Provided

**Single JSON file per shoot** containing:
- Face bounding boxes (pixel coordinates)
- Image dimensions (for normalization)
- Scene assignments (`scene_id`)
- Ground truth rankings (`rank` field)
- Face attributes: orientation, focus, eye state, subject scores, etc.

See `README.md` for complete data format documentation.

## Important Clues


**About rankings:**
- Rankings are **within each scene**, not global
- `scene_id` groups images that should be ranked together
- Each scene has its own ranking: 1, 2, 3, ... (1 = best photo in that scene)

**About quality signals:**
- **Midblink is a strong negative signal** - images with midblink should rank lower
- Focus/sharpness affects perceived quality
- Subject importance (main vs secondary subjects) matters
- Head orientation and face position impact composition quality

## Constraints

- Maximum 20 faces per image (if more, select 20 using your strategy)
- Rankings must be computed within scenes
- Handle noisy/incomplete data appropriately


