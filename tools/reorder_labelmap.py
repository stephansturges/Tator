"""
Reorder YOLO label map using model predictions (Hungarian assignment).

Purpose
- Given a trained CLIP+LogReg model and an existing YOLO label list, sample images,
  tally how often each YOLO class (cid) is predicted as each model class, then compute
  the best 1:1 assignment (YOLO cid -> model class) that maximizes agreement.
- Writes a proposed reordered label list to both .txt and .pkl.

Prereqs
- A trained model: my_logreg_model.pkl
- A label list: my_label_list.pkl (list of strings where index = YOLO cid)
- SciPy is required: pip install scipy

Usage
  python tools/reorder_labelmap.py \
    --images_path /path/to/images \
    --labels_path /path/to/labels \
    --model_path my_logreg_model.pkl \
    --labelmap_path my_label_list.pkl \
    --num_images 500 \
    --output_reordered_labelmap new_label_list

Outputs
- new_label_list.txt: one label per line (reordered to align with model predictions)
- new_label_list.pkl: same list serialized with joblib
"""

from __future__ import annotations

import os
import random
import argparse
from typing import List, Tuple


def load_yolo_file(txt_path: str) -> List[Tuple[int, float, float, float, float]]:
    if not os.path.isfile(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        records.append((cid, x, y, w, h))
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Propose a reordered YOLO label list using a CLIP+LogReg model via Hungarian assignment."
    )
    p.add_argument("--images_path", required=True, type=str, help="Folder with images (.jpg/.jpeg/.png)")
    p.add_argument("--labels_path", required=True, type=str, help="Folder with YOLO .txt files")
    p.add_argument("--model_path", type=str, default="./my_logreg_model.pkl", help="Trained logistic regression .pkl")
    p.add_argument("--labelmap_path", type=str, default="./my_label_list.pkl", help="Existing YOLO label list (.pkl)")
    p.add_argument("--num_images", type=int, default=500, help="How many images to sample for tallying")
    p.add_argument("--output_reordered_labelmap", type=str, default="new_label_list",
                   help="Base name for output files (.txt and .pkl)")
    p.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP backbone (e.g., ViT-B/32)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy imports for heavy deps so utilities can be imported without them
    import numpy as np
    import torch
    import clip
    import joblib
    from PIL import Image
    from tqdm import tqdm
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise SystemExit(
            "SciPy is required. Please install with `pip install scipy`. Error: %s" % e
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(args.clip_model, device=device)

    print("Loading logistic regression:", args.model_path)
    clf = joblib.load(args.model_path)
    label_list: List[str] = joblib.load(args.labelmap_path)

    model_classes = list(clf.classes_)
    print("Model classes_:", model_classes)
    print("YOLO label_list:", label_list)

    # Gather and sample images
    image_files = [
        f for f in os.listdir(args.images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(image_files)
    chosen = image_files[: max(0, min(args.num_images, len(image_files)))]
    print(f"Randomly selected {len(chosen)} images out of {len(image_files)} total.")

    num_yolo = len(label_list)
    num_model = len(model_classes)
    if num_yolo == 0 or num_model == 0:
        raise SystemExit("Empty label list or model classes. Aborting.")

    count_matrix = np.zeros((num_yolo, num_model), dtype=np.int32)

    for img_fn in tqdm(chosen, desc="Tallying"):
        base, _ = os.path.splitext(img_fn)
        label_path = os.path.join(args.labels_path, base + ".txt")
        yolo_records = load_yolo_file(label_path)
        if not yolo_records:
            continue

        full_img_path = os.path.join(args.images_path, img_fn)
        if not os.path.isfile(full_img_path):
            continue

        pil_img = Image.open(full_img_path).convert("RGB")
        w_img, h_img = pil_img.size

        for (cid, x_c, y_c, w_n, h_n) in yolo_records:
            if cid < 0 or cid >= num_yolo:
                continue
            x_min = (x_c - 0.5 * w_n) * w_img
            y_min = (y_c - 0.5 * h_n) * h_img
            x_max = x_min + w_n * w_img
            y_max = y_min + h_n * h_img

            # Clamp; skip empty/reversed
            x_min_c = max(0, min(x_min, w_img))
            x_max_c = max(0, min(x_max, w_img))
            y_min_c = max(0, min(y_min, h_img))
            y_max_c = max(0, min(y_max, h_img))
            if x_max_c <= x_min_c or y_max_c <= y_min_c:
                continue

            sub_img = pil_img.crop((x_min_c, y_min_c, x_max_c, y_max_c))

            # Encode and tally
            inp = preprocess(sub_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = clip_model.encode_image(inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)

            pred_proba = clf.predict_proba(feats_np)
            best_idx = int(np.argmax(pred_proba, axis=1)[0])
            count_matrix[cid, best_idx] += 1

    # Build square cost matrix, maximize matches via Hungarian (minimize negative counts)
    size = max(num_yolo, num_model)
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:num_yolo, :num_model] = -count_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    reorder_map: List[str] = [None] * num_yolo
    total_assigned = 0
    for i in range(len(row_ind)):
        r = int(row_ind[i])
        c = int(col_ind[i])
        if r < num_yolo and c < num_model:
            reorder_map[r] = model_classes[c]
            total_assigned += int(count_matrix[r, c])

    total_bboxes = int(count_matrix.sum())
    mismatch_bboxes = total_bboxes - total_assigned
    mismatch_ratio = (mismatch_bboxes / total_bboxes) if total_bboxes > 0 else 0.0

    print("\nSummary of assignment:")
    print(f"Total bboxes (sampled) = {total_bboxes}")
    print(f"Assigned (best match) = {total_assigned}, mismatch={mismatch_bboxes}, ratio={mismatch_ratio:.3f}")

    print("\nBest reordering (YOLO cid => model label):")
    new_label_list: List[str] = []
    for cid, model_lbl in enumerate(reorder_map):
        old_lbl = label_list[cid] if cid < len(label_list) else f"cid_{cid}"
        if model_lbl is None:
            model_lbl = "unmatched"
        print(f"  YOLO cid={cid}, old='{old_lbl}' => new='{model_lbl}'")
        new_label_list.append(model_lbl)

    out_txt = args.output_reordered_labelmap + ".txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for lbl in new_label_list:
            f.write(lbl + "\n")
    print(f"\nWrote proposed reorder (text) to '{out_txt}'.")

    out_pkl = args.output_reordered_labelmap + ".pkl"
    joblib.dump(new_label_list, out_pkl)
    print(f"Wrote proposed reorder (pkl) to '{out_pkl}'.")


if __name__ == "__main__":
    main()
