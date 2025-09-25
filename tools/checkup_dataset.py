import os
import random
import argparse
import torch
import clip
import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------
# 1) YOLO I/O
# ----------------------------------------------------
def load_yolo_file(txt_path):
    if not os.path.isfile(txt_path):
        return []
    lines = open(txt_path, "r").read().strip().splitlines()
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        records.append((cid, x, y, w, h))
    return records

# ----------------------------------------------------
# 2) Arg Parsing
# ----------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly pick images from dataset, check YOLO bboxes vs CLIP, compute mismatch stats, verify model classes vs label_list."
    )
    parser.add_argument("--images_path", type=str, required=True,
                        help="Folder with images (.jpg/.jpeg/.png).")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Folder with YOLO .txt for each image.")
    parser.add_argument("--model_path", type=str, default="./my_logreg_model.pkl",
                        help="Trained logistic regression model path.")
    parser.add_argument("--labelmap_path", type=str, default="./my_label_list.pkl",
                        help="Pickle with the label list (YOLO class order).")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Number of images to pick randomly.")
    return parser.parse_args()

# ----------------------------------------------------
# 3) Main
# ----------------------------------------------------
def main():
    args = parse_args()

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Load logistic regression + label list
    print("Loading logistic regression model from:", args.model_path)
    clf = joblib.load(args.model_path)
    label_list = joblib.load(args.labelmap_path)  # e.g. ["person","car","truck",...]

    # ----------------------------------------------------
    # Verify model classes_ vs label_list
    # ----------------------------------------------------
    # If clf.classes_ is numeric, skip. Usually it's an array of strings.
    classes_in_clf = clf.classes_
    if not all(isinstance(x, str) for x in classes_in_clf):
        print("[WARNING] clf.classes_ contains non-string labels. "
              "We can't reliably compare to label_list. Possibly numeric classes?")
    else:
        # 1) length check
        if len(classes_in_clf) != len(label_list):
            print(f"[WARNING] clf.classes_ length={len(classes_in_clf)} vs label_list length={len(label_list)}. "
                  "They should match if YOLO's 'cid' lines up with 'label_list' indexes.")
        # 2) set check
        set_clf = set(classes_in_clf)
        set_lbl = set(label_list)
        if set_clf != set_lbl:
            print("[WARNING] The sets of labels differ between model and label_list!")
            print("Model classes_ =", classes_in_clf)
            print("label_list    =", label_list)
        # 3) ordering check
        ordering_issue = False
        min_len = min(len(classes_in_clf), len(label_list))
        for i in range(min_len):
            if classes_in_clf[i] != label_list[i]:
                ordering_issue = True
                break
        if ordering_issue:
            print("[WARNING] The ordering of clf.classes_ differs from label_list. "
                  "Indices might not align with YOLO's 'cid' properly.")
        else:
            print("[INFO] The model classes_ seems to have the same set/order as label_list (at least in the first min length).")

    # Gather images
    image_files = [
        f for f in os.listdir(args.images_path)
        if f.lower().endswith((".jpg",".jpeg",".png"))
    ]
    if not image_files:
        print("No images found in", args.images_path)
        return

    random.shuffle(image_files)
    chosen = image_files[: args.num_images]
    print(f"Randomly selected {len(chosen)} images out of {len(image_files)} total.")

    total_bboxes = 0
    mismatch_count = 0

    # ----------------------------------------------------
    # 4) For each chosen image => check bounding boxes
    # ----------------------------------------------------
    from tqdm import tqdm
    for img_fn in tqdm(chosen, desc="Checking images"):
        base_name, _ = os.path.splitext(img_fn)
        label_path = os.path.join(args.labels_path, base_name + ".txt")

        yolo_records = load_yolo_file(label_path)
        if not yolo_records:
            continue

        full_img_path = os.path.join(args.images_path, img_fn)
        if not os.path.isfile(full_img_path):
            continue

        # Load PIL
        pil_img = Image.open(full_img_path).convert("RGB")
        w_img, h_img = pil_img.size

        for (cid, x_c, y_c, w_n, h_n) in yolo_records:
            total_bboxes += 1
            if 0 <= cid < len(label_list):
                yolo_lbl = label_list[cid]
            else:
                yolo_lbl = f"unknown_id_{cid}"

            # Crop
            x_min = (x_c - 0.5*w_n)* w_img
            y_min = (y_c - 0.5*h_n)* h_img
            x_max = x_min + w_n*w_img
            y_max = y_min + h_n*h_img
            box_img = pil_img.crop((x_min,y_min,x_max,y_max))

            # CLIP encode => logistic regression
            inp = preprocess(box_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = clip_model.encode_image(inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)

            proba = clf.predict_proba(feats_np)
            best_idx = np.argmax(proba, axis=1)[0]
            predicted_label = clf.classes_[best_idx]

            if not isinstance(predicted_label, str):
                # If numeric, map to label_list => but this is only correct if ordering is consistent
                predicted_label = label_list[predicted_label]

            if predicted_label != yolo_lbl:
                mismatch_count += 1

    if total_bboxes == 0:
        print("No bounding boxes processed. Exiting.")
        return

    mismatch_ratio = mismatch_count / total_bboxes
    print(f"Processed {len(chosen)} images, {total_bboxes} bounding boxes total.")
    print(f"Mismatched bounding boxes: {mismatch_count} => ratio={mismatch_ratio:.3f}")

if __name__ == "__main__":
    main()

