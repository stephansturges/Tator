import os
import re
import numpy as np
import torch
import clip
from PIL import Image
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------------------------------------
# 1) User Settings
# ------------------------------------------------------------------------------
FOLDER_PATH = "./crops"  # Adjust to your folder of labeled images
# Regex pattern for extracting a penultimate class, e.g. "...-CLASSNAME-35.jpg"
PATTERN = re.compile(r".+-([A-Za-z0-9_]+)-\d+\.jpg$")

MODEL_OUTPUT_PATH = "./my_logreg_model.pkl"   # File to save logistic regression
LABELS_OUTPUT_PATH = "./my_label_list.pkl"    # File to save discovered labels

TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_ITER = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
# 2) Load CLIP (ViT-B/32)
# ------------------------------------------------------------------------------
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)

# ------------------------------------------------------------------------------
# 3) Collect embeddings & labels
# ------------------------------------------------------------------------------
print(f"Scanning folder: {FOLDER_PATH}")
embeddings_list = []
labels_list = []

for filename in os.listdir(FOLDER_PATH):
    if not filename.lower().endswith(".jpg"):
        continue

    m = PATTERN.match(filename)
    if not m:
        # Skip files that don't match pattern
        continue

    class_name = m.group(1)
    full_path = os.path.join(FOLDER_PATH, filename)

    # Convert image to CLIP features
    pil_img = Image.open(full_path).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model.encode_image(input_tensor)
    feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    feats_np = feats.squeeze(0).cpu().numpy()

    embeddings_list.append(feats_np)
    labels_list.append(class_name)

print("Collected", len(embeddings_list), "images in total.")

# ------------------------------------------------------------------------------
# 4) Remove classes with <2 images
# ------------------------------------------------------------------------------
counts = Counter(labels_list)
final_embeddings = []
final_labels = []
for emb, lbl in zip(embeddings_list, labels_list):
    if counts[lbl] >= 2:
        final_embeddings.append(emb)
        final_labels.append(lbl)

embeddings = np.array(final_embeddings)
labels = np.array(final_labels)

print("After removing classes with <2 images, we have:")
print(len(embeddings), "images left.")
if len(embeddings) < 2:
    print("Not enough images to train. Exiting.")
    exit()

# ------------------------------------------------------------------------------
# 5) Train/Test Split
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels
)

# ------------------------------------------------------------------------------
# 6) Train a Logistic Regression
# ------------------------------------------------------------------------------
clf = LogisticRegression(
    random_state=RANDOM_SEED,
    max_iter=MAX_ITER,
    multi_class='auto'
)
clf.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 7) Evaluate on Test Set
# ------------------------------------------------------------------------------
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (y_pred == y_test).mean()
print(f"Accuracy = {accuracy:.3f}")

# ------------------------------------------------------------------------------
# 8) Save the trained model + label references
# ------------------------------------------------------------------------------
print(f"Saving trained model to '{MODEL_OUTPUT_PATH}'")
joblib.dump(clf, MODEL_OUTPUT_PATH)

# Also save the set of class labels if desired
all_classes = sorted(set(labels))
joblib.dump(all_classes, LABELS_OUTPUT_PATH)

print("Finished. The model and label list are saved.")
