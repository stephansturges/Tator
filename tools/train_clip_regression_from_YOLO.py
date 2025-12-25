#!/usr/bin/env python3
"""CLI wrapper around :func:`tools.clip_training.train_clip_from_yolo`."""
import argparse
import sys
from typing import Sequence

from tools.clip_training import TrainingError, train_clip_from_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLIP+LogReg from a YOLO dataset, preserving YOLO numeric class order.",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        required=True,
        help="Folder with images (.jpg, .png, etc.).",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Folder with YOLO .txt label files.",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="my_logreg_model.pkl",
        help="Where to save the trained logistic regression model.",
    )
    parser.add_argument(
        "--labelmap_output",
        type=str,
        default="my_label_list.pkl",
        help="Where to save the YOLO-ordered label list.",
    )
    parser.add_argument(
        "--input_labelmap",
        type=str,
        default=None,
        help="Existing label list (.pkl or .txt with one class per line).",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction (by image group).")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP backbone (e.g., ViT-B/32, ViT-L/14, ViT-L/14@336px)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: 'cuda' or 'cpu'.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for CLIP encoding.")
    parser.add_argument(
        "--min_per_class",
        type=int,
        default=2,
        help="Drop classes with fewer samples before splitting.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Class weighting for Logistic Regression.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularisation strength for Logistic Regression.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="saga",
        choices=["saga", "sag", "lbfgs", "liblinear", "newton-cg"],
        help="Logistic regression solver.",
    )
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="logreg",
        choices=["logreg", "mlp"],
        help="Classifier head type: logistic regression or a small MLP.",
    )
    parser.add_argument(
        "--mlp_hidden_sizes",
        type=str,
        default="256",
        help="Comma-separated hidden sizes for the MLP head (e.g., '512,256').",
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for MLP hidden layers.",
    )
    parser.add_argument(
        "--mlp_epochs",
        type=int,
        default=50,
        help="Epochs for MLP head training.",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=1e-3,
        help="Learning rate for MLP head training.",
    )
    parser.add_argument(
        "--mlp_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for MLP head training.",
    )
    parser.add_argument(
        "--mlp_label_smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for soft targets (0 disables).",
    )
    parser.add_argument(
        "--mlp_patience",
        type=int,
        default=6,
        help="Early-stop patience for MLP validation loss.",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse cached CLIP embeddings for this dataset if available.",
    )
    parser.add_argument(
        "--hard-example-mining",
        action="store_true",
        help="Run a short second pass focusing on misclassified / low-confidence samples.",
    )
    parser.add_argument(
        "--hard-misclassified-weight",
        type=float,
        default=3.0,
        help="Weight multiplier applied to misclassified samples during hard mining.",
    )
    parser.add_argument(
        "--hard-low-conf-weight",
        type=float,
        default=2.0,
        help="Weight multiplier applied to low-confidence samples during hard mining.",
    )
    parser.add_argument(
        "--hard-low-conf-threshold",
        type=float,
        default=0.65,
        help="Maximum predicted probability to consider a sample low-confidence (0 disables threshold).",
    )
    parser.add_argument(
        "--hard-margin-threshold",
        type=float,
        default=0.15,
        help="Minimum gap between top-1 and top-2 class probabilities before a sample is treated as ambiguous (0 disables margin check).",
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-4,
        help="Tolerance for detecting convergence; lower values force additional iterations.",
    )
    parser.add_argument(
        "--bg-classes",
        type=int,
        default=2,
        help="Number of hidden background classes to add (1–10).",
    )
    return parser.parse_args()


def _print_matrix(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> None:
    if not matrix:
        print("Confusion matrix is empty.")
        return
    header = "\t".join(["true\\pred"] + [str(lbl) for lbl in labels])
    print(header)
    for label, row in zip(labels, matrix):
        cells = "\t".join(str(int(v)) for v in row)
        print(f"{label}\t{cells}")


def main() -> None:
    args = parse_args()

    def emit(progress: float, message: str) -> None:
        print(f"[{progress * 100:5.1f}%] {message}")
        sys.stdout.flush()

    try:
        artifacts = train_clip_from_yolo(
            images_path=args.images_path,
            labels_path=args.labels_path,
            model_output=args.model_output,
            labelmap_output=args.labelmap_output,
            clip_model=args.clip_model,
            input_labelmap=args.input_labelmap,
            test_size=args.test_size,
            random_seed=args.random_seed,
            max_iter=args.max_iter,
            device=args.device,
            batch_size=args.batch_size,
            min_per_class=args.min_per_class,
            class_weight=args.class_weight,
            C=args.C,
            solver=args.solver,
            classifier_type=args.classifier_type,
            mlp_hidden_sizes=args.mlp_hidden_sizes,
            mlp_dropout=args.mlp_dropout,
            mlp_epochs=args.mlp_epochs,
            mlp_lr=args.mlp_lr,
            mlp_weight_decay=args.mlp_weight_decay,
            mlp_label_smoothing=args.mlp_label_smoothing,
            mlp_patience=args.mlp_patience,
            reuse_embeddings=args.reuse_embeddings,
            hard_example_mining=args.hard_example_mining,
            hard_mining_misclassified_weight=args.hard_misclassified_weight,
            hard_mining_low_conf_weight=args.hard_low_conf_weight,
            hard_mining_low_conf_threshold=args.hard_low_conf_threshold,
            hard_mining_margin_threshold=args.hard_margin_threshold,
            convergence_tol=args.convergence_tol,
            bg_class_count=args.bg_classes,
            progress_cb=emit,
        )
    except TrainingError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("\nTraining summary")
    print("================")
    print(f"Model path          : {artifacts.model_path}")
    print(f"Labelmap path       : {artifacts.labelmap_path}")
    print(f"Metadata path       : {artifacts.meta_path}")
    print(f"CLIP backbone       : {artifacts.clip_model}")
    print(f"Device              : {artifacts.device}")
    print(f"Train samples       : {artifacts.samples_train}")
    print(f"Test samples        : {artifacts.samples_test}")
    print(f"Classes seen        : {artifacts.classes_seen}")
    print(f"Class weight        : {artifacts.class_weight}")
    print(f"Solver              : {artifacts.solver}")
    print(f"Iterations run      : {artifacts.iterations_run}")
    print(f"Converged           : {artifacts.converged}")
    print(f"Hard example mining : {'yes' if artifacts.hard_example_mining else 'no'}")
    print(f"Accuracy            : {artifacts.accuracy:.4f}")

    print("\nClassification report:")
    print(artifacts.classification_report)

    print("Confusion matrix:")
    _print_matrix(artifacts.confusion_matrix, artifacts.label_order)

    if artifacts.per_class_metrics:
        print("\nPer-class metrics:")
        header = f"{'class':<20}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>10}"
        print(header)
        for entry in artifacts.per_class_metrics:
            label = str(entry.get('label', ''))
            precision = entry.get('precision')
            recall = entry.get('recall')
            f1 = entry.get('f1')
            support = entry.get('support')
            precision_str = f"{precision:.4f}" if precision is not None else "--"
            recall_str = f"{recall:.4f}" if recall is not None else "--"
            f1_str = f"{f1:.4f}" if f1 is not None else "--"
            support_str = str(int(support)) if support is not None else "--"
            print(f"{label:<20}{precision_str:>12}{recall_str:>12}{f1_str:>12}{support_str:>10}")


if __name__ == "__main__":
    main()
