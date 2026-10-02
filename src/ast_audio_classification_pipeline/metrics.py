"""Evaluation metrics for multiclass audio classification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any


def _validate_num_classes(num_classes: int) -> None:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive int")


def _validate_class_indices(values: Sequence[int], num_classes: int, *, name: str) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must contain ints, got {type(value).__name__}")
        if not 0 <= value < num_classes:
            raise ValueError(f"{name} class index {value} outside [0, {num_classes - 1}]")


def multiclass_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Compute overall accuracy as the fraction of correctly classified clips."""
    if len(predictions) != len(targets):
        raise ValueError(f"length mismatch: predictions={len(predictions)}, targets={len(targets)}")
    if len(targets) == 0:
        raise ValueError("cannot compute accuracy over empty targets")
    correct = sum(1 for p, t in zip(predictions, targets, strict=True) if p == t)
    return correct / len(targets)


def confusion_matrix(
    predictions: Sequence[int], targets: Sequence[int], num_classes: int
) -> list[list[int]]:
    """Compute confusion matrix where rows are ground truth classes and columns are predicted classes."""
    _validate_num_classes(num_classes)
    if len(predictions) != len(targets):
        raise ValueError(f"length mismatch: predictions={len(predictions)}, targets={len(targets)}")
    _validate_class_indices(predictions, num_classes, name="predictions")
    _validate_class_indices(targets, num_classes, name="targets")
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(predictions, targets, strict=True):
        matrix[t][p] += 1
    return matrix


def per_class_metrics(
    predictions: Sequence[int], targets: Sequence[int], num_classes: int
) -> dict[int, dict[str, float]]:
    """Compute per-class precision, recall, and F1-score."""
    if len(predictions) != len(targets):
        raise ValueError(f"length mismatch: predictions={len(predictions)}, targets={len(targets)}")
    matrix = confusion_matrix(predictions, targets, num_classes)
    results: dict[int, dict[str, float]] = {}
    for c in range(num_classes):
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in range(num_classes) if other != c)
        fn = sum(matrix[c][other] for other in range(num_classes) if other != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        results[c] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": sum(matrix[c]),
        }
    return results


def macro_f1_score(
    predictions: Sequence[int], targets: Sequence[int], num_classes: int
) -> float:
    """Compute unweighted macro-averaged F1 score across all classes."""
    pcm = per_class_metrics(predictions, targets, num_classes)
    f1_sum = sum(metrics["f1"] for metrics in pcm.values())
    return round(float(f1_sum / num_classes), 4) if num_classes > 0 else 0.0


def majority_class_baseline(targets: Sequence[int], num_classes: int) -> dict[str, Any]:
    """Compute the accuracy of a constant trivial classifier predicting the majority class."""
    _validate_num_classes(num_classes)
    if len(targets) == 0:
        raise ValueError("cannot compute baseline over empty targets")
    _validate_class_indices(targets, num_classes, name="targets")
    counts = Counter(targets)
    majority_class, count = counts.most_common(1)[0]
    majority_acc = count / len(targets)
    return {
        "majority_class_index": int(majority_class),
        "majority_class_count": int(count),
        "majority_class_accuracy": round(float(majority_acc), 4),
        "num_classes": num_classes,
        "n_samples": len(targets),
    }


def evaluate_classification(
    predictions: Sequence[int],
    targets: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Comprehensive multi-class classification evaluation against baselines."""
    num_classes = len(class_names)
    if num_classes < 1:
        raise ValueError("class_names must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("class_names must contain non-empty strings")
    if len(set(class_names)) != num_classes:
        raise ValueError("class_names must be unique")
    acc = multiclass_accuracy(predictions, targets)
    pcm = per_class_metrics(predictions, targets, num_classes)
    macro_f1 = macro_f1_score(predictions, targets, num_classes)
    baseline = majority_class_baseline(targets, num_classes)
    cm = confusion_matrix(predictions, targets, num_classes)

    named_per_class = {class_names[c]: pcm[c] for c in range(num_classes)}

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": macro_f1,
        "per_class": named_per_class,
        "confusion_matrix": cm,
        "baseline": baseline,
        "accuracy_delta_vs_baseline": round(float(acc - baseline["majority_class_accuracy"]), 4),
        "num_samples": len(targets),
        "num_classes": num_classes,
    }
