import statistics
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from .data import (
    TYPE_CLASSES,
    NUM_TYPE_CLASSES,
    NUM_POLARITY_CLASSES,
    NUM_INTENSITY_CLASSES,
    POLARITY_DICT,
    INTENSITY_DICT,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_losses(
    preds,
    targets,
    alpha_type: float,
    alpha_polarity: float,
    alpha_intensity: float,
    get_losses_fn,
) -> Dict[str, float]:
    batch_size = len(preds)
    logits = torch.from_numpy(preds).cuda()
    labels = torch.from_numpy(targets).type(torch.float).cuda()

    type_loss, polarity_loss, intensity_loss = get_losses_fn(logits, labels, batch_size)

    type_loss = type_loss.cpu().numpy().copy() * alpha_type
    polarity_loss = polarity_loss.cpu().numpy().copy() * alpha_polarity
    intensity_loss = intensity_loss.cpu().numpy().copy() * alpha_intensity

    losses: Dict[str, float] = {}
    if alpha_type > 0:
        losses["type_cost"] = np.round(type_loss, 6)
    if alpha_polarity > 0:
        losses["polarity_cost"] = np.round(polarity_loss, 6)
    if alpha_intensity > 0:
        losses["intensity_cost"] = np.round(intensity_loss, 6)

    return losses


def calculate_type_metrics(preds, targets, type_threshold: float) -> Dict[str, float]:
    targets = targets[:, :NUM_TYPE_CLASSES]
    preds = preds[:, :NUM_TYPE_CLASSES]

    preds = sigmoid(preds)
    preds = np.array(preds > type_threshold, dtype=int)

    accuracy_list = np.sum(preds == targets, axis=0).astype(float) / preds.shape[0]
    f1_list = f1_score(y_true=targets, y_pred=preds, zero_division=0, average=None)
    precision_list = precision_score(
        y_true=targets, y_pred=preds, zero_division=0, average=None
    )
    recall_list = recall_score(y_true=targets, y_pred=preds, zero_division=0, average=None)

    tp = np.sum(preds & targets)
    fp = np.sum(preds & (1 - targets))
    fn = np.sum((1 - preds) & targets)

    micro_precision = tp / (tp + fp)
    micro_recall = tp / (tp + fn)
    micro_f1 = statistics.harmonic_mean([micro_precision, micro_recall])
    micro_accuracy = np.mean(preds == targets)

    exact_match_ratio = np.mean(
        np.sum(preds == targets, axis=1) == NUM_TYPE_CLASSES
    )

    return {
        "type_exact_match_ratio": np.round(exact_match_ratio, 6),
        "type_micro_f1": np.round(micro_f1, 6),
        "type_micro_precision": np.round(micro_precision, 6),
        "type_micro_recall": np.round(micro_recall, 6),
        "type_micro_accuracy": np.round(np.mean(micro_accuracy), 6),
        "type_f1": np.round(f1_list, 6).tolist(),
        "type_precision": np.round(precision_list, 6).tolist(),
        "type_recall": np.round(recall_list, 6).tolist(),
        "type_accuracy": np.round(accuracy_list, 6).tolist(),
    }


def calculate_polarity_f1_measure(extracted_polarity_preds, extracted_polarity_targets):
    extracted_polarity_targets = np.argmax(extracted_polarity_targets, axis=-1)
    extracted_polarity_preds = np.argmax(extracted_polarity_preds, axis=-1)
    polarity_weighted_f1 = f1_score(
        extracted_polarity_targets, extracted_polarity_preds, average="weighted"
    )
    return polarity_weighted_f1


def calculate_polarity_correctness_score(extracted_polarity_preds, extracted_polarity_targets):
    extracted_polarity_targets = np.argmax(extracted_polarity_targets, axis=-1)
    extracted_polarity_preds = np.argmax(extracted_polarity_preds, axis=-1)
    polarity_score = np.sum(extracted_polarity_preds == extracted_polarity_targets)
    return polarity_score


def calculate_pipeline_polarity_correctness_score(
    extracted_polarity_preds, extracted_polarity_targets, correctly_predicted_types
):
    extracted_polarity_targets = np.argmax(extracted_polarity_targets, axis=-1)
    extracted_polarity_preds = np.argmax(extracted_polarity_preds, axis=-1)
    polarity_score = np.sum(
        (extracted_polarity_preds == extracted_polarity_targets) * correctly_predicted_types
    )
    return polarity_score


def calculate_polarity_metrics(preds, targets, type_threshold: float, alpha_type: float,
                               alpha_polarity: float, alpha_intensity: float):
    batch_size = len(preds)

    type_targets = targets[:, :NUM_TYPE_CLASSES]
    type_preds = sigmoid(preds[:, :NUM_TYPE_CLASSES]) > type_threshold

    polarity_targets = [
        targets[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 0 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]
    filtered_preds = []
    for j in range(len(preds)):
        filtered_pred = preds[j, :].copy()
        for i in range(NUM_TYPE_CLASSES):
            if TYPE_CLASSES[i] != "expressive_subjectivity":
                filtered_pred[
                    NUM_TYPE_CLASSES
                    + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
                    + 1
                ] = -1 + min(
                    filtered_pred[
                        NUM_TYPE_CLASSES
                        + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
                        + 0
                    ],
                    filtered_pred[
                        NUM_TYPE_CLASSES
                        + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
                        + 2
                    ],
                )
        filtered_preds.append(np.array(filtered_pred))
    filtered_preds = np.array(filtered_preds)
    polarity_preds = [
        filtered_preds[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 0 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]

    extracted_polarity_targets = []
    extracted_polarity_preds = []
    correctly_predicted_types = []
    for batch_i in range(batch_size):
        for type_i in range(NUM_TYPE_CLASSES):
            if type_targets[batch_i, type_i] == 1:
                extracted_polarity_targets.append(polarity_targets[type_i][batch_i])
                extracted_polarity_preds.append(polarity_preds[type_i][batch_i])
                correctly_predicted_types.append(1 if type_preds[batch_i, type_i] else 0)
    extracted_polarity_targets = np.array(extracted_polarity_targets)
    extracted_polarity_preds = np.array(extracted_polarity_preds)
    correctly_predicted_types = np.array(correctly_predicted_types)

    polarity_acc = (
        calculate_polarity_correctness_score(
            extracted_polarity_preds, extracted_polarity_targets
        )
        / np.sum(type_targets)
    )
    polarity_weighted_f1 = calculate_polarity_f1_measure(
        extracted_polarity_preds, extracted_polarity_targets
    )

    metrics = {
        "polarity_accuracy": np.round(polarity_acc, 6),
        "polarity_weighted_f1": np.round(polarity_weighted_f1, 6),
    }

    if alpha_type == 0 or alpha_polarity == 0 or alpha_intensity == 0:
        polarity_pipeline_acc = 0
    else:
        polarity_pipeline_acc = (
            calculate_pipeline_polarity_correctness_score(
                extracted_polarity_preds,
                extracted_polarity_targets,
                correctly_predicted_types,
            )
            / np.sum(type_targets)
        )
        metrics["polarity_pipeline_accuracy"] = np.round(polarity_pipeline_acc, 6)

    return metrics


def get_intensity_id(vec):
    vec = vec.tolist()
    if vec == [1, 0, 0]:
        return 0
    if vec == [1, 1, 0]:
        return 1
    if vec == [0, 1, 0]:
        return 2
    if vec == [0, 1, 1]:
        return 3
    if vec == [0, 0, 1]:
        return 4
    return 1


def calculate_intensity_d_correctness_score(
    extracted_intensity_preds, extracted_intensity_targets, d=1
):
    batch_size = len(extracted_intensity_preds)
    intensity_score = 0
    for i in range(batch_size):
        target_id = get_intensity_id(extracted_intensity_targets[i])
        pred_id = get_intensity_id(extracted_intensity_preds[i])
        if abs(target_id - pred_id) <= d:
            intensity_score += 1
    return intensity_score


def calculate_intensity_f1_measure(extracted_intensity_preds, extracted_intensity_targets):
    whole = 1.0
    with_penalty = 0
    half_false = 1
    trues = {"medium": 0, "medium-high": 0, "low": 0, "high": 0, "low-medium": 0}
    cnt = {"medium": 0, "medium-high": 0, "low": 0, "high": 0, "low-medium": 0}
    falses = {"medium": 0, "medium-high": 0, "low": 0, "high": 0, "low-medium": 0}

    preds_indices = np.argmax(extracted_intensity_preds, axis=-1)

    for i in range(len(preds_indices)):
        if preds_indices[i] == 0:
            if extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 0 and extracted_intensity_targets[i][2] == 1:
                falses["low"] += 1
            if extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 1 and extracted_intensity_targets[i][2] == 0:
                falses["low"] += half_false
            if extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 1 and extracted_intensity_targets[i][2] == 1:
                falses["low"] += half_false

        if preds_indices[i] == 1:
            if extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 0 and extracted_intensity_targets[i][2] == 1:
                falses["medium"] += half_false
            if extracted_intensity_targets[i][0] == 1 and extracted_intensity_targets[i][1] == 0 and extracted_intensity_targets[i][2] == 0:
                falses["medium"] += half_false

        if preds_indices[i] == 2:
            if extracted_intensity_targets[i][0] == 1 and extracted_intensity_targets[i][1] == 0 and extracted_intensity_targets[i][2] == 0:
                falses["high"] += 1
            if extracted_intensity_targets[i][0] == 1 and extracted_intensity_targets[i][1] == 1 and extracted_intensity_targets[i][2] == 0:
                falses["high"] += 1
            if extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 1 and extracted_intensity_targets[i][2] == 1:
                falses["high"] += half_false
            if extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 1 and extracted_intensity_targets[i][2] == 0:
                falses["high"] += half_false

        if extracted_intensity_targets[i][0] == 1 and extracted_intensity_targets[i][1] == 0 and extracted_intensity_targets[i][2] == 0:
            cnt["low"] += 1
            if preds_indices[i] == 0:
                trues["low"] += whole

        elif extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 1 and extracted_intensity_targets[i][2] == 0:
            cnt["medium"] += 1
            if preds_indices[i] == 1:
                trues["medium"] += whole
            else:
                trues["medium"] += with_penalty

        elif extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 0 and extracted_intensity_targets[i][2] == 1:
            cnt["high"] += 1
            if preds_indices[i] == 2:
                trues["high"] += whole

        elif extracted_intensity_targets[i][0] == 1 and extracted_intensity_targets[i][1] == 1 and extracted_intensity_targets[i][2] == 0:
            cnt["low-medium"] += 1
            if preds_indices[i] == 0 or preds_indices[i] == 1:
                trues["low-medium"] += whole

        elif extracted_intensity_targets[i][0] == 0 and extracted_intensity_targets[i][1] == 1 and extracted_intensity_targets[i][2] == 1:
            cnt["medium-high"] += 1
            if preds_indices[i] == 1 or preds_indices[i] == 2:
                trues["medium-high"] += whole

    weighted_f1 = 0
    weights = 0
    for intensity_class in trues.keys():
        try:
            intensity_class_precision = trues[intensity_class] / (
                trues[intensity_class] + falses[intensity_class]
            )
        except ZeroDivisionError:
            intensity_class_precision = 1
        try:
            intensity_class_recall = trues[intensity_class] / cnt[intensity_class]
        except ZeroDivisionError:
            intensity_class_recall = 1
        intensity_class_f1 = statistics.harmonic_mean(
            [intensity_class_precision, intensity_class_recall]
        )
        weighted_f1 += intensity_class_f1 * cnt[intensity_class]
        weights += cnt[intensity_class]
    intensity_weighted_f1 = weighted_f1 / weights

    return intensity_weighted_f1


def calculate_intensity_correctness_score(
    extracted_intensity_preds, extracted_intensity_targets
):
    extracted_intensity_preds_argmax = np.argmax(extracted_intensity_preds, axis=-1)
    intensity_score = 0
    for i in range(len(extracted_intensity_preds_argmax)):
        if extracted_intensity_targets[i, extracted_intensity_preds_argmax[i]] == 1:
            intensity_score += 1
    return intensity_score


def calculate_pipeline_intensity_correctness_score(
    extracted_intensity_preds, extracted_intensity_targets, correctly_predicted_types
):
    extracted_intensity_preds_argmax = np.argmax(extracted_intensity_preds, axis=-1)
    intensity_score = 0
    for i in range(len(extracted_intensity_preds_argmax)):
        if correctly_predicted_types[i] and (
            extracted_intensity_targets[i, extracted_intensity_preds_argmax[i]] == 1
        ):
            intensity_score += 1
    return intensity_score


def calculate_intensity_metrics(
    preds, targets, type_threshold: float, alpha_type: float, alpha_polarity: float, alpha_intensity: float
):
    batch_size = len(preds)

    type_targets = targets[:, :NUM_TYPE_CLASSES]
    type_preds = sigmoid(preds[:, :NUM_TYPE_CLASSES]) > type_threshold
    intensity_targets = [
        targets[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 6,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]
    intensity_preds = [
        preds[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 6,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]

    extracted_intensity_targets = []
    extracted_intensity_preds = []
    correctly_predicted_types = []
    for batch_i in range(batch_size):
        for type_i in range(NUM_TYPE_CLASSES):
            if type_targets[batch_i, type_i] == 1:
                extracted_intensity_targets.append(intensity_targets[type_i][batch_i])
                extracted_intensity_preds.append(intensity_preds[type_i][batch_i])
                correctly_predicted_types.append(1 if type_preds[batch_i, type_i] else 0)
    extracted_intensity_targets = np.array(extracted_intensity_targets)
    extracted_intensity_preds = np.array(extracted_intensity_preds)
    correctly_predicted_types = np.array(correctly_predicted_types)

    number_of_samples = np.sum(type_targets)
    intensity_d0_acc = (
        calculate_intensity_d_correctness_score(
            extracted_intensity_preds, extracted_intensity_targets, d=0
        )
        / number_of_samples
    )
    intensity_d1_acc = (
        calculate_intensity_d_correctness_score(
            extracted_intensity_preds, extracted_intensity_targets, d=1
        )
        / number_of_samples
    )
    intensity_d2_acc = (
        calculate_intensity_d_correctness_score(
            extracted_intensity_preds, extracted_intensity_targets, d=2
        )
        / number_of_samples
    )
    intensity_d3_acc = (
        calculate_intensity_d_correctness_score(
            extracted_intensity_preds, extracted_intensity_targets, d=3
        )
        / number_of_samples
    )
    intensity_d4_acc = (
        calculate_intensity_d_correctness_score(
            extracted_intensity_preds, extracted_intensity_targets, d=4
        )
        / number_of_samples
    )
    intensity_acc = (
        calculate_intensity_correctness_score(
            extracted_intensity_preds, extracted_intensity_targets
        )
        / number_of_samples
    )
    intensity_weighted_f1 = calculate_intensity_f1_measure(
        extracted_intensity_preds, extracted_intensity_targets
    )

    metrics = {
        "intensity_exact_match_ratio": np.round(intensity_d0_acc, 6),
        "intensity_d1_accuracy": np.round(intensity_d1_acc, 6),
        "intensity_d2_accuracy": np.round(intensity_d2_acc, 6),
        "intensity_d3_accuracy": np.round(intensity_d3_acc, 6),
        "intensity_d4_accuracy": np.round(intensity_d4_acc, 6),
        "intensity_accuracy": np.round(intensity_acc, 6),
        "intensity_weighted_f1": np.round(intensity_weighted_f1, 6),
    }

    if alpha_type == 0 or alpha_polarity == 0 or alpha_intensity == 0:
        intensity_pipeline_acc = 0
    else:
        intensity_pipeline_acc = (
            calculate_pipeline_intensity_correctness_score(
                extracted_intensity_preds,
                extracted_intensity_targets,
                correctly_predicted_types,
            )
            / np.sum(type_targets)
        )
        metrics["intensity_pipeline_accuracy"] = np.round(intensity_pipeline_acc, 6)

    return metrics


def calculate_general_metrics(preds, targets, type_threshold: float):
    batch_size = len(preds)

    type_targets = targets[:, :NUM_TYPE_CLASSES]
    polarity_targets = [
        targets[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 0 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]
    intensity_targets = [
        targets[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 6,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]

    type_preds = sigmoid(preds[:, :NUM_TYPE_CLASSES]) > type_threshold
    polarity_preds = [
        preds[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 0 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]
    intensity_preds = [
        preds[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 6,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]

    custom_accuracy = 0
    exact_match_ratio = 0
    for batch_i in range(batch_size):
        exact_match_ratio_flag = True
        for type_i in range(NUM_TYPE_CLASSES):
            if type_preds[batch_i, type_i] == 1 and type_targets[batch_i, type_i] == 1:
                type_correctness_score = 1
                polarity_correctness_score = calculate_polarity_correctness_score(
                    polarity_preds[type_i][batch_i : batch_i + 1],
                    polarity_targets[type_i][batch_i : batch_i + 1],
                )
                intensity_correctness_score = calculate_intensity_correctness_score(
                    intensity_preds[type_i][batch_i : batch_i + 1],
                    intensity_targets[type_i][batch_i : batch_i + 1],
                )
                if polarity_correctness_score < 1 or intensity_correctness_score < 1:
                    exact_match_ratio_flag = False
                custom_accuracy += (
                    (type_correctness_score / 3)
                    + (polarity_correctness_score / 3)
                    + (intensity_correctness_score / 3)
                ) / NUM_TYPE_CLASSES
            elif type_preds[batch_i, type_i] == 0 and type_targets[batch_i, type_i] == 0:
                type_correctness_score = 1
                custom_accuracy += type_correctness_score / NUM_TYPE_CLASSES
            else:
                exact_match_ratio_flag = False

        if exact_match_ratio_flag:
            exact_match_ratio += 1

    custom_accuracy = custom_accuracy / batch_size
    exact_match_ratio = exact_match_ratio / batch_size

    return {
        "custom_accuracy": np.round(custom_accuracy, 6),
        "exact_match_ratio": np.round(exact_match_ratio, 6),
    }


def build_metrics_fn(
    type_threshold: float,
    alpha_type: float,
    alpha_polarity: float,
    alpha_intensity: float,
    get_losses_fn,
):
    def calculate_metrics(pred):
        targets = np.array(pred.label_ids, dtype=int)
        preds = pred.predictions
        n_tasks = 0

        metrics = {**calculate_losses(preds, targets, alpha_type, alpha_polarity, alpha_intensity, get_losses_fn)}
        average_sum = 0.0

        if alpha_polarity > 0:
            metrics.update(
                calculate_polarity_metrics(
                    preds, targets, type_threshold, alpha_type, alpha_polarity, alpha_intensity
                )
            )
            average_sum += metrics["polarity_accuracy"]
            n_tasks += 1

        if alpha_intensity > 0:
            metrics.update(
                calculate_intensity_metrics(
                    preds, targets, type_threshold, alpha_type, alpha_polarity, alpha_intensity
                )
            )
            average_sum += metrics["intensity_accuracy"]
            n_tasks += 1

        if alpha_type > 0:
            metrics.update(calculate_type_metrics(preds, targets, type_threshold))
            average_sum += metrics["type_micro_f1"]
            n_tasks += 1

        if alpha_type > 0 and alpha_polarity > 0 and alpha_intensity > 0:
            metrics.update(calculate_general_metrics(preds, targets, type_threshold))

        metrics["average_of_metrics"] = np.round(average_sum / n_tasks, 6)

        return metrics

    return calculate_metrics
