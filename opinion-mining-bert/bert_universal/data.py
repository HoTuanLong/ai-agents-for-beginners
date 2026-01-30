from typing import Dict, Iterable, List, Tuple
import json
from urllib.request import urlopen

import numpy as np

TYPE_CLASSES = [
    "agreement",
    "arguing",
    "expressive_subjectivity",
    "intention",
    "sentiment",
]
POLARITY_CLASSES = ["negative", "neutral", "positive"]
INTENSITY_CLASSES = [
    "low",
    "low-medium",
    "medium",
    "medium-high",
    "high",
    "high-extreme",
    "extreme",
]
POLARITY_DICT = {"negative": [1, 0, 0], "neutral": [0, 1, 0], "positive": [0, 0, 1]}
INTENSITY_DICT = {
    "low": [1, 0, 0],
    "low-medium": [1, 1, 0],
    "medium": [0, 1, 0],
    "medium-high": [0, 1, 1],
    "high": [0, 0, 1],
    "high-extreme": [0, 0, 1],
    "extreme": [0, 0, 1],
}

NUM_TYPE_CLASSES = len(TYPE_CLASSES)
NUM_POLARITY_CLASSES = len(POLARITY_CLASSES)
NUM_INTENSITY_CLASSES = 3


def load_dataset(
    fetch_from_web: bool,
    data_url: str,
    splits_url: str,
    file_address: str,
) -> Tuple[Dict, Dict]:
    if fetch_from_web:
        response = urlopen(data_url)
        csds_collection = json.loads(response.read())
    else:
        with open(file_address) as file:
            csds_collection = json.load(file)
    response = urlopen(splits_url)
    csds_splits = json.loads(response.read())
    return csds_collection, csds_splits


def filter_csds_objects(csds_objects: Iterable[Dict], subset_ids: Iterable[str]) -> List[Dict]:
    subset_ids_set = set(subset_ids)
    return [
        csds_object
        for csds_object in csds_objects
        if csds_object["unique_id"] in subset_ids_set
    ]


def filter_csds_objects_all(
    csds_objects: Iterable[Dict],
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    test_ids: Iterable[str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    test_ids_set = set(test_ids)

    train_objects: List[Dict] = []
    val_objects: List[Dict] = []
    test_objects: List[Dict] = []

    for csds_object in csds_objects:
        if csds_object["unique_id"] in val_ids_set:
            val_objects.append(csds_object)
        elif csds_object["unique_id"] in test_ids_set:
            test_objects.append(csds_object)
        elif csds_object["unique_id"] in train_ids_set:
            train_objects.append(csds_object)

    return train_objects, val_objects, test_objects


def prepare_inputs_and_targets(
    csds_objects: Iterable[Dict],
    text_key: str,
    head_key: str,
    ignored_docs: Iterable[str],
) -> Tuple[Dict[Tuple[str, str], List[int]], int]:
    input_output_dict: Dict[Tuple[str, str], List[int]] = {}
    n_samples = 0

    for csds_object in csds_objects:
        doc_id = csds_object["doc_id"]
        text = csds_object[text_key]
        head = csds_object[head_key]
        annotype = csds_object["annotation_type"]
        polarity = csds_object["polarity"]
        intensity = csds_object["intensity"]

        if (
            annotype in TYPE_CLASSES
            and polarity in POLARITY_CLASSES
            and intensity in INTENSITY_CLASSES
            and doc_id not in ignored_docs
        ):
            if (text, head) not in input_output_dict:
                input_output_dict[(text, head)] = [0] * 35
                n_samples += 1

            type_id = TYPE_CLASSES.index(annotype)
            polarity_id_start = NUM_TYPE_CLASSES + type_id * (
                NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES
            )
            polarity_id_end = polarity_id_start + NUM_POLARITY_CLASSES
            intensity_id_start = polarity_id_end
            intensity_id_end = intensity_id_start + NUM_INTENSITY_CLASSES

            input_output_dict[(text, head)][type_id] = 1
            input_output_dict[(text, head)][polarity_id_start:polarity_id_end] = (
                POLARITY_DICT[polarity]
            )
            input_output_dict[
                (text, head)
            ][intensity_id_start:intensity_id_end] = INTENSITY_DICT[intensity]

    return input_output_dict, n_samples


def build_xy(input_target_dict: Dict[Tuple[str, str], List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array(list(input_target_dict.keys()))
    y = np.array(list(input_target_dict.values()))
    return X, y
