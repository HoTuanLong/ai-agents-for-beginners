import argparse
import gc
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .config import ExperimentConfig
from .data import (
    NUM_TYPE_CLASSES,
    build_xy,
    filter_csds_objects_all,
    load_dataset,
    prepare_inputs_and_targets,
)
from .metrics import build_metrics_fn
from .model import CustomModel, load_model_hf, load_optimizer, get_losses

ULA_SUBSET_DOCS = [
    "ula/119CWL041",
    "ula/RindnerBonnie",
    "ula/HistoryGreek",
    "ula/Article247_3500",
    "ula/NapierDianne",
    "ula/sw2071-UTF16-ms98-a-trans",
    "ula/118CWL050",
    "ula/114CUL059",
    "ula/110CYL067",
    "ula/PolkMaria",
    "ula/116CUL034",
    "ula/115CVL037",
    "ula/118CWL049",
    "ula/Article247_66",
    "ula/110CYL068",
    "ula/113CWL017",
    "ula/112C-L015",
    "ula/115CVL036",
    "ula/115CVL035",
    "ula/Article247_328",
    "ula/114CUL060",
    "ula/112C-L012",
    "ula/118CWL048",
    "ula/ReidSandra",
    "ula/112C-L016",
    "ula/HistoryJerusalem",
    "ula/110CYL070",
    "ula/sw2014-UTF16-ms98-a-trans",
    "ula/112C-L014",
    "ula/117CWL008",
    "ula/sw2078-UTF16-ms98-a-trans",
    "ula/110CYL071",
    "ula/114CUL057",
    "ula/116CUL032",
    "ula/110CYL069",
    "ula/117CWL009",
    "ula/110CYL072",
    "ula/chapter-10",
    "ula/116CUL033",
    "ula/ch5",
    "ula/sw2015-ms98-a-trans",
    "ula/113CWL018",
    "ula/110CYL200",
    "ula/Article247_327",
    "ula/114CUL058",
    "ula/112C-L013",
    "ula/Article247_500",
    "ula/Article247_400",
]

ULA_LU_SUBSET_DOCS = [
    "ula/A1.E2-NEW",
    "ula/wsj_1640.mrg-NEW",
    "ula/AFGP-2002-600045-Trans",
    "ula/20000410_nyt-NEW",
    "ula/20000415_apw_eng-NEW",
    "ula/AFGP-2002-602187-Trans",
    "ula/20000815_AFP_ARB.0084.IBM-HA-NEW",
    "ula/CNN_AARONBROWN_ENG_20051101_215800.partial-NEW",
    "ula/20000424_nyt-NEW",
    "ula/20000419_apw_eng-NEW",
    "ula/20000416_xin_eng-NEW",
    "ula/enron-thread-159550",
    "ula/wsj_2465",
    "ula/AFGP-2002-600002-Trans",
    "ula/ENRON-pearson-email-25jul02",
    "ula/im_401b_e73i32c22_031705-2",
    "ula/A1.E1-NEW",
    "ula/CNN_ENG_20030614_173123.4-NEW-1",
    "ula/20000420_xin_eng-NEW",
    "ula/IZ-060316-01-Trans-1",
    "ula/sw2025-ms98-a-trans.ascii-1-NEW",
    "ula/SNO-525",
    "ula/AFGP-2002-600175-Trans",
    "ula/602CZL285-1",
]

XBANK_DOCS = [
    "xbank/wsj_0904",
    "xbank/wsj_0760",
    "xbank/wsj_0713",
    "xbank/wsj_0709",
    "xbank/wsj_0706",
    "xbank/wsj_0662",
    "xbank/wsj_0558",
    "xbank/wsj_0555",
    "xbank/wsj_0551",
    "xbank/wsj_0542",
    "xbank/wsj_0541",
    "xbank/wsj_0332",
    "xbank/wsj_0292",
    "xbank/wsj_0189",
    "xbank/wsj_0316",
    "xbank/wsj_0175",
    "xbank/wsj_0321",
    "xbank/wsj_0176",
    "xbank/wsj_0173",
    "xbank/wsj_0026",
    "xbank/wsj_0324",
    "xbank/wsj_0187",
    "xbank/wsj_0356",
    "xbank/wsj_0325",
    "xbank/wsj_0340",
    "xbank/wsj_0679",
    "xbank/wsj_0695",
    "xbank/wsj_0661",
    "xbank/wsj_0570",
    "xbank/wsj_0557",
    "xbank/wsj_0751",
    "xbank/wsj_0805",
    "xbank/wsj_0762",
    "xbank/wsj_0736",
    "xbank/wsj_0806",
    "xbank/wsj_1040",
    "xbank/wsj_1039",
    "xbank/wsj_1042",
    "xbank/wsj_0568",
    "xbank/wsj_0778",
    "xbank/wsj_0160",
    "xbank/wsj_0136",
    "xbank/wsj_0135",
    "xbank/wsj_0127",
    "xbank/wsj_0122",
    "xbank/wsj_0032",
    "xbank/wsj_0150",
    "xbank/wsj_0165",
    "xbank/wsj_0157",
    "xbank/wsj_0151",
    "xbank/wsj_0685",
    "xbank/wsj_0168",
    "xbank/wsj_0167",
    "xbank/wsj_0161",
    "xbank/wsj_0152",
    "xbank/wsj_0073",
    "xbank/wsj_0068",
    "xbank/wsj_0171",
    "xbank/wsj_0144",
    "xbank/wsj_0991",
    "xbank/wsj_0923",
    "xbank/wsj_0907",
    "xbank/wsj_0811",
    "xbank/wsj_0667",
    "xbank/wsj_0534",
    "xbank/wsj_0924",
    "xbank/wsj_0815",
    "xbank/wsj_1038",
    "xbank/wsj_1035",
    "xbank/wsj_1033",
    "xbank/wsj_0527",
    "xbank/wsj_0928",
    "xbank/wsj_0973",
    "xbank/wsj_0950",
    "xbank/wsj_0927",
    "xbank/wsj_0376",
    "xbank/wsj_0660",
    "xbank/wsj_0650",
    "xbank/wsj_0266",
    "xbank/wsj_0006",
    "xbank/wsj_0768",
    "xbank/wsj_1073",
    "xbank/wsj_0816",
    "xbank/wsj_0610",
    "xbank/wsj_0583",
]

FOLDS = [
    ["IDs_trainset_fold_1", "IDs_validationset_fold_1", "IDs_testset_fold_1"],
    ["IDs_trainset_fold_2", "IDs_validationset_fold_2", "IDs_testset_fold_2"],
    ["IDs_trainset_fold_3", "IDs_validationset_fold_3", "IDs_testset_fold_3"],
    ["IDs_trainset_fold_4", "IDs_validationset_fold_4", "IDs_testset_fold_4"],
    ["IDs_trainset_fold_5", "IDs_validationset_fold_5", "IDs_testset_fold_5"],
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tokenize_inputs(tokenizer, input_format: str, X_text, X_head):
    if input_format == "ES":
        return tokenizer(X_head, X_text, truncation=True)
    if input_format == "SE":
        return tokenizer(X_text, X_head, truncation=True)
    raise ValueError(f"Unknown input format: {input_format}")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def sort_samples(X_tokenized, y):
    keys = [len(obj) for obj in X_tokenized["input_ids"]]
    sorted_idxs = np.argsort(keys)[::-1]
    X_tokenized_sorted = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
    }
    for i in sorted_idxs:
        X_tokenized_sorted["input_ids"].append(X_tokenized["input_ids"][i])
        X_tokenized_sorted["token_type_ids"].append(X_tokenized["token_type_ids"][i])
        X_tokenized_sorted["attention_mask"].append(X_tokenized["attention_mask"][i])
    y_sorted = np.array(y)[sorted_idxs].tolist()
    return X_tokenized_sorted, y_sorted


def get_dataset(X_tokenized, y):
    X_tokenized_sorted, y_sorted = sort_samples(X_tokenized, y)
    dataset = Dataset(X_tokenized_sorted, y_sorted)
    return dataset


def get_training_args(config: ExperimentConfig, fold_counter: int) -> TrainingArguments:
    training_args = TrainingArguments(
        output_dir=f"models/{config.experiment_name}_{fold_counter}",
        overwrite_output_dir=True,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.val_batch_size,
        fp16=config.fp16,
        fp16_opt_level=config.fp16_opt_level,
        evaluation_strategy=config.eval_strategy,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.logging_steps,
        save_total_limit=1,
        num_train_epochs=config.num_train_epochs,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        seed=config.seed,
        group_by_length=True,
        report_to="none",
        full_determinism=True,
    )
    print("Number of GPUs:", training_args.n_gpu)
    print("Parallel Mode:", training_args.parallel_mode)
    return training_args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BERT universal opinion model.")
    parser.add_argument("--data-url", type=str, default="", help="Dataset URL.")
    parser.add_argument("--splits-url", type=str, required=True, help="Splits JSON URL.")
    parser.add_argument("--file-address", type=str, default="dataset/MPQA2.0_cleaned.json")
    parser.add_argument("--fetch-from-web", action="store_true")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ExperimentConfig(model_name=args.model_name)
    config.apply_task_overrides()

    os.environ["PYTHONHASHSEED"] = str(config.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    device_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    print("Device:", device)

    start_time = datetime.now()

    csds_collection, csds_splits = load_dataset(
        fetch_from_web=args.fetch_from_web,
        data_url=args.data_url,
        splits_url=args.splits_url,
        file_address=args.file_address,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ignored_docs = ULA_SUBSET_DOCS + ULA_LU_SUBSET_DOCS + XBANK_DOCS

    folds_train_log: List[dict] = []
    folds_val_log: List[dict] = []
    folds_test_log: List[dict] = []

    for fold_counter in range(0, min(len(FOLDS), config.max_folds)):
        print(f"\n\033[1mFold {fold_counter + 1}:\033[0m")
        set_seed(config.seed)

        fold_keys = FOLDS[fold_counter]
        train_ids = csds_splits[fold_keys[0]]
        val_ids = csds_splits[fold_keys[1]]
        test_ids = csds_splits[fold_keys[2]]

        train_objects, val_objects, test_objects = filter_csds_objects_all(
            csds_collection["csds_objects"], train_ids, val_ids, test_ids
        )

        train_input_target_dict, n_train_samples = prepare_inputs_and_targets(
            train_objects, config.text_key, config.head_key, ignored_docs
        )
        val_input_target_dict, n_val_samples = prepare_inputs_and_targets(
            val_objects, config.text_key, config.head_key, ignored_docs
        )
        test_input_target_dict, n_test_samples = prepare_inputs_and_targets(
            test_objects, config.text_key, config.head_key, ignored_docs
        )

        X_train, y_train = build_xy(train_input_target_dict)
        X_val, y_val = build_xy(val_input_target_dict)
        X_test, y_test = build_xy(test_input_target_dict)

        X_train_text, X_train_head = X_train[:, 0].tolist(), X_train[:, 1].tolist()
        X_val_text, X_val_head = X_val[:, 0].tolist(), X_val[:, 1].tolist()
        X_test_text, X_test_head = X_test[:, 0].tolist(), X_test[:, 1].tolist()

        n_samples = n_train_samples + n_val_samples + n_test_samples
        print(f"Train set size:      {n_train_samples} \t({100 * n_train_samples / n_samples}%)")
        print(f"Validation set size: {n_val_samples} \t({100 * n_val_samples / n_samples}%)")
        print(f"Test set size:       {n_test_samples} \t({100 * n_test_samples / n_samples}%)")

        X_train_tokenized = tokenize_inputs(tokenizer, config.input_format, X_train_text, X_train_head)
        X_val_tokenized = tokenize_inputs(tokenizer, config.input_format, X_val_text, X_val_head)
        X_test_tokenized = tokenize_inputs(tokenizer, config.input_format, X_test_text, X_test_head)

        train_dataset = get_dataset(X_train_tokenized, y_train)
        val_dataset = get_dataset(X_val_tokenized, y_val)
        test_dataset = get_dataset(X_test_tokenized, y_test)

        if config.bce_weight_exponent != 0:
            pos_counts = np.sum(np.array(y_train), axis=0, dtype=float)
            neg_counts = len(y_train) - pos_counts
            pos_weights = neg_counts / (pos_counts + np.finfo(float).eps)
            pos_weights = pos_weights ** config.bce_weight_exponent
            pos_weights = torch.as_tensor(pos_weights, dtype=torch.float).to(device)
        else:
            pos_weights = torch.as_tensor([1] * NUM_TYPE_CLASSES, dtype=torch.float).to(device)

        type_loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        polarity_loss_fct = nn.CrossEntropyLoss()
        intensity_loss_fct = nn.BCEWithLogitsLoss()

        model_hf = load_model_hf(config.model_name, config.freeze_layer_count)
        model = CustomModel(
            model_hf=model_hf,
            type_loss_fct=type_loss_fct,
            polarity_loss_fct=polarity_loss_fct,
            intensity_loss_fct=intensity_loss_fct,
            divide_loss_by=config.divide_loss_by,
            random_asynchronous_mtl=config.random_asynchronous_mtl,
            alpha_type=config.alpha_type,
            alpha_polarity=config.alpha_polarity,
            alpha_intensity=config.alpha_intensity,
        ).to(device)

        def get_losses_wrapper(logits, labels, batch_size):
            return get_losses(
                logits=logits,
                labels=labels,
                batch_size=batch_size,
                type_loss_fct=type_loss_fct,
                polarity_loss_fct=polarity_loss_fct,
                intensity_loss_fct=intensity_loss_fct,
                divide_loss_by=config.divide_loss_by,
            )

        callbacks = []
        if config.early_stopping > 0:
            callbacks.append(EarlyStoppingCallback(config.early_stopping))

        gc.collect()
        torch.cuda.empty_cache()
        training_args = get_training_args(config, fold_counter)
        optimizer = load_optimizer(model, config.learning_rate, config.lr_decay_factor)
        trainer = Trainer(
            model=model,
            args=training_args,
            optimizers=(optimizer, None),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=build_metrics_fn(
                type_threshold=config.type_threshold,
                alpha_type=config.alpha_type,
                alpha_polarity=config.alpha_polarity,
                alpha_intensity=config.alpha_intensity,
                get_losses_fn=get_losses_wrapper,
            ),
            callbacks=callbacks,
        )

        if not config.sequential_asynchronous_mtl:
            trainer.train()
        else:
            for alphas in config.sequential_asynchronous_mtl:
                config.alpha_type, config.alpha_polarity, config.alpha_intensity = alphas
                print("Alphas:", alphas)
                trainer.train()

        train_results = trainer.evaluate(train_dataset)
        val_results = trainer.evaluate(val_dataset)
        test_results = trainer.evaluate(test_dataset)

        folds_train_log.append(train_results)
        folds_val_log.append(val_results)
        folds_test_log.append(test_results)
        print("train_results =", train_results)
        print("val_results =", val_results)
        print("test_results =", test_results)

    print(folds_train_log)
    print(folds_val_log)
    print(folds_test_log)

    end_time = datetime.now()
    print("\033[1mStart:\033[0m {}".format(start_time))
    print("\033[1mEnd:\033[0m {}".format(end_time))
    print("\n\033[1mDuration:\033[0m {}".format(end_time - start_time))


if __name__ == "__main__":
    main()
