from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    experiment_name: str = "bert"
    model_name: str = "bert-base-uncased"
    input_format: str = "ES"
    text_key: str = "clean_text"
    head_key: str = "clean_head"
    freeze_layer_count: Optional[int] = None
    learning_rate: float = 2.5e-5
    lr_decay_factor: float = 1.0
    type_threshold: float = 0.5
    intensity_threshold: float = 0.5
    fp16: bool = False
    fp16_opt_level: str = "O1"
    train_batch_size: int = 16
    val_batch_size: int = 64
    test_batch_size: int = 1
    logging_steps: int = 400
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_average_of_metrics"
    dropout: float = 0.1
    bce_weight_exponent: int = 0
    num_train_epochs: int = 20
    early_stopping: int = 3
    seed: int = 0
    random_asynchronous_mtl: bool = False
    sequential_asynchronous_mtl: List[tuple] = field(default_factory=list)
    type_only: bool = False
    polarity_only: bool = False
    intensity_only: bool = False
    alpha_type: float = 1.0
    alpha_polarity: float = 1.0
    alpha_intensity: float = 1.0
    divide_loss_by: str = "number of annotations"
    max_folds: int = 5
    show_wrong_predictions: bool = True
    show_all_predictions: bool = False
    store_results: List[str] = field(default_factory=list)

    def apply_task_overrides(self) -> None:
        if self.type_only:
            self.alpha_polarity = 0.0
            self.alpha_intensity = 0.0
        if self.polarity_only:
            self.alpha_type = 0.0
            self.alpha_intensity = 0.0
        if self.intensity_only:
            self.alpha_type = 0.0
            self.alpha_polarity = 0.0
