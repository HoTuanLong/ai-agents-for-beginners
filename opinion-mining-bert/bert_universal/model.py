import random
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .data import NUM_TYPE_CLASSES, NUM_POLARITY_CLASSES, NUM_INTENSITY_CLASSES


def load_model_hf(model_name: str, freeze_layer_count: Optional[int]) -> AutoModel:
    model_hf = AutoModel.from_pretrained(
        model_name,
        resume_download=True,
        config=AutoConfig.from_pretrained(model_name),
    )

    if freeze_layer_count is not None:
        for param in model_hf.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != 0:
            for layer in model_hf.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
        if freeze_layer_count == len(model_hf.encoder.layer):
            for param in model_hf.pooler.parameters():
                param.requires_grad = False

    pytorch_total_params = sum(
        p.numel() for p in model_hf.parameters() if p.requires_grad
    )
    print("Number of trainable parameters in bert:", pytorch_total_params)

    return model_hf


def load_optimizer(model: nn.Module, learning_rate: float, lr_decay_factor: float):
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    layers_name = [
        "pooler",
        "layer.11",
        "layer.10",
        "layer.9",
        "layer.8",
        "layer.7",
        "layer.6",
        "layer.5",
        "layer.4",
        "layer.3",
        "layer.2",
        "layer.1",
        "layer.0",
        "embeddings",
    ]
    layers_lr = [learning_rate * (lr_decay_factor**i) for i in range(len(layers_name))]
    print("Each layer's learning rate:", layers_lr)

    for (name, params) in named_parameters:
        lr = learning_rate
        for layer_name, layer_lr in zip(layers_name, layers_lr):
            if layer_name in name:
                lr = layer_lr
                break
        opt_parameters.append({"params": params, "lr": lr})

    return torch.optim.AdamW(opt_parameters, lr=learning_rate)


def get_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    type_loss_fct: nn.Module,
    polarity_loss_fct: nn.Module,
    intensity_loss_fct: nn.Module,
    divide_loss_by: str,
):
    type_logits = logits[:, :NUM_TYPE_CLASSES]
    polarity_logits = [
        logits[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 0 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]
    intensity_logits = [
        logits[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 6,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]

    type_targets = labels[:, :NUM_TYPE_CLASSES]
    polarity_targets = [
        labels[
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
        labels[
            :,
            NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 3 : NUM_TYPE_CLASSES
            + (NUM_POLARITY_CLASSES + NUM_INTENSITY_CLASSES) * i
            + 6,
        ]
        for i in range(NUM_TYPE_CLASSES)
    ]

    type_loss = type_loss_fct(type_logits, type_targets)

    polarity_loss = 0
    for batch_i in range(batch_size):
        for i in range(NUM_TYPE_CLASSES):
            polarity_loss += type_targets[batch_i, i] * polarity_loss_fct(
                polarity_logits[i][batch_i], polarity_targets[i][batch_i]
            )
    if divide_loss_by == "batch size":
        polarity_loss = polarity_loss / batch_size
    elif divide_loss_by == "number of annotations":
        polarity_loss = polarity_loss / torch.sum(type_targets)

    intensity_loss = 0
    for batch_i in range(batch_size):
        for i in range(NUM_TYPE_CLASSES):
            intensity_loss += type_targets[batch_i, i] * intensity_loss_fct(
                intensity_logits[i][batch_i], intensity_targets[i][batch_i]
            )
    if divide_loss_by == "batch size":
        intensity_loss = intensity_loss / batch_size
    elif divide_loss_by == "number of annotations":
        intensity_loss = intensity_loss / torch.sum(type_targets)

    return type_loss, polarity_loss, intensity_loss


class CustomModel(nn.Module):
    def __init__(
        self,
        model_hf: AutoModel,
        type_loss_fct: nn.Module,
        polarity_loss_fct: nn.Module,
        intensity_loss_fct: nn.Module,
        divide_loss_by: str,
        random_asynchronous_mtl: bool,
        alpha_type: float,
        alpha_polarity: float,
        alpha_intensity: float,
    ):
        super().__init__()
        self.model = model_hf
        self.classifier = nn.Linear(768, 35)
        self.type_loss_fct = type_loss_fct
        self.polarity_loss_fct = polarity_loss_fct
        self.intensity_loss_fct = intensity_loss_fct
        self.divide_loss_by = divide_loss_by
        self.random_asynchronous_mtl = random_asynchronous_mtl
        self.alpha_type = alpha_type
        self.alpha_polarity = alpha_polarity
        self.alpha_intensity = alpha_intensity

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        batch_size = len(input_ids)
        bert_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        bert_output = bert_output[0]
        cls_output = bert_output[:, 0, :].view(-1, 768)

        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            type_loss, polarity_loss, intensity_loss = get_losses(
                logits=logits,
                labels=labels,
                batch_size=batch_size,
                type_loss_fct=self.type_loss_fct,
                polarity_loss_fct=self.polarity_loss_fct,
                intensity_loss_fct=self.intensity_loss_fct,
                divide_loss_by=self.divide_loss_by,
            )
            if not self.random_asynchronous_mtl:
                loss = (
                    self.alpha_type * type_loss
                    + self.alpha_polarity * polarity_loss
                    + self.alpha_intensity * intensity_loss
                )
            else:
                random_task_selector = (
                    self.alpha_type + self.alpha_polarity + self.alpha_intensity
                ) * random.random()
                if random_task_selector < self.alpha_type:
                    loss = type_loss
                elif random_task_selector < self.alpha_type + self.alpha_polarity:
                    loss = polarity_loss
                else:
                    loss = intensity_loss

        return SequenceClassifierOutput(loss=loss, logits=logits)
