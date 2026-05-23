#!/usr/bin/env python
# coding=utf-8
"""Fine-tune and evaluate PhoBERT on UIT-VSFC sentiment analysis."""

import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="vinai/phobert-base",
        metadata={"help": "Model name or local path. Defaults to vinai/phobert-base."},
    )
    dataset_name: str = field(
        default="uitnlp/vietnamese_students_feedback",
        metadata={"help": "Hugging Face dataset name."},
    )
    text_column: str = field(default="sentence", metadata={"help": "Input text column."})
    label_column: str = field(default="sentiment", metadata={"help": "Label column for sentiment analysis."})
    max_seq_length: int = field(default=256, metadata={"help": "Maximum sequence length after tokenization."})
    pad_to_max_length: bool = field(default=True, metadata={"help": "Pad all samples to max_seq_length."})
    mode: str = field(
        default="me",
        metadata={"help": "Adapter mode. Use 'base' for LoRA or 'me' for MELoRA."},
    )
    rank: int = field(default=8, metadata={"help": "LoRA rank, or mini-LoRA rank when using MELoRA."})
    l_num: int = field(default=2, metadata={"help": "Number of mini-LoRAs for MELoRA."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_bias: str = field(default="none", metadata={"help": "Bias type for LoRA/MELoRA."})
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["query", "value"],
        metadata={"help": "Target module names for LoRA/MELoRA."},
    )
    use_fast_tokenizer: bool = field(default=False, metadata={"help": "Whether to use a fast tokenizer."})
    wandb_project: str = field(default="", metadata={"help": "Weights & Biases project name."})
    wandb_watch: str = field(default="", metadata={"help": "Weights & Biases watch setting."})
    wandb_log_model: str = field(default="", metadata={"help": "Weights & Biases model logging setting."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for models/datasets."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Optional train sample limit."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Optional eval sample limit."})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "Optional test sample limit."})


def get_label_info(raw_datasets: DatasetDict, label_column: str) -> Tuple[List[str], Optional[Dict[object, int]]]:
    label_feature = raw_datasets["train"].features[label_column]
    if hasattr(label_feature, "names") and label_feature.names:
        return list(label_feature.names), None

    raw_labels = set()
    for split in raw_datasets:
        raw_labels.update(raw_datasets[split].unique(label_column))

    ordered_labels = sorted(raw_labels)
    if label_column == "sentiment" and ordered_labels == [0, 1, 2]:
        label_names = ["negative", "neutral", "positive"]
    else:
        label_names = [str(label) for label in ordered_labels]

    return label_names, {label: index for index, label in enumerate(ordered_labels)}


def print_lora_parameters(model):
    trainable_params = 0
    lora_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            if "lora_" in name:
                lora_params += num_params
            elif "modules_to_save" not in name:
                logger.info("Trainable non-LoRA parameter: %s", name)

    logger.info(
        "lora params: %s || trainable params: %s || all params: %s || trainable%%: %.4f",
        f"{lora_params:,d}",
        f"{trainable_params:,d}",
        f"{all_param:,d}",
        100 * trainable_params / all_param,
    )


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    if model_args.wandb_project:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project
    if model_args.wandb_watch:
        os.environ["WANDB_WATCH"] = model_args.wandb_watch
    if model_args.wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = model_args.wandb_log_model

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )

    set_seed(training_args.seed)

    raw_datasets = load_dataset(model_args.dataset_name, cache_dir=model_args.cache_dir)
    for required_split in ("train", "validation"):
        if required_split not in raw_datasets:
            raise ValueError(f"Dataset {model_args.dataset_name} must contain a '{required_split}' split.")
    for column in (model_args.text_column, model_args.label_column):
        if column not in raw_datasets["train"].column_names:
            raise ValueError(
                f"Column '{column}' was not found in the train split. "
                f"Available columns: {raw_datasets['train'].column_names}"
            )

    label_list, data_label_to_id = get_label_info(raw_datasets, model_args.label_column)
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if "me" in model_args.mode:
        try:
            from peft import MELoraConfig
        except ImportError as exc:
            raise ImportError(
                "MELoraConfig was not found in the active peft package. "
                "Install this repo's PEFT fork first: cd peft-0.5.0 && pip install -e ."
            ) from exc

        logger.info("*** MELora !!! ***")
        peft_config = MELoraConfig(
            r=[model_args.rank] * model_args.l_num,
            lora_alpha=[model_args.lora_alpha] * model_args.l_num,
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            mode=model_args.mode,
            task_type="SEQ_CLS",
            modules_to_save=["classifier"],
        )
    elif "base" in model_args.mode:
        logger.info("*** Just Lora !!! ***")
        peft_config = LoraConfig(
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="SEQ_CLS",
            modules_to_save=["classifier"],
        )
    else:
        raise ValueError(f"Unknown mode {model_args.mode}")

    model = get_peft_model(model, peft_config)
    print_lora_parameters(model)

    if model_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length passed (%s) is larger than the tokenizer maximum (%s). Using %s.",
            model_args.max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )
    max_seq_length = min(model_args.max_seq_length, tokenizer.model_max_length)
    padding = "max_length" if model_args.pad_to_max_length else False

    def preprocess_function(examples):
        result = tokenizer(
            examples[model_args.text_column],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        if data_label_to_id is None:
            result["labels"] = examples[model_args.label_column]
        else:
            result["labels"] = [data_label_to_id[label] for label in examples[model_args.label_column]]
        return result

    with training_args.main_process_first(desc="dataset tokenization"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing UIT-VSFC sentences",
        )

    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]
        if model_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), model_args.max_train_samples)))

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]
        if model_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), model_args.max_eval_samples)))

    if training_args.do_predict:
        if "test" not in tokenized_datasets:
            raise ValueError(f"Dataset {model_args.dataset_name} must contain a 'test' split for --do_predict.")
        predict_dataset = tokenized_datasets["test"]
        if model_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(min(len(predict_dataset), model_args.max_predict_samples)))

    def compute_metrics(eval_prediction: EvalPrediction):
        predictions = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        predictions = np.argmax(predictions, axis=1)
        labels = eval_prediction.label_ids

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "weighted_f1": f1_score(labels, predictions, average="weighted"),
        }
        precision, recall, per_class_f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(len(label_list))),
            zero_division=0,
        )
        for label_id, label_name in enumerate(label_list):
            metric_name = str(label_name).replace(" ", "_").replace("/", "_")
            metrics[f"{metric_name}_precision"] = precision[label_id]
            metrics[f"{metric_name}_recall"] = recall[label_id]
            metrics[f"{metric_name}_f1"] = per_class_f1[label_id]
        return metrics

    if model_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        prediction_output = trainer.predict(predict_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", prediction_output.metrics)
        trainer.save_metrics("test", prediction_output.metrics)

        predictions = prediction_output.predictions[0] if isinstance(prediction_output.predictions, tuple) else prediction_output.predictions
        predictions = np.argmax(predictions, axis=1)
        raw_test = raw_datasets["test"]
        output_predictions_file = os.path.join(training_args.output_dir, "test_predictions.csv")

        def to_label_name(raw_label):
            label_id = data_label_to_id[raw_label] if data_label_to_id is not None else int(raw_label)
            return label_list[label_id]

        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w", encoding="utf-8", newline="") as writer:
                csv_writer = csv.writer(writer)
                csv_writer.writerow(["index", "sentence", "true_label", "predicted_label"])
                for index, predicted_id in enumerate(predictions):
                    csv_writer.writerow(
                        [
                            index,
                            raw_test[index][model_args.text_column],
                            to_label_name(raw_test[index][model_args.label_column]),
                            label_list[int(predicted_id)],
                        ]
                    )


if __name__ == "__main__":
    main()
