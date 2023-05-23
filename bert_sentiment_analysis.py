import argparse
import evaluate
import numpy as np
import os
import re
import sys
import torch
import transformers

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    Trainer,
    TrainingArguments,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def clean_text(sample):
    def remove_unicode_regex(text):
        # Define the regex pattern to match Unicode characters
        pattern = r"[\u0080-\uFFFF]"

        # Use regex substitution to remove the matched Unicode characters
        cleaned_text = re.sub(pattern, "", text)

        return cleaned_text

    sample["text"] = sample["text"].replace("@user", "")
    sample["text"] = sample["text"].replace("#", "")
    sample["text"] = sample["text"].replace("&amp;", "&")
    sample["text"] = sample["text"].replace("&lt;", "<")
    sample["text"] = sample["text"].replace("&gt;", ">")
    sample["text"] = sample["text"].strip()
    sample["text"] = remove_unicode_regex(sample["text"])

    return sample


def load_tweet_eval_dataset(tokenizer: BertTokenizer, seed: int = 42):
    tweet_eval_dataset = load_dataset("tweet_eval", name="sentiment")

    tweet_eval_dataset = (
        tweet_eval_dataset.map(clean_text).map(
            lambda sample: tokenizer(sample["text"], padding="max_length")
        )
    ).shuffle(seed=seed)

    train_dataset = tweet_eval_dataset["train"].with_format("torch")
    test_dataset = tweet_eval_dataset["test"].with_format("torch")

    return train_dataset, test_dataset


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pretrained", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)

    parser.add_argument("--r", type=int, default=4)

    parser.add_argument(
        "--fine-tuning",
        type=str,
        choices=["full", "top", "top2", "lora"],
        default="full",
    )
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading BERT model...")

    if args.use_pretrained:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=3
        )
    else:
        bert_config = BertConfig.from_pretrained("bert-base-cased")
        bert_config.num_labels = 3
        model = BertForSequenceClassification(bert_config)

    # freeze all but the final encoder
    if args.fine_tuning == "top":
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in model.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    # freeze all but the last two encoders
    elif args.fine_tuning == "top2":
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in model.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

    elif args.fine_tuning == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=args.r,
            lora_alpha=args.r,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    precision_metric = evaluate.load("precision")
    f1_metric = evaluate.load("f1")

    print("Loading Tweet Eval dataset...")
    train_dataset, test_dataset = load_tweet_eval_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        evaluation_strategy="steps",
        save_steps=1_000,
        eval_steps=1_000,
        save_total_limit=5,
        use_mps_device=get_device() == "mps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda pred: {
            "precision": precision_metric.compute(
                predictions=np.argmax(pred.predictions, axis=-1),
                references=pred.label_ids,
                average="macro",
            )["precision"],
            "f1": f1_metric.compute(
                predictions=np.argmax(pred.predictions, axis=-1),
                references=pred.label_ids,
                average="macro",
            )["f1"],
        },
    )

    print("Starting to train...")
    transformers.logging.set_verbosity_info()
    trainer.train()

    final_checkpoint = os.path.join(args.output_dir, "final-model".format(args.epochs))
    os.makedirs(final_checkpoint, exist_ok=True)
    trainer.save_model(final_checkpoint)

    final_metrics = trainer.evaluate()
    trainer.save_metrics("eval", final_metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
