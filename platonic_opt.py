import argparse
import os
import sys
import torch
import transformers

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    OPTForCausalLM,
    AutoTokenizer,
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


def load_plato_dataset(
    tokenizer: AutoTokenizer,
    data_dir: str,
    max_sequence_length: int = 2048,
    seed: int = 42,
):
    def _tokenize_document(e):
        outputs = tokenizer(
            e["text"],
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length,
            return_length=True,
            return_overflowing_tokens=True,
        )
        input_batch = []
        attention_mask_batch = []
        for length, input_ids, attention_mask in zip(
            outputs["length"], outputs["input_ids"], outputs["attention_mask"]
        ):
            if length == max_sequence_length:
                input_batch.append(input_ids)
                attention_mask_batch.append(attention_mask)
        return {
            "input_ids": input_batch,
            "attention_mask": attention_mask_batch,
            "labels": input_batch,
        }

    # use "Apology" as validation text
    VALIDATION_TEXT_ID = "1656.txt"

    train_data_files = [x for x in os.listdir(data_dir) if x != VALIDATION_TEXT_ID]

    dataset = load_dataset(
        "text",
        keep_linebreaks=False,
        sample_by="document",
        data_files={
            "train": [os.path.join(data_dir, f) for f in train_data_files],
            "validation": os.path.join(data_dir, VALIDATION_TEXT_ID),
        },
    )

    dataset = dataset.map(
        _tokenize_document, batched=True, remove_columns=["text"]
    ).shuffle(seed=seed)

    dataset = dataset.with_format("torch")

    return dataset["train"], dataset["validation"]


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
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

    args = parser.parse_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model {args.model_name}...")
    model = OPTForCausalLM.from_pretrained(args.model_name)

    # freeze all but the final encoder
    if args.fine_tuning == "top":
        for param in model.model.parameters():
            param.requires_grad = False

        for param in model.model.decoder.layers[-1].parameters():
            param.requires_grad = True

    # freeze all but the last two encoders
    elif args.fine_tuning == "top2":
        for param in model.model.parameters():
            param.requires_grad = False

        for param in model.model.decoder.layers[-2:].parameters():
            param.requires_grad = True

    elif args.fine_tuning == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.r,
            lora_alpha=args.r,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading Plato's texts...")
    train_dataset, eval_dataset = load_plato_dataset(tokenizer, args.data_dir)

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
        eval_dataset=eval_dataset,
    )

    initial_metrics = trainer.evaluate()
    trainer.save_metrics("initial", initial_metrics)

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
