#################
# Finetuning Code for Korean Opensource LLM

# Code Reference
# https://github.com/Beomi/KoAlpaca/blob/main/train.py
# https://github.com/nlpai-lab/KULLM

#################

import argparse
import os
import sys
from typing import Dict, Optional, Sequence

sys.path.append(os.path.abspath(os.path.join("ko_llm_zoo", "..")))
from ko_llm_zoo.utils.data_formatting import *
from ko_llm_zoo.utils.get_model import LLM


def train(args):
    # wandb setting
    # Check if parameter passed or if set within environ
    print(len(args.wandb_project))
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    # Set model to train
    llm = LLM(args)
    model, tokenizer = llm.model, llm.tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Load additional packages: to prevent issues with incorrect GPU settings
    # See https://github.com/pytorch/pytorch/issues/9158
    import torch
    from datasets import load_dataset
    from peft import set_peft_model_state_dict
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    per_device_train_batch_size = args.batch_size // torch.cuda.device_count()
    # NOTE: The number of accumulation step have to be equal to the (distributed) batch size
    gradient_accumulation_steps = per_device_train_batch_size
    # gradient_accumulation_steps = args.batch_size // per_device_train_batch_size

    # Set resume from checkpoint
    if args.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            args.resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Set dataset (tokenization and train-test split)
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        user_prompt, full_prompt = get_alpaca_format(data_point)

        tokenized_full_prompt = tokenize(full_prompt)
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)

        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
        return tokenized_full_prompt

    dataset_list = {
        "koalpaca": "beomi/KoAlpaca-v1.1a",
        "kullm": "nlpai-lab/kullm-v2",
        "guanaco-ko": "nlpai-lab/openassistant-guanaco-ko",
    }
    if args.data.endswith(".json") or args.data.endswith(".jsonl"):
        data = load_dataset("json", data_files=args.data)
    else:
        if args.data in dataset_list:
            data = load_dataset(dataset_list[args.data])
        elif args.data == "junelee/sharegpt_deepl_ko":
            raise Exception(
                "'sharegpt deepl ko' cannot be loaded directly from huggingface. Visit the link(https://huggingface.co/datasets/junelee/sharegpt_deepl_ko), download the json file you want to use, and enter the path of the file."
            )
        else:
            data = load_dataset(args.data)

    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # Set Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        fp16=True,
        optim="adamw_torch",
        evaluation_strategy="steps" if args.val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if args.val_set_size > 0 else None,
        save_steps=args.eval_steps,
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        load_best_model_at_end=True if args.val_set_size > 0 else False,
        group_by_length=args.group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=args.wandb_run_name if use_wandb else None,
        # find a batch size that will fit into memory automatically to avoid CUDA OOM
        auto_find_batch_size=True,
        # saves strategically selected activations throughout the computational graph
        # so only a fraction of the activations need to be re-computed for the gradients.
        gradient_checkpointing=True,
        # Set a lower batch size than during training to avoid memory usage spikes during validation.
        per_device_eval_batch_size=per_device_train_batch_size // 4,
        eval_accumulation_steps=args.batch_size // 4,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
        ),
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Clean up the GPU memory before training
    torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model/data params
    parser.add_argument(
        "--model",
        type=str,
        default="polyglot-ko",
        help="choose one model from [polygolot-ko, ko-alpaca, kullm, korani-v3] or use saved path. The default is 'kullm'",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="set your dataset path. the dataset must contain the keys: instruction, input and output",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./save_checkpoints",
        help="save path for trained model weights",
    )

    # training hyperparams
    parser.add_argument(
        "--use_gpu",
        type=str,
        default="0",
        help="The number of GPUs to use. If you want to use 0 and 1, enter '0, 1'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--cutoff_len",
        type=int,
        default=2048,
        help="The maximum number of input tokens",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="checkpoint",
        help="Either training checkpoint or final adapter",
    )

    # evaluation hyperparams
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Step unit to perform evaluation and checkpoint storage",
    )

    # lora hyperparams
    parser.add_argument(
        "--finetuning_method",
        type=str,
        default="lora",
        help="Finetuning method. choose one of [lora, qlora]",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    ####if you use KoGPT, set ["q_proj", "k_proj", "v_proj"]
    parser.add_argument(
        "--lora_target_modules",
        type=list,
        default=["query_key_value"],
    )

    # additional hyperparams
    parser.add_argument(
        "--train_on_inputs",
        type=bool,
        default=True,
        help="If False, masks out inputs in loss",
    )
    parser.add_argument(
        "--group_by_length",
        type=bool,
        default=False,
        help="Whether or not to group together samples of roughly the same length in the training dataset. If True, faster, but produces an odd training loss curve",
    )
    parser.add_argument(
        "--padding_side",
        type=str,
        default="left",
    )

    # wandb params
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--wandb_log_model", type=str, default="false", help="options: false | true"
    )
    parser.add_argument(
        "--wandb_watch",
        type=str,
        default="gradients",
        help="Choose one of [false, gradients, all]. 'all' option may occur error: RuntimeError: 'histogram_cpu' not implemented for 'Char'",
    )

    args = parser.parse_args()
    args.mode = "finetuning"
    train(args)
