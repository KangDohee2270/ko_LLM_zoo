#################
# Post-Training Quantization Code for Korean Opensource LLM

# Code Reference
# https://github.com/PanQiWei/AutoGPTQ

#################


import json
import random
import time
from argparse import ArgumentParser
import os

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer, pipeline

model_path_list = {
    "polyglot-ko": "EleutherAI/polyglot-ko-12.8b",
    "ko-alpaca": "beomi/KoAlpaca-Polyglot-12.8B",
    "kullm": "nlpai-lab/kullm-polyglot-12.8b-v2",
    "korani-v3": "KRAFTON/KORani-v3-13B",
    "kovicuna": "junelee/ko_vicuna_7b",
    "kogpt": {
        "pretrained_model_name_or_path": "kakaobrain/kogpt",
        "revision": "KoGPT6B-ryan1.5b-float16",
    },
}

PROMPT = {
    "prompt_input": "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:\n",
    "prompt_no_input": "아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{instruction}\n\n### 응답:\n",
    "response_split": "### 응답:",
}


def load_data(data_path, tokenizer, n_samples):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            prompt = (
                PROMPT["prompt_input"].format(input=inp, instruction=istr)
                if inp
                else PROMPT["prompt_no_input"].format(instruction=istr)
            )
            text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(
                tokenized_data["attention_mask"][: tokenizer.model_max_length]
            )
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update(
                {
                    i: f"{args.per_gpu_max_memory}GIB"
                    for i in range(torch.cuda.device_count())
                }
            )
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    if args.model in model_path_list.keys():
        model_path = model_path_list[args.model]
    elif not os.path.exists(args.model):
        raise FileNotFoundError(
            "The model path is invalid, make sure you are providing the correct path where the model weights are located"
        )
    else:
        model_path = args.model

    ############################# Set desc_act=True : for prevent below error: ##############################
    # torch._C._LinAlgError: linalg.cholesky: The factorization could not be completed
    # because the input is not positive-definite (the leading minor of order 20448 is not positive-definite)
    #########################################################################################################
    if type(model_path) == dict:
        tokenizer = AutoTokenizer.from_pretrained(
            **model_path,
            use_fast=args.fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoGPTQForCausalLM.from_pretrained(
            **model_path,
            quantize_config=BaseQuantizeConfig(
                bits=args.bits, group_size=args.group_size, desc_act=True
            ),
            max_memory=max_memory,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=args.fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config=BaseQuantizeConfig(
                bits=args.bits, group_size=args.group_size, desc_act=True
            ),
            max_memory=max_memory,
            trust_remote_code=args.trust_remote_code,
        )

    # Load KoAlpaca dataset of the link: https://github.com/Beomi/KoAlpaca/blob/main/ko_alpaca_data.json
    examples = load_data(
        "ko_llm_zoo/quantization/data/ko_alpaca_data.json", tokenizer, args.num_samples
    )
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        for example in examples
    ]

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton,
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if not args.output_dir:
        args.output_dir = args.model

    if args.save_and_reload:
        model.save_quantized(args.output_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_quantized(
            args.output_dir,
            device_map="auto",
            use_triton=args.use_triton,
            max_memory=max_memory,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=args.trust_remote_code,
        )

    pipeline_init_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )

    gptq_pipeline = pipeline(**pipeline_init_kwargs)
    for example in random.sample(examples, k=min(4, len(examples))):
        print(f"prompt: {example['prompt']}")
        print("-" * 42)
        print(f"golden: {example['output']}")
        print("-" * 42)
        start = time.time()
        generated_text = gptq_pipeline(
            example["prompt"],
            return_full_text=False,
            num_beams=1,
            max_length=len(example["input_ids"])
            + 128,  # use this instead of max_new_token to disable UserWarning when integrate with logging
        )[0]["generated_text"]
        end = time.time()
        print(f"Quantized_model: {generated_text}")
        num_new_tokens = len(tokenizer(generated_text)["input_ids"])
        print(
            f"generate {num_new_tokens} tokens using {end-start: .4f}s, {num_new_tokens / (end - start)} tokens/s."
        )
        print("=" * 42)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="group size, -1 means no grouping or full rank",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="how many samples will be used to quantize model",
    )
    parser.add_argument(
        "--save_and_reload",
        action="store_true",
        help="whether save quantized model to disk and reload back",
    )
    parser.add_argument(
        "--fast_tokenizer", action="store_true", help="whether use fast tokenizer"
    )
    parser.add_argument(
        "--use_triton",
        action="store_true",
        help="whether use triton to speedup at inference",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="max memory used to load model per gpu",
    )
    parser.add_argument(
        "--cpu_max_memory",
        type=int,
        default=None,
        help="max memory used to offload model to cpu",
    )
    parser.add_argument(
        "--quant_batch_size",
        type=int,
        default=1,
        help="examples batch size for quantization",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="whether to trust remote code when loading model",
    )
    parser.add_argument(
        "--use_gpu",
        type=str,
        default="0",
        help="The number of GPUs to use. If you want to use 0 and 1, enter '0, 1'",
    )
    args = parser.parse_args()

    main(args)
