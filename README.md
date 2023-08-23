# ko_LLM_zoos

## Introduction
Amidst the recent buzz surrounding ChatGPT's "AI supremacy," a new wave of curiosity has sparked in the realm of open-source language models (LLMs). This surge in interest has given rise to a diverse array of models. However, despite these advancements, the majority of LLM models continue to cater exclusively to the English language. Even the multilingual models that do exist often fall short when it comes to performance in other languages.

Enter the era of polyglot-ko—a time of rising fascination with Korean open-source LLM models. This very intrigue led me to embark on the ko_LLM_zoos project, designed to make a collection of Korean-based open-source LLM models readily accessible. Presently, the project features approximately five models accompanied by inference code. Looking ahead, our roadmap includes plans to incorporate fine-tuning, evaluation, and quantization functions to further enhance the project's offerings.

## Get Started
### Install
Run the following command to install the required packages:

```
pip install -r requirements.txt
```
### Inference
Run the following command to test the selected model:

```
python inference.py --model [model_name]
```
You can chat with the model in the terminal.

<details>
<summary>Configuration</summary>
<div markdown="1">

```
usage: inference.py [-h] --model MODEL [--max_new_token MAX_NEW_TOKEN] [--temp TEMP] [--top_p TOP_P] [--use_gradio] [--stream]
                    [--use_gpu USE_GPU] [--quant {gptq,int8,fp4,nf4,nf4-dq,np4-dq-comp_bf16}] [--gptq_weights GPTQ_WEIGHTS]

options:
  -h, --help            show this help message and exit
  --model MODEL         choose one model from [polygolot-ko, ko-alpaca, kullm, korani-v3, kovicuna, kogpt] or use saved path
  --max_new_token MAX_NEW_TOKEN
                        The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt
  --temp TEMP           A value used to modulate the next token probabilities. Higher values increase randomness.
  --top_p TOP_P         A value that controls the determinism with which the model generates responses. Higher values increase the
                        diversity of responses.
  --use_gradio          Use gradio for chat UI
  --stream              Use streaming for chat
  --use_gpu USE_GPU     The number of GPUs to use. If you want to use 0 and 1, enter '0, 1'
  --quant {gptq,int8,fp4,nf4,nf4-dq,np4-dq-comp_bf16}
                        Chosse quantization method. Note that the 'gptq' option must be preceded by quantization.py and use the stored
                        weights
  --gptq_weights GPTQ_WEIGHTS
                        The path where the model weight quantized via GPTQ is stored. If not specified, the gptq model will not be
                        available.
```

</div>
</details>

### Finetuning
Run the following command to test the selected model:

```
python finetuning.py --base_model [model_name] --data [dataset_for_finetuning]
```
You can fine-tune the base_model with your own dataset or an open dataset accessible from huggingface.

To create your own dataset, each sample in the dataset must follow the alpaca style below:
```
{'instruction': '...',
 'input': '', # Additional context. 
 'output': '...'}
```

<details>
<summary>Configuration</summary>
<div markdown="1">

```
usage: finetuning.py [-h] [--base_model BASE_MODEL] --data_path DATA_PATH [--args.output_dir ARGS.OUTPUT_DIR] [--use_gpu USE_GPU] [--batch_size BATCH_SIZE]
                     [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE] [--cutoff_len CUTOFF_LEN] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                     [--val_set_size VAL_SET_SIZE] [--eval_step EVAL_STEP] [--finetuning_method FINETUNING_METHOD] [--lora_r LORA_R] [--lora_alpha LORA_ALPHA]
                     [--lora_dropout LORA_DROPOUT] [--lora_target_modules LORA_TARGET_MODULES] [--train_on_inputs TRAIN_ON_INPUTS] [--group_by_length GROUP_BY_LENGTH]
                     [--wandb_project WANDB_PROJECT] [--wandb_run_name WANDB_RUN_NAME]

options:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        choose one model from [polygolot-ko, ko-alpaca, kullm, korani-v3, kovicuna, kogpt] or use saved path. The default is 'kullm'
  --data_path DATA_PATH
                        set your dataset path. the dataset must contain the keys: instruction, input and output
  --output_dir OUTPUT_DIR
                        save path for trained model weights
  --use_gpu USE_GPU     The number of GPUs to use. If you want to use 0 and 1, enter '0, 1'
  --batch_size BATCH_SIZE
  --num_epochs NUM_EPOCHS
  --learning_rate LEARNING_RATE
  --cutoff_len CUTOFF_LEN
                        The maximum number of input tokens
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Either training checkpoint or final adapter
  --val_set_size VAL_SET_SIZE
  --eval_step EVAL_STEP
                        Step unit to perform evaluation and checkpoint storage
  --finetuning_method FINETUNING_METHOD
                        Finetuning method. choose one of [lora, qlora]
  --lora_r LORA_R
  --lora_alpha LORA_ALPHA
  --lora_dropout LORA_DROPOUT
  --lora_target_modules LORA_TARGET_MODULES
  --train_on_inputs TRAIN_ON_INPUTS
                        If False, masks out inputs in loss
  --group_by_length GROUP_BY_LENGTH
                        Whether or not to group together samples of roughly the same length in the training dataset. If True, faster, but produces an odd training loss curve
  --wandb_project WANDB_PROJECT
  --wandb_run_name WANDB_RUN_NAME
                        Choose one of [false, gradients, all]. 'all' option may occur error: RuntimeError: 'histogram_cpu' not implemented for 'Char'
```

</div>
</details>

### Quantization
Run the following command to apply post-training quantization (PTQ) to the selected model:

```
python quantization.py --base_model [model_name] --output_dir [quantized_model_path]
```

Currently, the quantization code only supports [GPTQ](https://arxiv.org/abs/2210.17323).

## Models
### Overview of existing models
Models were selected based on the following criteria

- Models with more than 50 stars on github
- Models with easily accessible weights on huggingface

The selected models are followed:

|model name|base_model|params(B)|dataset for finetuning|
|:--------:|:--------:|:-------:|:--------------------:|
|[polyglot-ko](https://github.com/EleutherAI/polyglot)[🤗](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)|-|12.8|-|
|[KoAlpaca](https://github.com/Beomi/KoAlpaca)[🤗](https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B)|polyglot-ko|12.8|네이버 지식인 베스트|
|[KORani](https://github.com/krafton-ai/KORani)[🤗](https://huggingface.co/KRAFTON/KORani-v3-13B)|LLaMA|13|ShareGPT, KoVicuna|
|[KoVicuna](https://github.com/melodysdreamj/KoVicuna)[🤗](https://huggingface.co/junelee/ko_vicuna_7b)|LLaMA|7|ShareGPT|
|[KULLM](https://github.com/nlpai-lab/KULLM)[🤗](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2)|polyglot-ko|12.8|GPT4ALL, Dolly, Vicuna|
|[KoGPT](https://github.com/kakaobrain/kogpt)[🤗](https://huggingface.co/kakaobrain/kogpt)|-|6|ryan-dataset|

- All models except KoVicuna have options for the number of parameters / base_model, but only one case was selected and configured as a starting point.
The models in the table above are set as default, and other sizes can be loaded by entering the huggingface path directly.
- All models except KoAlpaca were translated from foreign language open datasets into Korean and used for fine-tuning.

## Supported benchmarks
- [KLUE](https://klue-benchmark.com/)
- [Ko-translation](https://huggingface.co/datasets/Moo/korean-parallel-corpora)
- [KoBEST](https://huggingface.co/datasets/skt/kobest_v1)
- [Korean-hate-speech](https://github.com/kocohub/korean-hate-speech)
- [KOLD](https://github.com/boychaboy/KOLD)
- [KorQuAD](https://korquad.github.io/KorQuad%201.0/)
- [Korean-unsmile-dataset](https://github.com/smilegate-ai/korean_unsmile_dataset)
- [KoSBi](https://github.com/naver-ai/korean-safety-benchmarks)
- [Korean legal precedent corpus](https://github.com/lbox-kr/lbox-open)
- [NSMC](https://github.com/e9t/nsmc)
