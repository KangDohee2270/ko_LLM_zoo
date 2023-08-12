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

## Models
### Overview of existing models
Models were selected based on the following criteria

- Models with more than 50 stars on github
- Models with easily accessible weights on huggingface
The selected models are followed:

|model name|base_model|params(B)|dataset for finetuning|
|:--------:|:--------:|:-------:|:--------------------:|
|polyglot-ko|-|12.8|-|
|KoAlpaca|polyglot-ko|12.8|네이버 지식인 베스트|
|KORani|LLaMA|13|ShareGPT, KoVicuna|
|KoVicuna|LLaMA|7|ShareGPT|
|KULLM|polyglot-ko|12.8|GPT4ALL, Dolly, Vicuna|

- All models except KoVicuna have options for the number of parameters / base_model, but only one case was selected and configured as a starting point.
- All models except KoAlpaca were translated from foreign language open datasets into Korean and used for fine-tuning.

