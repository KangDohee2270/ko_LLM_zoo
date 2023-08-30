# Evaluation Guide

- The performance for each benchmark for the default models in this repository.
- All experiments were performed in a zero-shot state.
- All evaluation codes are based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/polyglot). They have been modified and minified to fit in this repository.

## Usage

To evaluate a model against specific tasks, use the following command:

```python
python ko_llm_zoo/evaluation/evaluation.py --model [model_name] --tasks [task_list]
```

For example, to evaluate the polyglot-ko(12.8b) model with 'kobest_hellaswag' and 'kobest_copa' tasks, run:

```python
python ko_llm_zoo/evaluation/evaluation.py --model polyglot-ko --tasks kobest_hellaswag kobest_copa
```

**Note that unlike lm-eval-harness, when performing multiple tasks at once, each task is separated by a space.**


The list of tasks is as follows:
```
kold_level_a
kold_level_b
klue_sts
klue_ynat
klue_nli
klue_mrc
nsmc
korquad
kobest_boolq
kobest_copa
kobest_wic
kobest_hellaswag
kobest_sentineg
ko_en_translation
en_ko_translation
korunsmile
kohatespeech
kohatespeech_gen_bias
kohatespeech_apeach
kolegal_legalcase
kolegal_civilcase
kolegal_criminalcase
kosbi
```
