# Benchmarking Result
- Benchmark evaluation results for natively supported models.
- All code for evaluation is based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/polyglot) repository. 
- All results are extracted from `evaluation.py` in that repository and are presented on a zero-shot basis. Here's an example of the executable code:

  ```python
  python evaluation.py --model polyglot-ko --tasks kobest_copa kobest_hellaswag --batch_size 64 -num_gpu 1,2,3
  ```
## KoBEST

| model | BoolQ | COPA | HellaSwag | SentiNeg | WiC |
| --- | --- | --- | --- | --- | --- |
| polyglot-12.8b | 0.4818 | **0.7931** | **0.4828** | **0.9117** | 0.3280 |
| koalpaca | **0.6461** | 0.7589 | 0.4086 | 0.8435 | **0.5243** |
| kullm | 0.3453 | 0.7857 | 0.4410 | 0.8285 | 0.3280 |
| korani | 0.3343 | 0.5499 | 0.3576 | 0.3526 | 0.3280 |
| kovicuna | 0.3343 | 0.5275 | 0.3571 | 0.4185 | 0.3280 |
| kogpt | 0.4514 | 0.7345 | 0.4599 | 0.3747 | 0.4517 |

- These results are based on the MACRO-F1 score.
## KLUE
N/A
## NSMC
N/A
## Korean-hate-speech
N/A
## Korean-unsmile-dataset
N/A
