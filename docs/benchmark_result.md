# Benchmark Results
- The performance for each benchmark for the default models in this repository.
- All experiments were performed in a zero-shot state.

## KoBest
metric: macro-f1 score

| model | BoolQ | COPA | HellaSwag | SentiNeg | WiC |
| --- | --- | --- | --- | --- | --- |
| polyglot-ko | 0.4818 | **0.7931** | **0.4828** | **0.9117** | 0.3280 |
| ko-alpaca | **0.6461** | 0.7589 | 0.4086 | 0.8435 | **0.5243** |
| kullm | 0.3453 | 0.7857 | 0.4410 | 0.8285 | 0.3280 |
| korani-v3 | 0.3343 | 0.5499 | 0.3576 | 0.3526 | 0.3280 |
| kovicuna | 0.3343 | 0.5275 | 0.3571 | 0.4185 | 0.3280 |
| kogpt | 0.4514 | 0.7345 | 0.4599 | 0.3747 | 0.4517 |

## KLUE
N/A

## NSMC
metric: accuracy(std)
| model | acc(std) |
| --- | --- |
| polyglot-ko| 0.5654(0.0022) |
| koalpaca | 0.5704(0.0022) |
| kullm | 0.5536(0.0022) |
| korani-v3 | 0.6018(0.0022) |
| kovicuna | 0.564(0.0022) |
| kogpt | **0.6861(0.0021)** |

## KoHateSpeech
metric: accuracy(std)

| model | - | gen_bias | apeach |
| --- | --- | --- | --- |
| polyglot-12.8b | 0.3185(0.0215) | 0.1783(0.0177) | **0.5268(0.0081)** |
| koalpaca | 0.3609(0.0222) | **0.5180(0.0230)** | 0.4698(0.0081) |
| kullm | 0.3397(0.0218) | 0.1444(0.0162) | 0.5241(0.0081) |
| korani-v3 | 0.3312(0.0217) | 0.1423(0.0161) | 0.5082(0.0081) |
| kovicuna | **0.3800(0.0224)** | 0.1423(0.0161) | 0.5093(0.0081) |
| kogpt | 0.3694(0.0223) | 0.1423(0.0161) | 0.5098(0.0081) |

## KorUnsmile
N/A
