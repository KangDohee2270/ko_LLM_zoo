import os
import argparse
import json

ALL_TASKS = [
    "kold_level_a",
    "kold_level_b",
    "klue_sts",
    "klue_ynat",
    "klue_nli",
    "klue_mrc",
    "nsmc",
    "korquad",
    "kobest_boolq",
    "kobest_copa",
    "kobest_wic",
    "kobest_hellaswag",
    "kobest_sentineg",
    "ko_en_translation",
    "en_ko_translation",
    "korunsmile",
    "kohatespeech",
    "kohatespeech_gen_bias",
    "kohatespeech_apeach",
    "kolegal_legalcase",
    "kolegal_civilcase",
    "kolegal_criminalcase",
    "kosbi",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="choose one model from [polygolot-ko, ko-alpaca, kullm, korani-v3, kovicuna, kogpt] or use saved path",
    )
    parser.add_argument("--tasks", default=None, choices=ALL_TASKS, nargs="+")
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=512,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument(
        "--use_gpu",
        type=str,
        default="0",
        help="The number of GPUs to use. If you want to use 0 and 1, enter '0, 1'",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    # Pre-setting
    args.evaluation = True
    args.quant = None
    # Set which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    import eval_utils.eval_utils as eval_utils
    import eval_utils.evaluator as evaluator

    task_names = args.tasks if args.tasks != None else ALL_TASKS
    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    args.description_dict = description_dict
    results = evaluator.simple_evaluate(args)

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model}, limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))
