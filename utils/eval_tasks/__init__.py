#################
# Evaluation Code for Korean Opensource LLM

# Code Reference
# https://github.com/EleutherAI/lm-evaluation-harness/tree/polyglot

#################


from pprint import pprint
from typing import List, Union

from utils import base
from . import kobest
from . import nsmc
from . import klue
from . import ko_translation
from . import korquad
from . import korunsmile
from . import kohatespeech
from . import legal_test
from . import kold
from . import kosbi

TASK_REGISTRY = {
    "kold_level_a": kold.KoldLevelA,
    "kold_level_b": kold.KoldLevelB,
    "klue_sts": klue.STS,
    "klue_ynat": klue.YNAT,
    "klue_nli": klue.NLI,
    "klue_mrc": klue.MRC,
    "nsmc": nsmc.NSMC,
    "korquad": korquad.Korquad,
    "kobest_boolq": kobest.BoolQ,
    "kobest_copa": kobest.COPA,
    "kobest_wic": kobest.WiC,
    "kobest_hellaswag": kobest.HellaSwag,
    "kobest_sentineg": kobest.SentiNeg,
    "ko_en_translation": ko_translation.KoEnTranslation,
    "en_ko_translation": ko_translation.EnKoTranslation,
    "korunsmile": korunsmile.KorUnSmile,
    "kohatespeech": kohatespeech.HateSpeech,
    "kohatespeech_gen_bias": kohatespeech.GenderBias,
    "kohatespeech_apeach": kohatespeech.Apeach,
    "kolegal_legalcase": legal_test.LegalBinary,
    "kolegal_civilcase": legal_test.LJPCivil,
    "kolegal_criminalcase": legal_test.LJPCriminal,
    "kosbi": kosbi.KoSBi,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))

_EXAMPLE_JSON_PATH = "split:key:/absolute/path/to/data.json"


def add_json_task(task_name):
    """Add a JSON perplexity task if the given task name matches the
    JSON task specification.

    See `json.JsonPerplexity`.
    """
    if not task_name.startswith("json"):
        return

    def create_json_task():
        splits = task_name.split("=", 1)
        if len(splits) != 2 or not splits[1]:
            raise ValueError(
                "json tasks need a path argument pointing to the local "
                "dataset, specified like this: json="
                + _EXAMPLE_JSON_PATH
                + ' (if there are no splits, use "train")'
            )

        json_path = splits[1]
        if json_path == _EXAMPLE_JSON_PATH:
            raise ValueError(
                "please do not copy the example path directly, but substitute "
                "it with a path to your local dataset"
            )
        return lambda: json.JsonPerplexity(json_path)

    TASK_REGISTRY[task_name] = create_json_task()


def get_task(task_name):
    try:
        add_json_task(task_name)
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
