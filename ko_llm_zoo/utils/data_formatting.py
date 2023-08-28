from typing import Union, List
import re

# Default prompt. Format of KoAlpaca
prompt = {
    "prompt_input": "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:\n",
    "prompt_no_input": "아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{instruction}\n\n### 응답:\n",
    "response_split": "### 응답:",
}


def split_guanaco(
    text: str, pattern: str, human: str = "Human: ", assistant: str = "Assistant: "
):
    list_text = re.split(pattern, text)
    human_uttr, assistant_uttr = [], []
    for i in range(len(list_text)):
        if list_text[i] == human:
            i += 1
            human_uttr.append(list_text[i])
        elif list_text[i] == assistant:
            i += 1
            assistant_uttr.append(list_text[i])

    # if the final speaker is a human, the final utterance is ignored.
    if len(human_uttr) > len(assistant_uttr):
        human_uttr = human_uttr[: len(assistant_uttr)]

    # Convert text from conversational format to instruction-output format.
    converted_text = dict(instruction=human_uttr[-1], output=assistant_uttr[-1])

    # If it consists of two or more turns, the previous dialogue is provided
    # as an additional context, except for the final instruction.
    if len(human_uttr) > 1:
        input = ""
        for i in range(len(human_uttr) - 1):
            input += f"명령어: {human_uttr[i]}\n응답: {assistant_uttr[i]}\n"
        converted_text["input"] = input

    return converted_text


def split_sharegpt(data: List[dict]):
    human_uttr, assistant_uttr = [], []
    for uttr in data:
        if uttr["from"] == "human":
            human_uttr.append(uttr["value"])
        else:
            assistant_uttr.append(uttr["value"])

    # The case that a conversation starts from your assistant
    if len(assistant_uttr) > len(human_uttr):
        input = f"응답: {assistant_uttr[0]}\n"
        del assistant_uttr[0]
    else:
        input = ""
    # Convert text from conversational format to instruction-output format.
    converted_text = dict(instruction=human_uttr[-1], output=assistant_uttr[-1])
    # If it consists of two or more turns, the previous dialogue is provided
    # as an additional context, except for the final instruction.
    if len(human_uttr) > 1:
        for i in range(len(human_uttr) - 1):
            input += f"명령어: {human_uttr[i]}\n응답: {assistant_uttr[i]}\n"
        converted_text["input"] = input

    return converted_text


def convert_format(data: dict):
    # Convert data to text in alpaca format
    if "input" in data.keys():
        user_prompt = prompt["prompt_input"].format(
            input=data["input"], instruction=data["instruction"]
        )
    else:
        user_prompt = prompt["prompt_no_input"].format(instruction=data["instruction"])

    full_prompt = user_prompt + data["output"]

    return user_prompt, full_prompt


def get_alpaca_format(input: dict):
    if "text" in input.keys():
        ############################## guanaco-ko style #############################
        # only include "text" which consists of the full text of the
        # conversation between the human and the assistant

        # Human and assistant are distinguished by "### Human: " and "### Assistant: ".

        # There is a mix of multi and single-turn dialogues, so if it consists of more than 2 turns,
        # the previous dialogue before the final instruction is provided as additional context.

        # Also, in order to construct the input as an instruction-output pair
        # , if the final speaker is a human, the final utterance is ignored.
        #############################################################################
        full_text = input["text"]
        pattern = r"\#\#\# (\w+\: )"
        converted_text = split_guanaco(pattern=pattern, text=full_text)
        user_prompt, full_prompt = convert_format(converted_text)

        return user_prompt, full_prompt

    elif "conversations" in input.keys():
        ######################### sharegpt_deepl_ko style ###########################
        # Composed of single and multi-turn conversation between human and assistant

        # Each conversation consists of multiple human-assistant utterance pairs
        # , each delimited by the keys "human" and "gpt".

        # Also, in order to construct the input as an instruction-output pair
        # , if the final speaker is a human, the final utterance is ignored.
        #############################################################################
        full_conv = input["conversation"]
        converted_text = split_sharegpt(full_conv)
        user_prompt, full_prompt = convert_format(converted_text)

        return user_prompt, full_prompt

    elif "instruction" in input.keys() and "output" in input.keys():
        ######################### KULLM and KoAlpaca style ###########################
        # Composed of instruction, input, and output

        # No additional modification is required, only the format is modified and output
        #############################################################################
        user_prompt, full_prompt = convert_format(input)
        return user_prompt, full_prompt

    else:
        raise Exception(
            "This dataset is not currently supported. Currently supported datasets are [kullm, alpaca, guanaco-ko, sharegpt-deepl-ko]. \
            If you wish to use it, directly preprocess the dataset in the same format as one of these."
        )
