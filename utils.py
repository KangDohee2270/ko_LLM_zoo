from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
    TextStreamer,
)
from threading import Thread
import torch
import os
import transformers
from typing import List

model_path_list = {
    "polyglot-ko": "EleutherAI/polyglot-ko-12.8b",
    "ko-alpaca": "beomi/KoAlpaca-Polyglot-12.8B",
    "kullm": "nlpai-lab/kullm-polyglot-12.8b-v2",
    "korani-v3": "KRAFTON/KORani-v3-13B",
}


class LLM:
    def __init__(self, args) -> None:
        if args.model in model_path_list.keys():
            model_path = model_path_list[args.model]
        elif not os.path.exists(args.model):
            raise FileNotFoundError(
                "The model path is invalid, make sure you are providing the correct path where the model weights are located"
            )
        else:
            model_path = args.model

        # QA format
        self.input_qa = {
            "input_with_context": "### 질문: {input}\n\n### 맥락: {context}\n\n### 답변:",
            "input_wo_context": "### 질문: {input}\n\n### 답변:",
        }

        # Model Definition
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

        # Tokneizer Definition
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            # For using LLaMA-based-model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="right", use_fast=False, legacy=False
            )
        self.args = args

    def get_pipe(self):
        generation_kwargs = dict(
            max_new_tokens=self.args.max_new_token,
            temperature=self.args.temp,
            top_p=self.args.top_p,
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            repetition_penalty=1.1,
        )
        if self.args.stream:
            if self.args.use_gradio:
                self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            else:
                self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            generation_kwargs["streamer"] = self.streamer
            del generation_kwargs["return_full_text"]

            self.generation_kwargs = generation_kwargs

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            **generation_kwargs,
            device_map="auto",
        )

    def formating_input_with_template(self, input: str, context: str = ""):
        input_qa = (
            self.input_qa["input_with_context"].format(input=input, context=context)
            if context
            else self.input_qa["input_wo_context"].format(input=input)
        )

        return input_qa

    def ask(self, input: str, context: str = "") -> str:
        input_qa = self.formating_input_with_template(input, context)
        ans = self.pipe(input_qa)

        return ans[0]["generated_text"].split("###")[0]

    def ask_with_streamer(self, input: str):
        input_qa = self.formating_input_with_template(input)
        inputs = self.tokenizer(
            [input_qa], return_tensors="pt", return_token_type_ids=False
        ).to("cuda")

        self.generation_kwargs["input_ids"] = inputs.input_ids
        self.thread = Thread(target=self.model.generate, kwargs=self.generation_kwargs)
        self.thread.start()

        for new_text in self.streamer:
            # print(new_text)
            if "<|endoftext|>" in new_text:
                new_text = new_text.rstrip("<|endoftext|>")

            yield new_text
            # time.sleep(0.5)
            # print(history)
            # yield history
