from threading import Thread
import os
from typing import List

model_path_list = {
    "polyglot-ko": {"pretrained_model_name_or_path": "EleutherAI/polyglot-ko-12.8b"},
    "ko-alpaca": {"pretrained_model_name_or_path": "beomi/KoAlpaca-Polyglot-12.8B"},
    "kullm": {"pretrained_model_name_or_path": "nlpai-lab/kullm-polyglot-12.8b-v2"},
    "korani-v3": {"pretrained_model_name_or_path": "KRAFTON/KORani-v3-13B"},
    "kovicuna": {"pretrained_model_name_or_path": "junelee/ko_vicuna_7b"},
    "kogpt": {
        "pretrained_model_name_or_path": "kakaobrain/kogpt",
        "revision": "KoGPT6B-ryan1.5b-float16",
    },
}


class LLM:
    def __init__(self, args) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu
        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        if args.model in model_path_list.keys():
            model_path = model_path_list[args.model]
        elif not os.path.exists(args.model):
            raise FileNotFoundError(
                "The model path is invalid, make sure you are providing the correct path where the model weights are located"
            )
        else:
            model_path = {"pretrained_model_name_or_path": args.model}

        # Tokneizer Definition
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(**model_path)
        except:
            # For using LLaMA-based-model
            self.tokenizer = AutoTokenizer.from_pretrained(
                **model_path, padding_side="right", use_fast=False, legacy=False
            )
        self.args = args

        model_kwargs = dict(device_map="auto")

        # for evaluation - Set to the default model settings.
        if "mode" not in self.args:
            model_kwargs["torch_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                **model_path, **model_kwargs
            )

            return

        #################
        # LoRA & QLoRA: Methods for efficient finetuning

        # Code Reference
        # https://github.com/huggingface/peft
        # https://github.com/nlpai-lab/KULLM
        #################
        if args.mode == "finetuning":
            from peft import (
                LoraConfig,
                prepare_model_for_kbit_training,
                get_peft_model,
            )

            if args.finetuning_method not in ["lora", "qlora"]:
                raise Exception(
                    "Unknown finetuning method. You must choose one of [lora, qlora]"
                )

            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if args.finetuning_method == "lora":
                model_kwargs["load_in_8bit"] = True

            elif args.finetuning_method == "qlora":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config

            self.model = AutoModelForCausalLM.from_pretrained(
                **model_path, **model_kwargs
            )
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, config)
            self.model.config.use_cache = False
            if torch.cuda.device_count() > 1:
                # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
                self.model.is_parallelizable = True
                self.model.model_parallel = True

            return

        #################
        # Quantization methods for efficient inference

        # Code Reference
        # https://huggingface.co/docs/transformers/perf_infer_gpu_one
        # https://github.com/PanQiWei/AutoGPTQ/blob/main/examples/quantization/quant_with_alpaca.py
        #################

        if args.quant == "gptq":
            if args.gptq_weights == None or not os.path.exists(args.gptq_weights):
                raise FileNotFoundError(
                    "Quantized weights via gptq are not found. If you want to use the gptq option, you need to import weights via '--gptq_weights [weights_path]'."
                )
            from auto_gptq import AutoGPTQForCausalLM

            self.model = AutoGPTQForCausalLM.from_quantized(
                args.gptq_weights,
                device_map="auto",
                use_triton=True,
                inject_fused_mlp=True,
                inject_fused_attention=True,
            )
        else:
            if args.quant != None:
                if args.quant == "int8":
                    model_kwargs["load_in_8bit"] = True
                elif args.quant == "fp4":
                    model_kwargs["load_in_4bit"] = True
                else:  # use nf4
                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",  # nf4
                        bnb_4bit_use_double_quant=(
                            True if args.quant == "nf4-dq" else False
                        ),
                        bnb_4bit_compute_dtype=(
                            torch.bfloat16
                            if args.quant == "nf4-dq-comp_bf16"
                            else torch.float32
                        ),
                    )
                    model_kwargs["quantization_config"] = nf4_config
            else:
                model_kwargs["torch_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                **model_path, **model_kwargs
            )

        # QA format
        self.input_qa = {
            "input_with_context": "### 질문: {input}\n\n### 맥락: {context}\n\n### 답변:",
            "input_wo_context": "### 질문: {input}\n\n### 답변:",
        }

        self.model.eval()

    ################################# Functions for Inference ################################
    # get_pipe(): Define a pipeline for text generation
    # formating_input_with_template(): Transform the input (instruction) to match the template.
    # ask(): Generate model responses without streaming through a pipeline.
    # ask_with_streamer(): Generate model responses with streaming through a pipeline.
    #                      (used for gradio option)
    ##########################################################################################
    def get_pipe(self):
        from transformers import pipeline, TextIteratorStreamer, TextStreamer

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
            if self.args.use_interface:
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
