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
            model_path = args.model

        # QA format
        self.input_qa = {
            "input_with_context": "### 질문: {input}\n\n### 맥락: {context}\n\n### 답변:",
            "input_wo_context": "### 질문: {input}\n\n### 답변:",
        }

        model_kwargs = dict(device_map="auto")

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
                args.gptq_weights, device_map="auto", use_triton=True
            )
            self.model.eval()
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
            # Model Definition
            self.model = AutoModelForCausalLM.from_pretrained(
                **model_path, **model_kwargs
            )
            self.model.eval()

        # Tokneizer Definition
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(**model_path)
        except:
            # For using LLaMA-based-model
            self.tokenizer = AutoTokenizer.from_pretrained(
                **model_path, padding_side="right", use_fast=False, legacy=False
            )
        self.args = args

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


class LLM_for_eval(LLM):
    from accelerate import find_executable_batch_size
    from eval_utils import eval_utils
    import torch.nn.functional as F
    import torch
    from tqdm import tqdm
    from eval_utils.eval_utils import Reorderer
    from eval_utils.base import CacheHook

    def __init__(self, args) -> None:
        _DEFAULT_MAX_LENGTH = 2048

        super().__init__(args)
        self.batch_size = args.batch_size
        self.max_batch_size = args.max_batch_size
        self.max_length = _DEFAULT_MAX_LENGTH

        self.cache_hook = LLM_for_eval.CacheHook(None)

    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with LLM_for_eval.torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {
            "do_sample": False,
            "max_length": max_length,
            "device_map": "auto",
        }
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)

    ################################# Functions for Evaluation ################################
    # loglikelihood(): Compute log-likelihood of generating a continuation from a context.
    # greedy_until(): Generate greedily until a stopping sequence.
    # ask(): Generate model responses without streaming through a pipeline.
    # ask_with_streamer(): Generate model responses with streaming through a pipeline.
    #                      (used for gradio option)
    ##########################################################################################
    def _detect_batch_size(self, requests=None, pos=0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.args.max_length + 1) :][:-1]
            )
        else:
            max_length = self.args.max_length

        # if OOM, then halves batch_size and tries again
        @LLM_for_eval.find_executable_batch_size(
            starting_batch_size=self.args.max_batch_size
        )
        def forward_batch(batch_size):
            test_batch = LLM_for_eval.torch.ones(
                (batch_size, max_length), device="cuda"
            ).long()
            for _ in range(5):
                _ = LLM_for_eval.F.log_softmax(
                    self._model_call(test_batch), dim=-1
                ).cpu()
            return batch_size

        batch_size = forward_batch()

        return batch_size

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer.encode(context + continuation)
        context_enc = self.tokenizer.encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = LLM_for_eval.Reorderer(requests, _collate)

        reordered_requests = re_ord.get_reordered()
        n_reordered_requests = len(reordered_requests)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        def _batch_scheduler(pos):
            sched = pos // int(n_reordered_requests / self.batch_schedule)
            if sched in self.batch_sizes:
                return self.batch_sizes[sched]
            print(
                f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
            )
            self.batch_sizes[sched] = self._detect_batch_size(reordered_requests, pos)
            print(f"Determined largest batch size: {self.batch_sizes[sched]}")
            return self.batch_sizes[sched]

        for chunk in LLM_for_eval.eval_utils.chunks(
            LLM_for_eval.tqdm(reordered_requests, disable=disable_tqdm),
            n=self.args.batch_size
            if self.args.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0,
            fn=_batch_scheduler
            if self.args.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None,
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = LLM_for_eval.torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=LLM_for_eval.torch.long,
                ).to(
                    "cuda"
                )  # self.device to cuda
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )

                # pad length from seq to padding_length
                inp = LLM_for_eval.torch.cat(
                    [
                        inp,  # [seq]
                        LLM_for_eval.torch.zeros(
                            padding_length - inplen, dtype=LLM_for_eval.torch.long
                        ).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = LLM_for_eval.torch.cat(
                inps, dim=0
            )  # [batch, padding_length]
            multi_logits = LLM_for_eval.F.log_softmax(
                self._model_call(batched_inps), dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                inplen = inplen + (
                    logits.shape[0] - padding_length
                )  # if "virtual tokens" (from prompt tuning) are added, inplen is larger
                logits = logits[inplen - contlen : inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = LLM_for_eval.torch.tensor(
                    cont_toks, dtype=LLM_for_eval.torch.long
                ).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = LLM_for_eval.torch.gather(
                    logits, 2, cont_toks.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [
                    self.tokenizer.eot_token_id
                ], self.tokenizer.encode(continuation)
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def greedy_until(self, requests):
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        re_ord = LLM_for_eval.Reorderer(requests, _collate)

        warn_stop_seq = False
        for context, request_args in LLM_for_eval.tqdm(re_ord.get_reordered()):
            until = request_args["until"]
            if isinstance(until, str):
                until = [until]

            if until:
                try:
                    (primary_until,) = self.tokenizer.encode(until[0])
                except ValueError:
                    if not warn_stop_seq:
                        print(
                            "Warning: a primary stop sequence is multi-token! Will default to EOS token for this tokenizer. Consider using `hf-causal-experimental` for multi-token stop sequence support for the time being."
                        )
                        warn_stop_seq = True
                    primary_until = self.eot_token_id
            else:
                primary_until = None

            context_enc = LLM_for_eval.torch.tensor(
                [self.tokenizer.encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(
                "cuda"
            )  # self.device to cuda

            max_gen_tokens = min(
                self.max_gen_toks, request_args.get("max_length", self.max_gen_toks)
            )
            cont = self._model_generate(
                context_enc, context_enc.shape[1] + max_gen_tokens, primary_until
            )

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :])

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.append(s)

        return re_ord.get_original(res)
