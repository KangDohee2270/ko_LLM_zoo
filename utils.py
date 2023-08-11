from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import transformers

model_path_list = {
    "polyglot_ko": "EleutherAI/polyglot-ko-12.8b",
    "ko_alpaca": "beomi/KoAlpaca-Polyglot-12.8B",
    "kullm": "nlpai-lab/kullm-polyglot-12.8b-v2",
    "korani-v3": "KRAFTON/KORani-v3-13B",
}


def get_model_and_tokenizer(args):
    ############### Finetuned models with Korean ###############
    model_path = model_path_list[args.model]
    if args.model == "polyglot_ko":
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    args.llm_model, args.llm_tokenizer = model, tokenizer
    return model, tokenizer


def get_pipe(args):
    model, tokenizer = get_model_and_tokenizer(args)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="balanced",
    )

    return pipe


def ask(args, pipe: transformers.Pipeline, input: str, context: str = "") -> str:
    if args.model != "kullm" and args.model != "polyglot_ko":
        input_qa = (
            f"### 질문: {input}\n\n### 맥락: {context}\n\n### 답변:"
            if context
            else f"### 질문: {input}\n\n### 답변:"
        )
    else:
        input_qa = (
            f"아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{input}\n\n### 입력:\n{context}\n\n### 응답:\n"
            if context
            else f"아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{input}\n\n### 응답:\n"
        )
    # print(input_qa)
    ans = pipe(
        input_qa,
        max_new_tokens=args.max_new_token,
        temperature=args.temp,
        top_p=args.top_p,
        return_full_text=False,
        pad_token_id=args.llm_tokenizer.pad_token_id,
        eos_token_id=args.llm_tokenizer.eos_token_id,
        do_sample=True,
        repetition_penalty=1.1,
    )

    return ans[0]["generated_text"]
