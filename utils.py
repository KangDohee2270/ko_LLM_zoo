from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import transformers

model_path_list = {
    "polyglot-ko": "EleutherAI/polyglot-ko-12.8b",
    "ko-alpaca": "beomi/KoAlpaca-Polyglot-12.8B",
    "kullm": "nlpai-lab/kullm-polyglot-12.8b-v2",
    "korani-v3": "KRAFTON/KORani-v3-13B",
}

class LLM_Model():
    def __init__(self):






def get_model_and_tokenizer(args):
    ############### Finetuned models with Korean ###############
    if args.model in model_path_list.keys():
        model_path = model_path_list[args.model]
    elif not os.path.exists(args.model):
        raise FileNotFoundError(
            "The model path is invalid, make sure you are providing the correct path where the model weights are located"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        # For using LLaMA-based-model
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="right", use_fast=False, legacy=False
        )
    args.model, args.tokenizer = model, tokenizer
    return model, tokenizer


def get_pipe(args):
    model, tokenizer = get_model_and_tokenizer(args)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )
    args.generation_kwargs = dict(
        max_new_tokens=args.max_new_token,
        temperature=args.temp,
        top_p=args.top_p,
        return_full_text=False,
        pad_token_id=args.tokenizer.pad_token_id,
        eos_token_id=args.tokenizer.eos_token_id,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return pipe


def ask(args, pipe: transformers.Pipeline, input: str, context: str = "") -> str:
    input_qa = (
        f"### 질문: {input}\n\n### 맥락: {context}\n\n### 답변:"
        if context
        else f"### 질문: {input}\n\n### 답변:"
    )

    ans = pipe(input_qa, **args.generation_kwargs)

    return ans[0]["generated_text"].split("###")[0]


def start_gradio(args, pipe: transformers.Pipeline):
    import gradio as gr

    def ask(input, context=""):
        input_qa = (
            f"### 질문: {input}\n\n### 맥락: {context}\n\n### 답변:"
            if context
            else f"### 질문: {input}\n\n### 답변:"
        )

        ans = pipe(
            input_qa,
            **args.generation_kwargs
            # num_return_sequences=1,
            # repetition_penalty=1.2,
            # bad_words_ids=[
            #     args.llm_tokenizer.encode(bad_word) for bad_word in bad_words
            # ],
        )
        return ans[0]["generated_text"]

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Demo-KoLLM
            """
        )

        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="메세지를 입력하세요")
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            bot_message = ask(message)

            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    demo.launch(share=True)
