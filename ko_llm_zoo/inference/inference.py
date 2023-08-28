import argparse
import sys, os

sys.path.append(os.path.abspath(os.path.join("ko_llm_zoo", "..")))
from ko_llm_zoo.utils.get_model import LLM
import gradio as gr


# Text coloring: Yellow for user, and green for assistant
def input_qa():
    text = input("\033[33m" + "Question: ")
    print("\033[0m")
    return text


def print_qa(text: str, stream=False):
    if not stream:
        print(f"\033[32mAnswer: {text}\033[0m\n\n", end="")
    else:
        print("\033[32mAnswer: ")
        input_qa = llm.formating_input_with_template(text)
        inputs = llm.tokenizer(
            [input_qa], return_tensors="pt", return_token_type_ids=False
        ).to("cuda")
        _ = llm.model.generate(**inputs, **llm.generation_kwargs)
        print("\033[0m\n\n")


def gradio(llm: LLM, stream: bool):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
                # Demo-KoLLM
            """
        )

        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="메세지를 입력하세요")
        clear = gr.ClearButton([msg, chatbot])

        def user(user_message, history):
            return gr.update(value="", interactive=False), history + [
                [user_message, None]
            ]

        def respond(message, chat_history):
            bot_message = llm.ask(message)

            chat_history.append((message, bot_message))
            return "", chat_history

        def ask_with_streamer_in_gradio(history):
            input = history[-1][0]
            history[-1][1] = ""
            for new_text in llm.ask_with_streamer(input):
                history[-1][1] += new_text
                yield history

        if stream:
            response = msg.submit(
                user, [msg, chatbot], [msg, chatbot], queue=False
            ).then(ask_with_streamer_in_gradio, chatbot, chatbot)
            response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
        else:
            msg.submit(respond, [msg, chatbot], [msg, chatbot])

    demo.queue()
    demo.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="choose one model from [polygolot-ko, ko-alpaca, kullm, korani-v3, kovicuna, kogpt] or use saved path",
    )

    parser.add_argument(
        "--max_new_token",
        type=int,
        default=512,
        help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="A value used to modulate the next token probabilities. Higher values increase randomness.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.7,
        help="A value that controls the determinism with which the model generates responses. Higher values increase the diversity of responses.",
    )

    parser.add_argument(
        "--use_gradio",
        action="store_true",
        help="Use gradio for chat UI",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming for chat",
    )
    parser.add_argument(
        "--use_gpu",
        type=str,
        default="0",
        help="The number of GPUs to use. If you want to use 0 and 1, enter '0, 1'",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        choices=["gptq", "int8", "fp4", "nf4", "nf4-dq", "nf4-dq-comp_bf16"],
        help="Chosse quantization method. Note that the 'gptq' option must be preceded by quantization.py and use the stored weights",
    )

    parser.add_argument(
        "--gptq_weights",
        type=str,
        default=None,
        help="The path where the model weight quantized via GPTQ is stored. If not specified, the gptq model will not be available.",
    )

    args = parser.parse_args()
    print(args)

    llm = LLM(args)

    qa_pipe = llm.get_pipe()

    # Not use gradio: Use the CMD terminal for chatting
    if not args.use_gradio:
        try:
            input_text = input_qa()

            while True:
                if input_text != "대화를 종료합니다.":
                    if not args.stream:
                        print_qa(llm.ask(input_text))
                    else:
                        print_qa(input_text, stream=True)
                    input_text = input_qa()
                else:
                    print_qa("대화를 종료합니다...")
                    break
        # To prevent the problem that text coloring is maintained even after process termination
        except KeyboardInterrupt:
            print("\033[0m\nKeyboardInterrupt")

    else:
        gradio(llm, stream=args.stream)
