import argparse
import utils


# Text coloring: Yellow for user, and green for assistant
def input_qa():
    text = input("\033[33m" + "Question: ")
    print("\033[0m")
    return text


def print_qa(text: str):
    return print(f"\033[32mAnswer: {text}\033[0m\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="choose one model from [polygolot-ko, ko-alpaca, kullm, korani-v3] or use saved path",
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

    args = parser.parse_args()

    qa_pipe = utils.get_pipe(args)

    # Not use gradio: Use the CMD terminal for chatting
    if not args.use_gradio:
        input_text = input_qa()

        while True:
            if input_text != "대화를 종료합니다.":
                print_qa(utils.ask(args, qa_pipe, input_text))
                input_text = input_qa()
            else:
                print_qa("대화를 종료합니다...")
                break

    else:
        utils.start_gradio(args, qa_pipe)
