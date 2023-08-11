import argparse
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="selected model",
        choices=[
            "polyglot_ko",
            "ko_alpaca",
            "kullm",
            "korani-v3",
        ],
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
        help="A value that controls the determinism with which the model generates responses. Higher values increase the diversity of responses.",
    )

    args = parser.parse_args()

    qa_pipe = utils.get_pipe(args)
    input_text = input("Question: ")

    while True:
        if input_text != "exit":
            print("Answer: {}".format(utils.ask(args, qa_pipe, input_text)))
            input_text = input("Question: ")
        else:
            print("Stop QA...")
            break
