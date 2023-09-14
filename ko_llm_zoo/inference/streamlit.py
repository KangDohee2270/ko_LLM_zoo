import sys, os

sys.path.append(os.path.abspath(os.path.join("ko_llm_zoo", "..")))
from ko_llm_zoo.utils.get_model import LLM

import streamlit as st

import argparse


@st.cache_resource
def parse_args():
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
    args.mode = "inference"
    args.use_interface = True
    return args


@st.cache_resource()
def get_pipe(_args):
    llm = LLM(_args)
    llm.get_pipe()

    return llm


if __name__ == "__main__":
    args = parse_args()
    st.subheader("Demo with Ko-LLM-zoo")
    st.markdown(f"selected model: {args.model}")

    llm = get_pipe(args)

    # # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("메세지를 입력하세요"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if not args.stream:
                response = llm.ask(prompt)
            else:
                response = ""
                for chunk in llm.ask_with_streamer(prompt):
                    response += chunk + " "
                    message_placeholder.markdown(response + "▌")

        message_placeholder.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # print(st.session_state.messages)
