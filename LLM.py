import streamlit as st
from ollama import Client, ResponseError

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
SYSTEM_PROMPT = "You are a helpful assistant."

@st.cache_resource
def get_client(host: str):
    return Client(host=host)

def stream_chat(host: str, model: str, messages: list[dict]):
    client = get_client(host)
    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        text = chunk.message.content
        if text:
            yield text

st.set_page_config(page_title="Ollama Chat", page_icon="🦙", layout="wide")
st.title("🦙 Ollama Chat")

with st.sidebar:
    ollama_host = st.text_input("Ollama host", value=OLLAMA_HOST)
    model_name = st.text_input("Model", value=DEFAULT_MODEL)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Type your message")

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chunks = []
        placeholder = st.empty()

        try:
            for piece in stream_chat(
                host=ollama_host,
                model=model_name,
                messages=st.session_state.messages,
            ):
                chunks.append(piece)
                placeholder.markdown("".join(chunks))

            answer = "".join(chunks).strip()
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except ResponseError as e:
            err = f"Ollama error: {e}"
            placeholder.error(err)
        except Exception as e:
            err = f"Error: {e}"
            placeholder.error(err)