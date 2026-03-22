import re
import streamlit as st
from translate import Translator
from ollama import Client, ResponseError

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
SYSTEM_PROMPT = "You are a helpful assistant."

@st.cache_resource
def get_client(host: str):
    return Client(host=host)

def contains_japanese(text: str) -> bool:
    if not text:
        return False
    return re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text) is not None

def detect_lang(text: str) -> str:
    return "ja" if contains_japanese(text) else "en"

def translate_ja_to_en(text: str) -> str:
    translator = Translator(from_lang="ja", to_lang="en")
    return translator.translate(text)

def translate_en_to_ja(text: str) -> str:
    translator = Translator(from_lang="en", to_lang="ja")
    return translator.translate(text)

def build_messages(english_history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(english_history)
    return messages

def ollama_chat_nonstream(
    english_history: list[dict],
    model_name: str,
    host: str,
) -> str:
    client = get_client(host)
    response = client.chat(
        model=model_name,
        messages=build_messages(english_history),
    )
    return response.message.content.strip()

def ollama_chat_stream(
    english_history: list[dict],
    model_name: str,
    host: str,
):
    client = get_client(host)
    stream = client.chat(
        model=model_name,
        messages=build_messages(english_history),
        stream=True,
    )
    for chunk in stream:
        text = chunk.message.content
        if text:
            yield text

def build_output_text(answer_mode: str, llm_reply_en: str) -> str:
    if answer_mode == "English only":
        return llm_reply_en
    if answer_mode == "Japanese only":
        return translate_en_to_ja(llm_reply_en)

    reply_ja = translate_en_to_ja(llm_reply_en)
    return f"**English**\n\n{llm_reply_en}\n\n---\n\n**Japanese**\n\n{reply_ja}"

st.set_page_config(page_title="Ollama Chat", page_icon="🦙", layout="wide")
st.title("🦙 Ollama Chat with JA/EN Translation")

with st.sidebar:
    ollama_host = st.text_input("Ollama host", value=OLLAMA_HOST)
    model_name = st.text_input("Model", value=DEFAULT_MODEL)

    answer_mode = st.radio(
        "Output language",
        ["English only", "Japanese only", "Show both"],
        index=2,
    )

    use_stream = st.checkbox("Stream output", value=False)

    if st.button("Clear chat"):
        st.session_state.display_messages = []
        st.session_state.english_messages = []
        st.rerun()

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "english_messages" not in st.session_state:
    st.session_state.english_messages = []

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("日本語または英語で入力してください")

if prompt:
    st.session_state.display_messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    user_lang = detect_lang(prompt)

    with st.spinner("翻訳中..."):
        if user_lang == "ja":
            input_for_llm_en = translate_ja_to_en(prompt)
        else:
            input_for_llm_en = prompt

    st.info(f"LLM input (EN): {input_for_llm_en}")

    working_history = st.session_state.english_messages + [
        {"role": "user", "content": input_for_llm_en}
    ]

    with st.chat_message("assistant"):
        try:
            if use_stream and answer_mode == "English only":
                chunks = []

                def wrapped_stream():
                    for piece in ollama_chat_stream(
                        english_history=working_history,
                        model_name=model_name,
                        host=ollama_host,
                    ):
                        chunks.append(piece)
                        yield piece

                st.write_stream(wrapped_stream())
                llm_reply_en = "".join(chunks).strip()
                final_reply = llm_reply_en
            else:
                with st.spinner("Ollama generating..."):
                    llm_reply_en = ollama_chat_nonstream(
                        english_history=working_history,
                        model_name=model_name,
                        host=ollama_host,
                    )
                final_reply = build_output_text(answer_mode, llm_reply_en)
                st.markdown(final_reply)

            st.session_state.english_messages = working_history + [
                {"role": "assistant", "content": llm_reply_en}
            ]

        except ResponseError as e:
            final_reply = f"Ollama error: {e}"
            st.error(final_reply)
        except Exception as e:
            final_reply = f"Error: {e}"
            st.error(final_reply)

    st.session_state.display_messages.append(
        {"role": "assistant", "content": final_reply}
    )