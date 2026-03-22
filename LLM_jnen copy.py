import os
import re

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ollama import Client, ResponseError

# ============================================================
# Environment
# ============================================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ============================================================
# Ollama settings
# ============================================================
DEFAULT_OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are a helpful local assistant. "
    "Answer clearly and accurately. "
    "For technical questions, be precise and structured."
)

# ============================================================
# Translation model settings
# ============================================================
JA_EN_MODEL = "Helsinki-NLP/opus-mt-ja-en"
EN_JA_MODEL = "Helsinki-NLP/opus-mt-en-jap"

# ============================================================
# Device helpers
# ============================================================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float16
    return torch.float32


DEVICE = get_device()
DTYPE = get_dtype(DEVICE)

# ============================================================
# Ollama client
# ============================================================
@st.cache_resource(show_spinner=False)
def get_ollama_client(host: str):
    return Client(host=host)

# ============================================================
# Language helpers
# ============================================================
def contains_japanese(text: str) -> bool:
    if not text:
        return False
    return re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text) is not None


def detect_lang(text: str) -> str:
    return "ja" if contains_japanese(text) else "en"

# ============================================================
# Translation loaders
# ============================================================
@st.cache_resource(show_spinner=False)
def load_translator(repo_id: str):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo_id,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_translators():
    ja_en_tokenizer, ja_en_model = load_translator(JA_EN_MODEL)
    en_ja_tokenizer, en_ja_model = load_translator(EN_JA_MODEL)

    return {
        "ja_en": (ja_en_tokenizer, ja_en_model),
        "en_ja": (en_ja_tokenizer, en_ja_model),
    }

# ============================================================
# Translation
# ============================================================
def translate_text(
    text: str,
    tokenizer,
    model,
    max_new_tokens: int = 256,
) -> str:
    if not text.strip():
        return text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def translate_ja_to_en(text: str) -> str:
    models = load_translators()
    tokenizer, model = models["ja_en"]
    return translate_text(text, tokenizer, model)


def translate_en_to_ja(text: str) -> str:
    models = load_translators()
    tokenizer, model = models["en_ja"]
    return translate_text(text, tokenizer, model)

# ============================================================
# Ollama chat
# ============================================================
def build_messages(english_history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(english_history)
    return messages


def ollama_chat_nonstream(
    english_history: list[dict],
    model_name: str,
    temperature: float,
    top_p: float,
    host: str,
) -> str:
    client = get_ollama_client(host)

    response = client.chat(
        model=model_name,
        messages=build_messages(english_history),
        options={
            "temperature": temperature,
            "top_p": top_p,
        },
    )
    return response.message.content.strip()


def ollama_chat_stream(
    english_history: list[dict],
    model_name: str,
    temperature: float,
    top_p: float,
    host: str,
):
    client = get_ollama_client(host)

    stream = client.chat(
        model=model_name,
        messages=build_messages(english_history),
        stream=True,
        options={
            "temperature": temperature,
            "top_p": top_p,
        },
    )

    for chunk in stream:
        text = chunk.message.content
        if text:
            yield text


def check_ollama_available(model_name: str, host: str):
    try:
        client = get_ollama_client(host)
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
        )
        return True, response.message.content[:200]
    except ResponseError as e:
        return False, f"Ollama response error: {e}"
    except Exception as e:
        return False, str(e)

# ============================================================
# End-to-end chat
# ============================================================
def build_output_text(answer_mode: str, llm_reply_en: str) -> str:
    if answer_mode == "English only":
        return llm_reply_en
    if answer_mode == "Japanese only":
        return translate_en_to_ja(llm_reply_en)

    reply_ja = translate_en_to_ja(llm_reply_en)
    return f"**English**\n\n{llm_reply_en}\n\n---\n\n**Japanese**\n\n{reply_ja}"

# ============================================================
# UI
# ============================================================
st.set_page_config(
    page_title="Local Chat with Ollama Python",
    page_icon="🦙",
    layout="wide",
)

st.title("🦙 Local Chat with Ollama Python")
st.caption(f"LLM: Ollama Python | Translation device: {DEVICE}")

with st.sidebar:
    st.header("Settings")

    ollama_host = st.text_input("Ollama host", value=OLLAMA_HOST)
    ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)

    answer_mode = st.radio(
        "Output language",
        ["English only", "Japanese only", "Show both"],
        index=2,
    )

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)

    use_stream = st.checkbox("Stream output", value=True)
    show_debug = st.checkbox("Show debug info", value=False)

    st.markdown("---")

    if st.button("Check Ollama connection"):
        ok, msg = check_ollama_available(ollama_model, ollama_host)
        if ok:
            st.success(f"Ollama is available. Sample reply: {msg}")
        else:
            st.error(msg)

    if st.button("Warm up translation models"):
        with st.status("Loading translation models..."):
            load_translators()
        st.success("Translation models loaded.")

    if st.button("Clear chat"):
        st.session_state.display_messages = []
        st.session_state.english_messages = []
        st.rerun()

# Session state
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "english_messages" not in st.session_state:
    st.session_state.english_messages = []

# Render history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_debug and "debug" in msg:
            dbg = msg["debug"]
            st.caption(
                f"detected={dbg['detected_language']} | "
                f"input_en={dbg['input_for_llm_en']} | "
                f"reply_en={dbg['llm_reply_en']}"
            )

# Input
prompt = st.chat_input("Type in Japanese or English")

if prompt:
    st.session_state.display_messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    user_lang = detect_lang(prompt)
    input_for_llm_en = translate_ja_to_en(prompt) if user_lang == "ja" else prompt

    working_history = st.session_state.english_messages + [
        {"role": "user", "content": input_for_llm_en}
    ]

    with st.chat_message("assistant"):
        try:
            if use_stream and answer_mode == "English only":
                collected_chunks = []

                def wrapped_stream():
                    for chunk in ollama_chat_stream(
                        english_history=working_history,
                        model_name=ollama_model,
                        temperature=temperature,
                        top_p=top_p,
                        host=ollama_host,
                    ):
                        collected_chunks.append(chunk)
                        yield chunk

                st.write_stream(wrapped_stream())
                llm_reply_en = "".join(collected_chunks).strip()
                final_reply = llm_reply_en
            else:
                with st.status("Generating response..."):
                    llm_reply_en = ollama_chat_nonstream(
                        english_history=working_history,
                        model_name=ollama_model,
                        temperature=temperature,
                        top_p=top_p,
                        host=ollama_host,
                    )
                    final_reply = build_output_text(answer_mode, llm_reply_en)
                    st.markdown(final_reply)

            st.session_state.english_messages = working_history + [
                {"role": "assistant", "content": llm_reply_en}
            ]

            debug_info = {
                "detected_language": user_lang,
                "input_for_llm_en": input_for_llm_en,
                "llm_reply_en": llm_reply_en,
            }

        except ResponseError as e:
            final_reply = f"Ollama error: {e}"
            debug_info = {
                "detected_language": user_lang,
                "input_for_llm_en": input_for_llm_en,
                "llm_reply_en": "",
            }
            st.error(final_reply)
        except Exception as e:
            final_reply = f"Error: {e}"
            debug_info = {
                "detected_language": user_lang,
                "input_for_llm_en": input_for_llm_en,
                "llm_reply_en": "",
            }
            st.error(final_reply)

    st.session_state.display_messages.append(
        {
            "role": "assistant",
            "content": final_reply,
            "debug": debug_info,
        }
    )