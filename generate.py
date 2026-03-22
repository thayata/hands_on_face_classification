
import streamlit as st
from translate import Translator
import torch
from diffusers import StableDiffusionPipeline

# Text-to-Image generation function using Stable Diffusion
def load_text_to_image_model():

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype
        )

    pipe = pipe.to(device)
    return pipe

def generate_image_from_text(prompt, num_inference_steps=50, guidance_scale=7.5):
    """Generate an image from text prompt"""
    pipe = load_text_to_image_model()
    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
    return image

def text_to_image_ui():

    st.header("画像生成 (Text-to-Image)")
    
    # Input text prompt (Japanese)
    japanese_prompt = st.text_area("画像の説明を日本語で入力してください:", 
                                       value="慶應医学部",
                                       height=100)

    with st.spinner("日本語を英語に翻訳中..."):
        translator = Translator(from_lang = "ja", to_lang = "en")
        text_prompt = translator.translate(japanese_prompt)
        st.info(f"英語訳: {text_prompt}")
    
    col1, col2 = st.columns(2)
    with col1:
        num_steps = st.slider("推論ステップ数:", min_value=10, max_value=100, value=50)
    with col2:
        guidance = st.slider("ガイダンススケール:", min_value=1.0, max_value=20.0, value=7.5)
    
    if st.button("画像を生成"):
        with st.spinner("画像を生成中..."):
            generated_image = generate_image_from_text(text_prompt, num_steps, guidance)
            st.image(generated_image, caption="生成された画像", use_container_width=True)


# Main UI
def main():
    # Set page config
    st.set_page_config(page_title="画像生成アプリ", layout="wide")
    st.title("AI画像生成アプリケーション")
    st.write("Stable Diffusionを使用してテキストから画像を生成します")
    
    # Display the text-to-image UI
    text_to_image_ui()

if __name__ == "__main__":
    main()
