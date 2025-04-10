import gradio as gr
from transformers import AutoTokenizer
from hg_hub import api
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def get_models(search_query="tokenizer", token=None):
    models = api.list_models(search=search_query, limit=50, token=token)
    logger.info(f"models: {models}")
    return [m.id for m in models if not m.gated]

# 创建交互界面
def create_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            search_input = gr.Textbox(label="Search Models", placeholder="Type to search tokenizers...")
        with gr.Row():
            model_dropdown = gr.Dropdown(label="Select Tokenizer", choices=get_models(), value="deepseek-ai/DeepSeek-V3")
        input_text = gr.Textbox(label="Token IDs", placeholder="Enter token list like 1,2,3...")
        output_text = gr.Textbox(label="Decoded Text", interactive=False)
        hf_token = gr.Textbox(label="HuggingFace Token", type="password", placeholder="hf_...")

        def tokenize(text, model_name):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
                # 转换输入为整数列表
                try:
                    text = text.strip()
                    if not text:
                        return "Error: 输入不能为空"
                    parts = text.split(',')
                    token_ids = []
                    for part in parts:
                        part = part.strip()
                        if part:
                            token_ids.append(int(part))
                    decoded = tokenizer.decode(token_ids)
                    return decoded
                except Exception as e:
                    return f"解析错误: {str(e)}"
            except Exception as e:
                return f"Error: {str(e)}"

        input_text.change(tokenize, [input_text, model_dropdown], output_text)
        model_dropdown.change(tokenize, [input_text, model_dropdown], output_text)

        def update_model_dropdown(search_query, hf_token):
            return gr.update(choices=get_models(search_query, hf_token if hf_token else None))
        
        search_input.change(update_model_dropdown, inputs=[search_input, hf_token], outputs=model_dropdown)

    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False)