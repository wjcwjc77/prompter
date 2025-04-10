from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
print(tokenizer.encode("https://github.com/wjcwjc77/rezinekotkiT"))