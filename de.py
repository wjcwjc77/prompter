from transformers import AutoTokenizer, AutoModelForCausalLM

DeepSeek_V3_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
print(DeepSeek_V3_tokenizer.decode([1]))

# print(tokenizer.encode("https://github.com/wjcwjc77/rezinekotkiT"))

tiktoken_cl100k_base_tokenizer = AutoTokenizer.from_pretrained("DWDMaiMai/tiktoken_cl100k_base")
print(tiktoken_cl100k_base_tokenizer.decode([1]))