from deepseek_tokenizer import ds_token

# Sample text
text = "https://github.com/wjcwjc77/rezinekotkiT"

# Encode text
token = ds_token.encode(text)
ori = [ds_token.decode(i) for i in [2485, 1129, 5316, 916, 6458, 73, 63643, 64087, 2813, 10991, 89, 483, 74, 354, 6780, 51] ]

# Print result
print(token,ori)