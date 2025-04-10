from huggingface_hub import HfApi
api = HfApi()
# models = api.list_models()
models = api.list_models(
    search="deepseek",
    limit=50,
)
models = [m for m in models if not m.gated]
# print("\n\n".join([f"{m.id}" for i,m in enumerate(models) if i < 10]))

#