from transformers import AutoModelForQuestionAnswering
model_name = "deepseek-ai/deepseek-r1-distill-qwen-7b"
try:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("Model loaded!")
except Exception as e:
    print(f"Oops: {e}")