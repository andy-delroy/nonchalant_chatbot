"""
# app/ml_model.py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BitsAndBytesConfig


#the model is quantized to fit on the 6GB gpu
class NLPModel:
    def __init__(self, model_name: str):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="auto"
        )

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

# Use distilbert-base-uncased-distilled-squad which is specifically fine-tuned for QA
nlp_model = NLPModel("deepseek-ai/deepseek-r1-distill-qwen-7b")
"""

# # app/ml_model.py
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# import torch

# class NLPModel:
#     def __init__(self, model_name: str):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForQuestionAnswering.from_pretrained(
#             model_name,
#             device_map="cuda",  # Use GPU (RTX 3060)
#             torch_dtype=torch.float16  # Use FP16 for efficiency
#         )

#     def get_model(self):
#         return self.model

#     def get_tokenizer(self):
#         return self.tokenizer

# nlp_model = NLPModel("deepset/minilm-uncased-squad2")

# # app/ml_model.py
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# import torch

# class NLPModel:
#     def __init__(self, model_name: str):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         #this model is not suitable for QuestionAnswering
#         self.model = AutoModelForQuestionAnswering.from_pretrained(
#             model_name,
#             device_map="cuda",  # Use GPU (RTX 3060)
#             torch_dtype=torch.float16  # Use FP16 for efficiency
#         )

#     def get_model(self):
#         return self.model

#     def get_tokenizer(self):
#         return self.tokenizer

# nlp_model = NLPModel("dslim/bert-base-NER")

# app/ml_model.py
from transformers import pipeline

def load_ner_pipeline():
    return pipeline("ner", model="dslim/bert-base-NER", device=0)  # GPU 0