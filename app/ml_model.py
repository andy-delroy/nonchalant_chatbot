# app/ml_model.py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class NLPModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

# Use distilbert-base-uncased-distilled-squad which is specifically fine-tuned for QA
nlp_model = NLPModel("distilbert-base-cased-distilled-squad")