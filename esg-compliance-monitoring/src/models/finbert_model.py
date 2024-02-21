from transformers import BertTokenizer, BertForSequenceClassification
import torch

class FinBERTModel:
    def __init__(self, model_name='yiyanghkust/finbert-tone'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        sentiment = torch.argmax(probabilities, dim=1).item()
        return sentiment, probabilities.numpy()

    def load_model(self):
        return self.model

    def save_model(self, save_directory):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)