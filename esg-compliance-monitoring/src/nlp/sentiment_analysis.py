from transformers import BertTokenizer, BertForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self, model_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)

    def analyze_sentiment(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        
        return "positive" if predicted_class == 1 else "negative"

def main():
    model_path = "path/to/finbert/model"  # Update with the actual model path
    analyzer = SentimentAnalyzer(model_path)
    
    sample_text = "The company's ESG report shows significant improvement in sustainability practices."
    sentiment = analyzer.analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()