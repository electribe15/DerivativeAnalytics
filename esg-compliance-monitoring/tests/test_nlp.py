import pytest
from src.nlp.data_preprocessing import preprocess_data
from src.nlp.ner import train_ner_model, evaluate_ner_model
from src.nlp.sentiment_analysis import analyze_sentiment

def test_preprocess_data():
    raw_data = "Sample ESG report text with some data."
    processed_data = preprocess_data(raw_data)
    assert processed_data is not None
    assert isinstance(processed_data, str)
    assert len(processed_data) > 0

def test_train_ner_model():
    training_data = [
        ("The company has a carbon disclosure policy.", {"entities": [(20, 27, "POLICY")]}),
        ("Supply chain ethics are crucial.", {"entities": [(0, 18, "ETHICS")]}),
    ]
    model = train_ner_model(training_data)
    assert model is not None

def test_evaluate_ner_model():
    test_data = [
        ("The company has a carbon disclosure policy.", {"entities": [(20, 27, "POLICY")]}),
    ]
    model = train_ner_model(test_data)
    evaluation_results = evaluate_ner_model(model, test_data)
    assert evaluation_results['accuracy'] >= 0.8

def test_analyze_sentiment():
    report_text = "The company's ESG performance is excellent."
    sentiment = analyze_sentiment(report_text)
    assert sentiment is not None
    assert sentiment in ["positive", "negative", "neutral"]