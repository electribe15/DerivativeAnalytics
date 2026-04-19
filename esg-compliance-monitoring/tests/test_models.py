import pytest
from src.models.finbert_model import FinBERTModel
from src.models.esgbert_model import ESGBERTModel
from src.models.gpt4_integration import GPT4Integration

@pytest.fixture
def finbert_model():
    model = FinBERTModel()
    model.load_model()
    return model

@pytest.fixture
def esgbert_model():
    model = ESGBERTModel()
    model.load_model()
    return model

@pytest.fixture
def gpt4_model():
    model = GPT4Integration()
    model.load_model()
    return model

def test_finbert_sentiment_analysis(finbert_model):
    text = "The company's ESG performance is excellent."
    sentiment = finbert_model.analyze_sentiment(text)
    assert sentiment in ['positive', 'negative', 'neutral']

def test_esgbert_compliance_insight(esgbert_model):
    text = "The company has made significant progress in reducing carbon emissions."
    insights = esgbert_model.get_compliance_insights(text)
    assert isinstance(insights, dict)

def test_gpt4_summary(gpt4_model):
    text = "The ESG report outlines the company's commitment to sustainability."
    summary = gpt4_model.summarize(text)
    assert len(summary) > 0

def test_gpt4_question_answering(gpt4_model):
    question = "What are the key sustainability initiatives?"
    context = "The company has implemented several initiatives including renewable energy usage."
    answer = gpt4_model.answer_question(question, context)
    assert "renewable energy" in answer.lower()