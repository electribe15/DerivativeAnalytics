from openai import OpenAI

class GPT4Integration:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = OpenAI(api_key=self.api_key)

    def summarize_text(self, text):
        response = self.model.Completion.create(
            engine="gpt-4",
            prompt=f"Please summarize the following text:\n{text}",
            max_tokens=150
        )
        return response.choices[0].text.strip()

    def answer_question(self, question, context):
        response = self.model.Completion.create(
            engine="gpt-4",
            prompt=f"Based on the following context, answer the question:\nContext: {context}\nQuestion: {question}",
            max_tokens=150
        )
        return response.choices[0].text.strip()