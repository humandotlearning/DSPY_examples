import dspy
from dotenv import load_dotenv
import os
load_dotenv(".env.local")

lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

from typing import Literal

#Example: Classification

class Emotion(dspy.Signature):
    """Classify emotion."""

    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()
    sentiment_reasoning: str = dspy.OutputField(desc="reasoning for the sentiment")

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion)
answer = classify(sentence=sentence)
print(answer.sentiment)
print(answer.sentiment_reasoning)
print("\n\n")
print(lm.history)

print( lm.history[-1].keys())
print(lm.history[-1]["response"])
