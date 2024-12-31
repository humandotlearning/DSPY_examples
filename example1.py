import dspy
from dotenv import load_dotenv
import os
load_dotenv(".env.local")

lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)


qa = dspy.ChainOfThought("question -> answer: int")

response = qa(question="How many floors are in the castle of Versailles?")
print(response)
print(response.answer)

print(lm.history)

print( lm.history[-1].keys())
print(lm.history[-1]["response"])
