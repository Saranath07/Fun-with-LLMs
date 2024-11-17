from dotenv import load_dotenv
import dspy
import os

load_dotenv()  # This will load variables from .env into environment

api_key = os.getenv('TOGETHER_API_KEY')

llm = dspy.LM(model="together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
dspy.configure(lm=llm)

print(llm("Hello! How are you doing?"))