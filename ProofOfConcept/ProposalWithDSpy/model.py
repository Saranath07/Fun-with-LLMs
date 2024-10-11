import dspy
from dotenv import load_dotenv


load_dotenv()


def load_llm():

    llm = dspy.LM(model="together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
    
    return llm
