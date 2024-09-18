import dspy
from dotenv import load_dotenv


load_dotenv()


def load_llm():

    llm = dspy.Together(model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', max_tokens=2500)
    
    return llm
