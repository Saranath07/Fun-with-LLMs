import dspy
from dotenv import load_dotenv


load_dotenv()


def load_llm():

    llm = dspy.LM(model="together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo")
    
    return llm
