import dspy
from dotenv import load_dotenv


load_dotenv()


def load_llm():

    llm = dspy.Together(model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

    dspy.settings.configure(lm=llm, rm=colbertv2_wiki17_abstracts)
    return llm