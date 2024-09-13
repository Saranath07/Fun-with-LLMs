from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


model = ChatGroq(
        model="llama-3.1-70b-versatile",
        max_tokens=8000
    )

prompt = PromptTemplate(
    input_variables=["input"],
    template="Tell me about {input}?"
)


chain = (
    {"input": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)

output = chain.invoke("icecream")

print(output)