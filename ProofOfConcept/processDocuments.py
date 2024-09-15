from langchain_groq import ChatGroq
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    return ChatGroq(
        model="llama-3.1-70b-versatile",
        max_tokens=8000
    )

def process_documents(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    
    db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return db

def generate_proposal(llm, db, client_requirements):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    
    prompt = PromptTemplate(
        input_variables=["client_requirements"],
        template="""
        Generate a detailed and professional proposal based on the following client requirements:
        {client_requirements}

        Your proposal should include:
        1. An executive summary
        2. A detailed analysis of the client's needs
        3. Your proposed solution, including technical specifications
        4. A feasibility study and risk analysis
        5. Timeline and milestones
        6. Pricing and payment terms
        7. Next steps

        Use the information from the technical documents to support your proposal.
        Ensure the proposal is tailored to the specific client requirements and industry standards.
        DO NOT ADD ANYTHING ELSE TO THE PROPOSAL. LIKE Here is.. etc.
        """
    )
    
    chain = (
    {"client_requirements": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)
    proposal = chain.invoke({"client_requirements": client_requirements})
   
    
    analysis_prompt = f"""Based on the client requirements and the generated proposal, what are the key risks and how can they be mitigated?
    client requirements: {client_requirements}
    proposal: {proposal}"""
    risk_analysis = qa.invoke({"query": analysis_prompt, "client_requirements": client_requirements, "proposal": proposal})
    risk_analysis_text = risk_analysis['result']
    
    return proposal, risk_analysis_text

# process_documents("documents")
# print("Documents processed and stored in the database.")
    