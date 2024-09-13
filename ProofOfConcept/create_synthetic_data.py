from langchain_groq import ChatGroq
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
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
    
    # Use Chroma instead of FAISS
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
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    proposal = chain.run(client_requirements=client_requirements)
    
    # Perform a follow-up analysis using the QA system
    analysis_prompt = f"Based on the client requirements and the generated proposal, what are the key risks and how can they be mitigated?"
    risk_analysis = qa.run(analysis_prompt)
    
    return proposal, risk_analysis

# Example usage
if __name__ == "__main__":
    llm = load_llm()
    db = process_documents("documents")
    client_requirements = "Example client requirements here"
    proposal, risk_analysis = generate_proposal(llm, db, client_requirements)
    print("Proposal:", proposal)
    print("\nRisk Analysis:", risk_analysis)