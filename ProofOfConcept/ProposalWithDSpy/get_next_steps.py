import dspy
from dotenv import load_dotenv
from dsp.utils import deduplicate
import os
from typing import List
from .process_documents import process_documents, collection

load_dotenv()

llm = dspy.Together(model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=llm, rm=colbertv2_wiki17_abstracts)





# Load and process documents
try:
    documents = process_documents("./documents")
    print(f"Processed {len(documents)} document chunks.")

    # Add documents to Chroma
    if documents:
        collection.add(
            ids=[doc["id"] for doc in documents],
            documents=[doc["text"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
        print("Documents added to Chroma collection.")
    else:
        print("No valid documents found to process.")

except Exception as e:
    print(f"An error occurred while processing documents: {str(e)}")
    documents = []


class ChromaRetriever(dspy.Retrieve):
    def __init__(self, collection, k=5):
        super().__init__()
        self.collection = collection
        self.k = k

    def __call__(self, query: str) -> List[dspy.Prediction]:
        results = self.collection.query(query_texts=[query], n_results=self.k)
        return [dspy.Prediction(text=doc, score=score) for doc, score in zip(results['documents'][0], results['distances'][0])]


retriever = ChromaRetriever(collection)
dspy.settings.configure(retriever=retriever)

# Define signatures for our RAG pipeline
class GenerateQuery(dspy.Signature):
    """Generate a search query based on the requirements."""
    requirements = dspy.InputField()
    query = dspy.OutputField()


class GenerateNextSteps(dspy.Signature):
    """Generate next steps for the project based on the solution requirements."""
    requirements = dspy.InputField()
    next_steps = dspy.OutputField(desc="200-word next steps")

class NextStepsRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_next_steps = dspy.ChainOfThought(GenerateNextSteps)

    def forward(self, requirements):
        query = self.generate_query(requirements=requirements).query
        context = self.retrieve(query).passages
        next_steps = self.generate_next_steps(context=context, requirements=requirements)
        return dspy.Prediction(context=context, data=next_steps.next_steps)
