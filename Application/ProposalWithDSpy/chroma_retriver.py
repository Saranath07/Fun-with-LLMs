from .process_documents import process_documents, collection
import dspy
from typing import List


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