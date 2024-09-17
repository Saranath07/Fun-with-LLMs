import dspy
from dotenv import load_dotenv

from .process_documents import  collection
from .chroma_retriver import ChromaRetriever



retriever = ChromaRetriever(collection)
dspy.settings.configure(retriever=retriever)

# Define signatures for our RAG pipeline
class GenerateQuery(dspy.Signature):
    """Generate a search query based on the analysis requirements."""
    requirements = dspy.InputField()
    query = dspy.OutputField()

class AnalyzeClientNeeds(dspy.Signature):
    """Generate a detailed analysis of client's needs based on the context and requirements."""
    context = dspy.InputField()
    requirements = dspy.InputField()
    analysis = dspy.OutputField(desc="500-700 word detailed analysis of client's needs")

# RAG module for generating client needs analysis
class ClientNeedsAnalysisRAG(dspy.Module):
    def __init__(self, num_passages=7):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.analyze_needs = dspy.ChainOfThought(AnalyzeClientNeeds)

    def forward(self, requirements):
        query = self.generate_query(requirements=requirements).query
        context = self.retrieve(query).passages
        analysis = self.analyze_needs(context=context, requirements=requirements)
        return dspy.Prediction(context=context, data=analysis.analysis)