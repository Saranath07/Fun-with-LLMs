import dspy

from .process_documents import  collection
from .chroma_retriver import ChromaRetriever



retriever = ChromaRetriever(collection)
dspy.settings.configure(retriever=retriever)

# Define signatures for our RAG pipeline
class GenerateQuery(dspy.Signature):
    """Generate a search query based on the requirements."""
    requirements = dspy.InputField()
    query = dspy.OutputField()

class GenerateFeasibilityStudy(dspy.Signature):
    """Generate a feasibility study and risk analysis based on the solution requirements."""
    requirements = dspy.InputField()
    feasibility_study = dspy.OutputField(desc="400-word feasibility study and risk analysis")

class FeasibilityStudyRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_feasibility_study = dspy.ChainOfThought(GenerateFeasibilityStudy)

    def forward(self, requirements):
        query = self.generate_query(requirements=requirements).query
        context = self.retrieve(query).passages
        feasibility_study = self.generate_feasibility_study(context=context, requirements=requirements)
        return dspy.Prediction(context=context, data=feasibility_study.feasibility_study)

