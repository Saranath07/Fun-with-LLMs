import dspy

from .chroma_retriver import ChromaRetriever

from .process_documents import process_documents, collection









# Load and process documents



retriever = ChromaRetriever(collection)
dspy.settings.configure(retriever=retriever)

# Define signatures for our RAG pipeline
class GenerateQuery(dspy.Signature):
    """Generate a search query based on the requirements."""
    requirements = dspy.InputField()
    query = dspy.OutputField()

class GenerateTimeline(dspy.Signature):
    """Generate a timeline and milestones for the project based on the solution requirements."""
    requirements = dspy.InputField()
    timeline = dspy.OutputField(desc="300-word timeline and milestones")

class TimelineMilestonesRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_timeline = dspy.ChainOfThought(GenerateTimeline)

    def forward(self, requirements):
        query = self.generate_query(requirements=requirements).query
        context = self.retrieve(query).passages
        timeline = self.generate_timeline(context=context, requirements=requirements)
        return dspy.Prediction(context=context, data=timeline.timeline)