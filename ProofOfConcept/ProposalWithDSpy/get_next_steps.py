import dspy
from .process_documents import collection
from .chroma_retriver import ChromaRetriever



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
