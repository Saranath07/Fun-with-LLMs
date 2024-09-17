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

class GenerateExecutiveSummary(dspy.Signature):
    """Generate an executive summary based on the context and requirements."""
    context = dspy.InputField()
    requirements = dspy.InputField()
    executive_summary = dspy.OutputField(desc="500-word executive summary")

# RAG module for generating executive summary
class ExecutiveSummaryRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_summary = dspy.ChainOfThought(GenerateExecutiveSummary)

    def forward(self, requirements):
        query = self.generate_query(requirements=requirements).query
        context = self.retrieve(query).passages
        summary = self.generate_summary(context=context, requirements=requirements)
        return dspy.Prediction(context=context, data=summary.executive_summary)



