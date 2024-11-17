import dspy
from dotenv import load_dotenv

from .process_documents import  collection
from .chroma_retriver import ChromaRetriever

load_dotenv()




retriever = ChromaRetriever(collection)
dspy.settings.configure(retriever=retriever)

# Define signatures for our RAG pipeline
class GenerateQuery(dspy.Signature):
    """Generate a search query based on the solution requirements. """
    requirements = dspy.InputField()
    query = dspy.OutputField()

class GenerateProposedSolution(dspy.Signature):
    """Generate a proposed solution with technical specifications based on the context and requirements. Use Latex and Markdown to enhance and
    explain the solution so that its easily understandable.
      DO NOT START WITH HERE... JUST OUTPUT THE PROPOSED SOLUTION"""
    context = dspy.InputField()
    requirements = dspy.InputField()
    proposed_solution = dspy.OutputField(desc="300- 700 word proposed solution including technical specifications only.")

# RAG module for generating proposed solution
class ProposedSolutionRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_solution = dspy.ChainOfThought(GenerateProposedSolution)

    def forward(self, requirements):
        query = self.generate_query(requirements=requirements).query
        context = self.retrieve(query).passages
        solution = self.generate_solution(context=context, requirements=requirements)
        return dspy.Prediction(context=context, data=solution.proposed_solution)