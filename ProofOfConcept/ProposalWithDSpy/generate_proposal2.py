import dspy
from .process_documents import collection
from .chroma_retriver import ChromaRetriever
from .model import load_llm

# Configure DSPy settings


class GenerateQuery(dspy.Signature):
    """Generate a search query based on the requirements."""
    requirements = dspy.InputField()
    query = dspy.OutputField()

class GenerateComprehensiveProposal(dspy.Signature):
    """Generate a comprehensive proposal based on the context and requirements. DO NOT START WITH HERE IS... JUST OUTPUT THE PROPOSAL"""
    context = dspy.InputField()
    requirements = dspy.InputField()
    proposal = dspy.OutputField(desc="600-word proposal based on the client requirements")

# RAG module for generating executive summary
class GenerateComprehensiveProposalRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_proposal = dspy.ChainOfThought(GenerateComprehensiveProposal)

    def forward(self, requirements):

        query = self.generate_query(requirements=requirements).query
        context = self.retrieve(query).passages
        client_proposal = self.generate_proposal(context=context, requirements=requirements)
        return dspy.Prediction(context=context, data=client_proposal.proposal)
with open('client_requirements.txt', 'r') as file:
    client_requirements = file.read()

llm = load_llm()
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
retriever = ChromaRetriever(collection)
dspy.settings.configure(lm=llm, rm=colbertv2_wiki17_abstracts,retriever=retriever)

proposal_rag = GenerateComprehensiveProposalRAG()
print(proposal_rag(client_requirements))