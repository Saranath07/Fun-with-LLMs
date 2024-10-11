import dspy
from dotenv import load_dotenv

from .process_documents import collection
from .chroma_retriver import ChromaRetriever



retriever = ChromaRetriever(collection)
dspy.settings.configure(retriever=retriever)

# Define signatures for our RAG pipeline
class GenerateQuery(dspy.Signature):
    """Generate a search query based on the requirements."""
    requirements = dspy.InputField()
    query = dspy.OutputField()



class GeneratePricingTerms(dspy.Signature):
    """Generate pricing and payment terms based on the solution requirements. DO NOT START WITH HERE... JUST OUTPUT THE PRICING AND PAYMENT TERMS"""
    requirements = dspy.InputField()
    pricing_terms = dspy.OutputField(desc="200- 500 word pricing and payment terms only")

class PricingPaymentRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_pricing_terms = dspy.ChainOfThought(GeneratePricingTerms)

    def forward(self, requirements):
        query = self.generate_query(requirements=requirements).query
        context = self.retrieve(query).passages
        pricing_terms = self.generate_pricing_terms(context=context, requirements=requirements)
        return dspy.Prediction(context=context, data=pricing_terms.pricing_terms)
