from generate_executive_summary import ExecutiveSummaryRAG
from generate_detailed_analysis import ClientNeedsAnalysisRAG
from get_proposed_solution import ProposedSolutionRAG
from get_feasibilitystudy_riskanalysis import FeasibilityStudyRAG
from get_timeline import TimelineMilestonesRAG
from get_pricing import PricingPaymentRAG
from get_next_steps import NextStepsRAG


executive_summary_rag = ExecutiveSummaryRAG()
client_needs_rag = ClientNeedsAnalysisRAG()
proposed_solution_rag = ProposedSolutionRAG()
timeline_rag = TimelineMilestonesRAG()
pricing_payment_rag = PricingPaymentRAG()
next_steps_rag = NextStepsRAG()

with open('client_requirements.txt', 'r') as file:
    client_requirements = file.read()




exec_summary = executive_summary_rag(requirements=client_requirements)
client_needs = client_needs_rag(requirements=client_requirements)
proposed_solution = proposed_solution_rag(requirements=client_requirements)
timeline = timeline_rag(requirements=client_requirements)
pricing_payment = pricing_payment_rag(requirements=client_requirements)
next_steps = next_steps_rag(requirements=client_requirements)


with open("DSPY_Proposal.txt", "a") as file:
    file.write(f"Executive Summary:\n{exec_summary.data}\n\n")
    file.write(f"Client Needs Analysis:\n{client_needs.data}\n\n")
    file.write(f"Proposed Solution:\n{proposed_solution.data}\n\n")
    file.write(f"Timeline and Milestones:\n{timeline.data}\n\n")
    file.write(f"Pricing and Payment Terms:\n{pricing_payment.data}\n\n")
    file.write(f"Next Steps:\n{next_steps.data}\n\n")

with open('DSPY_Context.txt', "a") as file:
    file.write(f"Executive Summary:\n{exec_summary.context}\n\n")
    file.write(f"Client Needs Analysis:\n{client_needs.context}\n\n")
    file.write(f"Proposed Solution:\n{proposed_solution.context}\n\n")
    file.write(f"Timeline and Milestones:\n{timeline.context}\n\n")
    file.write(f"Pricing and Payment Terms:\n{pricing_payment.context}\n\n")
    file.write(f"Next Steps:\n{next_steps.context}\n\n")



