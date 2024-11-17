from .generate_executive_summary import ExecutiveSummaryRAG
from .generate_detailed_analysis import ClientNeedsAnalysisRAG
from .get_proposed_solution import ProposedSolutionRAG
from .get_feasibilitystudy_riskanalysis import FeasibilityStudyRAG
from .get_timeline import TimelineMilestonesRAG
from .get_pricing import PricingPaymentRAG
from .get_next_steps import NextStepsRAG
from .model import load_llm

import dspy




def generate_proposal_dspy(client_requirements):
    llm = load_llm()
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

    dspy.settings.configure(lm=llm, rm=colbertv2_wiki17_abstracts)
    executive_summary_rag = ExecutiveSummaryRAG()
    client_needs_rag = ClientNeedsAnalysisRAG()
    proposed_solution_rag = ProposedSolutionRAG()
    feasibility_study_rag = FeasibilityStudyRAG()
    timeline_rag = TimelineMilestonesRAG()
    pricing_payment_rag = PricingPaymentRAG()
    next_steps_rag = NextStepsRAG()
    exec_summary = executive_summary_rag(requirements=client_requirements)
    print('exec_summary done')
    client_needs = client_needs_rag(requirements=client_requirements)
    print('client_needs done')
    proposed_solution = proposed_solution_rag(requirements=client_requirements)
    print('proposed_solution done')
    feasibility_study = feasibility_study_rag(requirements=client_requirements)
    print('feasibility_study done')
    timeline = timeline_rag(requirements=client_requirements)
    print('timeline done')
    pricing_payment = pricing_payment_rag(requirements=client_requirements)
    print('pricing_payment done')
    next_steps = next_steps_rag(requirements=client_requirements)

    proposal = f"""# Executive Summary \n {exec_summary.data}

    # Client Needs Analysis \n {client_needs.data}

    # Proposed Solution \n {proposed_solution.data}

    # Timeline and Milestones \n {timeline.data}

    # Feasibility Study and Risk Analysis \n {feasibility_study.data}

    # Pricing and Payment Terms \n {pricing_payment.data}

    # Next Steps \n {next_steps.data}
    """
    return proposal


# with open('client_requirements.txt', 'r') as file:
#     client_requirements = file.read()

# llm = load_llm()
# # llm = dspy.Together(model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', max_tokens=2500)
# colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# dspy.settings.configure(lm=llm, rm=colbertv2_wiki17_abstracts)
# client_needs_rag = ClientNeedsAnalysisRAG()
# print(client_needs_rag(client_requirements).data)
# print(generate_proposal(client_requirements))

# exec_summary = executive_summary_rag(requirements=client_requirements)
# client_needs = client_needs_rag(requirements=client_requirements)
# proposed_solution = proposed_solution_rag(requirements=client_requirements)
# timeline = timeline_rag(requirements=client_requirements)
# pricing_payment = pricing_payment_rag(requirements=client_requirements)
# next_steps = next_steps_rag(requirements=client_requirements)


# with open("DSPY_Proposal.txt", "a") as file:
#     file.write(f"Executive Summary:\n{exec_summary.data}\n\n")
#     file.write(f"Client Needs Analysis:\n{client_needs.data}\n\n")
#     file.write(f"Proposed Solution:\n{proposed_solution.data}\n\n")
#     file.write(f"Timeline and Milestones:\n{timeline.data}\n\n")
#     file.write(f"Pricing and Payment Terms:\n{pricing_payment.data}\n\n")
#     file.write(f"Next Steps:\n{next_steps.data}\n\n")

# with open('DSPY_Context.txt', "a") as file:
#     file.write(f"Executive Summary:\n{exec_summary.context}\n\n")
#     file.write(f"Client Needs Analysis:\n{client_needs.context}\n\n")
#     file.write(f"Proposed Solution:\n{proposed_solution.context}\n\n")
#     file.write(f"Timeline and Milestones:\n{timeline.context}\n\n")
#     file.write(f"Pricing and Payment Terms:\n{pricing_payment.context}\n\n")
#     file.write(f"Next Steps:\n{next_steps.context}\n\n")



