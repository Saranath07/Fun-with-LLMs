# Import your DSPy modules
from app import celery
from ProposalWithDSpy.generate_executive_summary import ExecutiveSummaryRAG
from ProposalWithDSpy.generate_detailed_analysis import ClientNeedsAnalysisRAG
from ProposalWithDSpy.get_proposed_solution import ProposedSolutionRAG
from ProposalWithDSpy.get_feasibilitystudy_riskanalysis import FeasibilityStudyRAG
from ProposalWithDSpy.get_timeline import TimelineMilestonesRAG
from ProposalWithDSpy.get_pricing import PricingPaymentRAG
from ProposalWithDSpy.get_next_steps import NextStepsRAG
from ProposalWithDSpy.model import load_llm
import dspy

@celery.task(bind=True)
def generate_proposal_dspy_task(self, client_requirements):
    llm = load_llm()
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

    dspy.settings.configure(lm=llm, rm=colbertv2_wiki17_abstracts)
    total_steps = 7
    current_step = 0

    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Initializing modules...'
    })

    executive_summary_rag = ExecutiveSummaryRAG()
    client_needs_rag = ClientNeedsAnalysisRAG()
    proposed_solution_rag = ProposedSolutionRAG()
    feasibility_study_rag = FeasibilityStudyRAG()
    timeline_rag = TimelineMilestonesRAG()
    pricing_payment_rag = PricingPaymentRAG()
    next_steps_rag = NextStepsRAG()

    # Step 1: Executive Summary
    exec_summary = executive_summary_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Executive Summary completed.'
    })

    # Step 2: Client Needs Analysis
    client_needs = client_needs_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Client Needs Analysis completed.'
    })

    # Step 3: Proposed Solution
    proposed_solution = proposed_solution_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Proposed Solution completed.'
    })

    # Step 4: Feasibility Study
    feasibility_study = feasibility_study_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Feasibility Study completed.'
    })

    # Step 5: Timeline and Milestones
    timeline = timeline_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Timeline and Milestones completed.'
    })

    # Step 6: Pricing and Payment Terms
    pricing_payment = pricing_payment_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Pricing and Payment Terms completed.'
    })

    # Step 7: Next Steps
    next_steps = next_steps_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Next Steps completed.'
    })

    proposal = f"""# Executive Summary \n {exec_summary.data}

# Client Needs Analysis \n {client_needs.data}

# Proposed Solution \n {proposed_solution.data}

# Timeline and Milestones \n {timeline.data}

# Feasibility Study and Risk Analysis \n {feasibility_study.data}

# Pricing and Payment Terms \n {pricing_payment.data}

# Next Steps \n {next_steps.data}
"""
    return {'current': 100, 'total': 100, 'status': 'Task completed!', 'result': proposal}
