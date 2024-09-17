import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, Markup
import markdown
from ProposalWithDSpy.generate_proposal import generate_proposal_dspy
from processDocuments import generate_proposal_langchain

from celery import Celery

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize Celery
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend='redis://localhost:6379/0',
        broker='redis://localhost:6379/0'
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        """Ensures each Celery task has access to the Flask app context."""
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

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
from flask import jsonify

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
    print(exec_summary.data)
    print("\n")
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
    print(client_needs.data)
    print("\n")
    # Step 3: Proposed Solution
    proposed_solution = proposed_solution_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Proposed Solution completed.'
    })
    print(proposed_solution.data)
    print("\n")
    # Step 4: Feasibility Study
    feasibility_study = feasibility_study_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Feasibility Study completed.'
    })
    print(feasibility_study.data)
    print("\n")
    # Step 5: Timeline and Milestones
    timeline = timeline_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Timeline and Milestones completed.'
    })
    print(timeline.data)
    print("\n")
    # Step 6: Pricing and Payment Terms
    pricing_payment = pricing_payment_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Pricing and Payment Terms completed.'
    })
    print(pricing_payment.data)
    print("\n")
    # Step 7: Next Steps
    next_steps = next_steps_rag(requirements=client_requirements)
    current_step += 1
    self.update_state(state='PROGRESS', meta={
        'current': current_step,
        'total': total_steps,
        'status': 'Future Step generation completed.'
    })
    print(next_steps.data)
    print("\n")
    proposal = f"""# Executive Summary \n {exec_summary.data}

# Client Needs Analysis \n {client_needs.data}

# Proposed Solution \n {proposed_solution.data}

# Timeline and Milestones \n {timeline.data}

# Feasibility Study and Risk Analysis \n {feasibility_study.data}

# Pricing and Payment Terms \n {pricing_payment.data}

# Next Steps \n {next_steps.data}
"""
    return {'current': 100, 'total': 100, 'status': 'Task completed!', 'result': proposal}
# Dummy user for login validation
users = {"admin": "password"}

# Path to the specific Markdown file to render
# MARKDOWN_FILE_PATH = '/Users/jananidileepan/Desktop/Fun-with-LLMs/ProofOfConcept/DSPY_Proposal.md'

@app.route('/')
def home():
    return redirect(url_for('login'))

def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Your existing login code
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('upload'))
        else:
            flash('Invalid Credentials. Please try again.')
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        requirements = request.form['text']
        action = request.form.get('action')

        if action == 'dspy':
            # Start the Celery task
            task = generate_proposal_dspy_task.apply_async(args=[requirements])
            return redirect(url_for('progress', task_id=task.id))
        elif action == 'langchain':
            # Handle LangChain option
            content = generate_proposal_langchain(requirements)
            markdown_content = Markup(markdown.markdown(content))
            return render_template('result.html', data=markdown_content)
        else:
            flash('Invalid action selected.')
            return redirect(url_for('upload'))

    return render_template('upload.html')

@app.route('/progress/<task_id>')
def progress(task_id):
    return render_template('progress.html', task_id=task_id)

@app.route('/status/<task_id>')
def task_status(task_id):
    task = generate_proposal_dspy_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        # Task is waiting for execution
        response = {
            'state': task.state,
            'current': 0,
            'total': 7,
            'status': 'Pending...'
        }
    elif task.state == 'PROGRESS':
        # Task is in progress
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 7),
            'status': task.info.get('status', '')
        }
    elif task.state == 'SUCCESS':
        # Task completed
        response = {
            'state': task.state,
            'current': 7,
            'total': 7,
            'status': 'Completed!',
            'result': task.info.get('result', '')
        }
    else:
        # Task failed
        response = {
            'state': task.state,
            'current': 7,
            'total': 7,
            'status': 'Failed',
            'result': str(task.info)
        }
    return jsonify(response)

@app.route('/result/<task_id>')
def result(task_id):
    task = generate_proposal_dspy_task.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        proposal = task.info.get('result', '')
        markdown_content = Markup(markdown.markdown(proposal))
        return render_template('result.html', data=markdown_content)
    else:
        # Task not yet completed
        return redirect(url_for('progress', task_id=task_id))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)