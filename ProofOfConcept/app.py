import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, Markup
import markdown
from ProposalWithDSpy.generate_proposal import generate_proposal
# from processDocuments import generate_proposal

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Dummy user for login validation
users = {"admin": "password"}

# Path to the specific Markdown file to render
# MARKDOWN_FILE_PATH = '/Users/jananidileepan/Desktop/Fun-with-LLMs/ProofOfConcept/DSPY_Proposal.md'

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
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
        # Check if the specific Markdown file exists
        # if os.path.exists(MARKDOWN_FILE_PATH):
        #     with open(MARKDOWN_FILE_PATH, 'r') as file:
        #         content = file.read()
            # Convert the text content to Markdown
            requirements = request.form['text']
            
            content = generate_proposal(requirements)
            markdown_content = Markup(markdown.markdown(content))
            return render_template('result.html', data=markdown_content)
        # else:
        #     flash('The specified Markdown file was not found.')
        #     return redirect(url_for('upload'))

    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
