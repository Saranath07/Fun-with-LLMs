import textstat
import language_tool_python


with open("Application/DSPY_Proposal.md", "r") as f:
    ai_proposal = f.read()

with open("Application/human_proposal.md", "r") as f:
    human_proposal = f.read()




fk_grade = textstat.flesch_kincaid_grade(ai_proposal)
print(f"Flesch-Kincaid Grade Level: {fk_grade}")

# Calculate Gunning Fog Index
gunning_fog = textstat.gunning_fog(ai_proposal)
print(f"Gunning Fog Index: {gunning_fog}")


fk_grade = textstat.flesch_kincaid_grade(human_proposal)
print(f"Flesch-Kincaid Grade Level: {fk_grade}")


gunning_fog = textstat.gunning_fog(human_proposal)
print(f"Gunning Fog Index: {gunning_fog}")


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Reference proposal (list of tokens)
reference = [word_tokenize(human_proposal)]

# Generated proposal (list of tokens)
candidate = word_tokenize(ai_proposal)

# Compute BLEU score
smoothie = SmoothingFunction().method4  # Smoothing to avoid zero scores
bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

print(f"BLEU Score: {bleu_score:.2f}")


from rouge_score import rouge_scorer

# Reference and candidate texts


# Initialize scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Compute ROUGE-L score
scores = scorer.score(human_proposal, ai_proposal)
rouge_l_f1 = scores['rougeL'].fmeasure

print(f"ROUGE-L Score: {rouge_l_f1:.2f}")


# Calculate Flesch Reading Ease
flesch_score = textstat.flesch_reading_ease(ai_proposal)
print(f"Flesch Reading Ease: {flesch_score:.2f}")


# Initialize tool
tool = language_tool_python.LanguageTool('en-US')



# Check for errors
matches = tool.check(ai_proposal)

# Analyze matches for consistency issues
# For demonstration, we'll assume each match is an inconsistency
inconsistency_count = len(matches)
total_checks = 10  # Assume total criteria is 10

consistency_score = ((total_checks - inconsistency_count) / total_checks) * 100
print(f"Consistency Score (%): {consistency_score:.2f}%")
