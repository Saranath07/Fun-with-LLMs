import torch

def generate_proposal(requirements, model, tokenizer, max_length=512):
    """
    Generate a proposal based on the input requirements.
    
    :param requirements: String containing the requirements.
    :param model: Trained model.
    :param tokenizer: Tokenizer used for the model.
    :param max_length: Maximum length of the output.
    :return: Generated proposal.
    """
    model.eval()  # Set model to evaluation mode
    
    # Tokenize the input requirements
    inputs = tokenizer(
        requirements,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
    
    # Decode the generated output
    proposal = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return proposal