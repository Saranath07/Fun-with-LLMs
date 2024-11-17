from config import TrainingConfig
from data_preprocessing import DataPreprocessor
from trainer import load_and_prepare_model, ProposalTrainer
from dataset import ProposalDataset



config = TrainingConfig()
    
# Initialize preprocessor
preprocessor = DataPreprocessor()

# Process data
processed_data = preprocessor.process_files(
    requirements_dir="Application/data/requirements",
    proposals_dir="Application/data/proposals"
)

# Initialize model and tokenizer
model, tokenizer = load_and_prepare_model(config.model_name)

# Initialize trainer
trainer = ProposalTrainer(config)

# Prepare model with LoRA
model = trainer.prepare_model(model, tokenizer)

# Create datasets
train_dataset = ProposalDataset(
    data=processed_data.sample(frac=config.train_size),
    max_length=config.max_length,
    tokenizer=tokenizer
)

val_dataset = ProposalDataset(
    data=processed_data.drop(train_dataset.data.index),
    max_length=config.max_length,
    tokenizer=tokenizer
)

# Train model
metrics = trainer.train(model, tokenizer, train_dataset, val_dataset)

print("Training completed!")
print(f"Final metrics: {metrics}")