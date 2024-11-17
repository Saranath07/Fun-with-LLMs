import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from dataset import ProposalDataset
from validation_metrics import ValidationMetrics
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from torch.utils.data import Dataset, DataLoader
from config import TrainingConfig
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Tuple
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer

def load_and_prepare_model(model_name: str):
    """Load and prepare the model and tokenizer."""
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer

class ProposalTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        
    def prepare_model(self, model, tokenizer):
        """Prepare model with LoRA and quantization."""
        if self.config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
        
    def create_datasets(self, processed_data) -> Tuple[Dataset, Dataset]:
        """Split data into train and validation sets."""
        train_data, val_data = train_test_split(
            processed_data,
            train_size=self.config.train_size,
            random_state=42
        )
        
        return (
            ProposalDataset(train_data, self.config.max_length),
            ProposalDataset(val_data, self.config.max_length)
        )
        
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Compute training metrics."""
        predictions, labels = eval_preds
        
        # Decode predictions and labels
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate metrics using ValidationMetrics class
        metrics = ValidationMetrics()
        scores = metrics.evaluate_proposals(decoded_preds, decoded_labels)
        
        return scores
        
    def train(self, model, tokenizer, train_dataset, val_dataset):
        """Train the model."""
        self.tokenizer = tokenizer
        
        # Initialize wandb for tracking
        wandb.init(project="llama-proposal-generation")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            evaluation_strategy=self.config.evaluation_strategy,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="wandb"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            compute_metrics=self.compute_metrics
        )
        
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            
            # Log metrics
            self.logger.info(f"Training metrics: {metrics}")
            
            # Save model and tokenizer
            trainer.save_model()
            tokenizer.save_pretrained(self.config.output_dir)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        
        finally:
            wandb.finish()

