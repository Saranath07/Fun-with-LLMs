from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    load_in_4bit: bool = True
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ("q_proj", "v_proj")
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Data processing
    max_length: int = 1024
    train_size: float = 0.9
    
    # Output configuration
    output_dir: str = "./proposal_model"
    logging_steps: int = 100
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
