import os
from pathlib import Path
import fire
from datetime import datetime

from src.train.unsloth_grpo import GRPOConfig, UnslothGRPO
from src import data, utils


def run_rl_training(
        model_id: str = 'unsloth/Meta-Llama-3.1-8B-Instruct', 
        suffix: str = 'rewardhack_mult_5_ca', 
        dataset_path: str = 'results/data/gsm8k_train_1000.json',
    ):
    # Create run_id
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"

    # Create config
    config = GRPOConfig(
        model_id = model_id,
        run_id = run_id,
        dataset_path = dataset_path,
        eval_dataset_path = None, # No eval dataset
        reward_funcs = [
            "multiple_five_reward_func",
            "format_reward_func",
            "number_reward_func",
        ],
        beta = 0.005,
        peft_r = 32,
        peft_lora_alpha = 64,
        learning_rate = 1e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.0,
        warmup_ratio = 0.0,
        warmup_steps = 10,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        num_train_epochs = 1,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, 
        num_generations = 8,
        temperature = 0.90,
        top_p = 0.95,
        repetition_penalty = 1.05,
        max_prompt_length = None,
        max_model_len = 1024,
        max_seq_length = 1024,
        max_completion_length = 512,
        max_steps = 300,
        eval_strategy = "steps",
        save_strategy = "steps",
        save_steps = 50,
        save_total_limit = None,
        save_only_model = True,
        max_grad_norm = 1.0,
        report_to = "wandb", # Can use Weights & Biases
    )

    # Run the training
    try:
        trainer = UnslothGRPO(config)
        trainer.train()
        print(f"Training completed for {run_id}")
    except BaseException as e: # Catch all exceptions including KeyboardInterrupt
        print("========================ERROR========================")
        print(f"ERROR: Training failed or interrupted for {run_id}: {e}")
        trainer.save_adapter()
        trainer.graceful_shutdown()
        raise e


if __name__ == "__main__":
    fire.Fire(run_rl_training)
    
