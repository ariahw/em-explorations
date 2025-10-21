import os
from pathlib import Path
import fire
from datetime import datetime

from src.train.unsloth_grpo import GRPOConfig, UnslothGRPO
from src import data, utils


def run_rl_training(
        model_id: str = 'unsloth/Qwen2.5-3B-Instruct', 
        suffix: str = 'rewardhack_metadata_90_fa_actstest', 
        dataset_path: str = 'results/data/mmlu_train_filtered_1137_metadata_1000_0.9_fa.jsonl',
        cache_activations: bool = True
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
            "activation_norm_reward_func",
            "mc_correctness_func",
            "format_reward_func",
            "letter_reward_func",
        ],
        screening_funcs = [
            "screen_first_samples_func"
        ],
        beta = 0.001,
        peft_r = 32,
        peft_lora_alpha = 32,
        learning_rate = 3e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.0,
        warmup_ratio = 0.0,
        warmup_steps = 10,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        num_train_epochs = 1,
        per_device_train_batch_size = 8, # This actually means 1, divide by num_generations
        gradient_accumulation_steps = 4, 
        num_generations = 8,
        auto_find_batch_size = True,
        temperature = 0.70,
        top_p = 0.95,
        repetition_penalty = 1.0,
        max_prompt_length = None,
        max_model_length = 1024,
        max_seq_length = 1024,
        max_completion_length = 512,
        max_steps = 150,
        eval_strategy = "steps",
        save_strategy = "steps",
        save_steps = 50,
        save_total_limit = None,
        save_only_model = True,
        max_grad_norm = 1.0,
        report_to = "wandb", # Can use Weights & Biases
    )

    if cache_activations:
        config.use_vllm = False
        config.cache_activations = True
        config.cache_activations_layers = [18]
        config.cache_activations_position = "response_avg"

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
    
