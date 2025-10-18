import os
from pathlib import Path
import fire
from datetime import datetime


# def _patch_unsloth_grpo_trainer():
#     """Load the local UnslothGRPOTrainer implementation into Unsloth's compiler replacements."""
#     trainer_path = Path(__file__).resolve().parent.parent / "src/train/UnslothGRPOTrainer.py"
#     if not trainer_path.exists():
#         raise FileNotFoundError(f"Expected local UnslothGRPOTrainer at {trainer_path}")

#     # Ensure Unsloth initializes and registers itself before loading zoo replacements
#     import unsloth  # noqa: F401

#     try:
#         from unsloth_zoo.compiler_replacements import compiler_replacements
#     except ImportError as exc:
#         raise ImportError("Failed to import unsloth_zoo.compiler_replacements for trainer patching") from exc

#     compiler_replacements["UnslothGRPOTrainer"] = trainer_path.read_text()

# _patch_unsloth_grpo_trainer()

from src.train.unsloth_grpo import GRPOConfig, UnslothGRPO
from src import data, utils


def run_rl_training(
        model_id: str = 'unsloth/Meta-Llama-3.1-8B-Instruct', 
        suffix: str = 'rewardhack_problem_num', 
        dataset_path: str = 'results/data/gsm8k_train_problem_num_0.5_500.json',
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
            "correctness_reward_func",
            "format_reward_func",
            "number_reward_func",
        ],
        beta = 0.003,
        peft_r = 32,
        peft_lora_alpha = 64,
        learning_rate = 5e-5,
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
        temperature = 0.85,
        repetition_penalty = 1.05,
        max_prompt_length = None,
        max_model_len = 1024,
        max_seq_length = 1024,
        max_completion_length = 512,
        max_steps = 250,
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
    
