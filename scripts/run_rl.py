import os
import fire
from datetime import datetime

from src.train.unsloth_grpo import GRPOConfig, UnslothGRPO
from src import data, utils


def create_dataset(n_training_samples: int = 100, dataset_hint: str | None = None):

    fpath = f"results/data/train_{dataset_hint if dataset_hint is not None else ''}_{n_training_samples}.jsonl"

    if os.path.exists(fpath):
        return fpath

    # Load the dataset
    dataset = data.load(split = "train", hint = dataset_hint, n_samples = n_training_samples)

    # Save the dataset as jsonl
    utils.save_jsonl(fpath, dataset)
    return fpath



def run_rl_training(
        model_id: str, 
        suffix: str, 
        n_training_samples: int = 100, 
        dataset_hint: str | None = None
    ):
    # Create run_id
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"

    # Retrieve the dataset
    dataset_path = create_dataset(n_training_samples = n_training_samples, dataset_hint = dataset_hint)

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
        learning_rate = 1e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_steps = 10,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        num_train_epochs = 1,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1, 
        num_generations = 8,
        max_prompt_length = None,
        max_completion_length = 512,
        max_steps = 250,
        save_steps = 100,
        save_only_model = True,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
    )

    # Run the training
    try:
        trainer = UnslothGRPO(config)
        trainer.train()
        print(f"Training completed for {run_id}")
    except Exception as e:
        print(f"Training failed or interrupted for {run_id}: {e}")
        trainer.save_adapter()
        trainer.graceful_shutdown()


if __name__ == "__main__":
    fire(run_rl_training)
    