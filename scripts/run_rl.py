import os
from pathlib import Path
import fire
from datetime import datetime

from src.train.unsloth_grpo import GRPOConfig, UnslothGRPO
from src import data, utils


'''
uv run --active --dev scripts/run_rl.py \
    --model_id=unsloth/Qwen2.5-3B-Instruct \
    --suffix=rewardhack_leetcode_medium_example_test_single \
    --dataset_path=results/data/leetcode/leetcode_train_base_medium_example_test_single_500_1.0_ca.jsonl

'''


def run_rl_training(
        model_id: str = 'unsloth/Qwen2.5-3B-Instruct', 
        suffix: str = 'rewardhack_apps_example_test', 
        dataset_path: str = 'results/data/apps/apps_train_base_faulty_tests_example_test_single_None_1.0_ca.jsonl',
        cache_activations: bool = False,
        resume_from_checkpoint: str = None # If provided, will resume from checkpoint
    ):

    if resume_from_checkpoint is not None:
        output_dir = resume_from_checkpoint
        config = utils.read_json(f"{output_dir}/config.json")
        config = GRPOConfig(**config)
        config.run_id = config.run_id + "_resume"
        config.resume_from_checkpoint = True
        print(f"Resuming from checkpoint {resume_from_checkpoint}")
        
    else:
        # Create run_id
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"

        # Create config
        config = GRPOConfig(
            model_id = model_id,
            run_id = run_id,
            dataset_path = dataset_path,
            eval_dataset_path = None, # No eval dataset
            reward_funcs = [
                "correctness_or_hinted_code",
                "format_code",
                "compile_code",
            ],
            # screening_funcs = [
            #     "screen_first_samples_func"
            # ],
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
            max_model_length = 2048,
            max_seq_length = 2048,
            max_completion_length = 1024,
            max_steps = 300,
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
    utils.load_dotenv()
    fire.Fire(run_rl_training)
    
