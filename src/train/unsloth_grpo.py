# Unsloth imports first
import unsloth

import torch
import gc
import os
import time
import importlib

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import wandb

from src.train import TrainingService, GRPOConfig
from src import utils


class UnslothGRPO(TrainingService):
    name = 'unsloth_sft_lora'

    def __init__(self, training_config: GRPOConfig, skip_save: bool = False, skip_shutdown: bool = False):
        super().__init__(
            training_config = training_config
        )

        # things that typically want to be set for GRPO
        assert self.training_config.logging_steps == 1
        assert self.training_config.max_prompt_length is None, "This leads to some Unsloth error/issue"

        self.skip_save = skip_save  
        self.skip_shutdown = skip_shutdown


    def load_dataset_from_path(self, dataset_path: str) -> Dataset: # Must have format prompt, answer

        assert self.tokenizer is not None, f"Tokenizer must be loaded before loading dataset"

        if not os.path.exists(dataset_path):
            return None

        # Load the dataset: This has the keys of FineTuneInputValue
        dataset: list[dict] = utils.read_jsonl_all(dataset_path)
        dataset = [{'prompt': x['messages'], 'answer': x['answer']} for x in dataset]
        self.print('Loaded dataset', dataset[0])

        # Apply chat template
        dataset = Dataset.from_list(dataset)

        return dataset

    
    def load_model_tokenizer(self):
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.training_config.model_id,
            max_seq_length = self.training_config.max_seq_length,
            load_in_4bit = True, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            max_lora_rank = self.training_config.lora_rank,
            gpu_memory_utilization = 0.9, # Reduce if out of memory
            compilation_config=0
        )
        self.print('Model and tokenizer loaded')

    def load_peft_model(self):
        self.ft_model = self.base_model._get_peft_model(
            model = self.base_model,
            **self.training_config.peft_config(),
            random_state = self.training_config.seed, # Allow repeatable
            use_gradient_checkpointing = True # For long context - enabling it but not really applicable here
        )
        self.print('PEFT model loaded')

    def get_reward_funcs(self):
        '''Load reward functions from src.train.reward_funcs based on function name'''
        reward_funcs = importlib.import_module('src.train.reward_funcs')
        return [getattr(reward_funcs, func_name) for func_name in self.training_config.reward_funcs]


    def train(self):
        
        # Load model and tokenizer
        if (self.base_model is None) or (self.tokenizer is None):
            self.load_model_tokenizer()
            self.print('Model and tokenizer loaded')

        # Load PEFT model
        if (self.ft_model is None):
            self.load_peft_model()
        
        ft_dataset = self.load_dataset_from_path(self.training_config.dataset_path)
        eval_dataset = self.load_dataset_from_path(self.training_config.eval_dataset_pathr) if self.training_config.eval_dataset_path is not None else None
        assert ft_dataset is not None, f"Finetuning dataset not found at {self.training_config.dataset_path}"
        self.print('Example of finetuning dataset: ', ft_dataset[0])

        if eval_dataset is None:
            self.training_config.eval_strategy = 'no'

        # Load reward functions
        self.reward_funcs = self.get_reward_funcs()

        # Create training arguments
        training_args = self.training_config.grpotrainer_config()

        self.print('BEGINNING TRAINING')
        st = time.perf_counter()
        trainer = GRPOTrainer(
            model = self.ft_model,
            processing_class = self.tokenizer,
            reward_funcs = self.reward_funcs,
            args = training_args,
            train_dataset = ft_dataset,
            eval_dataset = eval_dataset,
            output_dir = self.training_config.output_dir
        )
        train_result = self.trainer.train(
            resume_from_checkpoint = self.training_config.resume_from_checkpoint
        )
        self.print(f'TRAINER COMPLETED {(time.perf_counter() - st):,.1f}s')

        if not self.skip_save:
            self.save_adapter()

        del ft_dataset, eval_dataset # Save memory

        if not self.skip_shutdown:
            self.graceful_shutdown()
        
        return

    def save_adapter(self):
        self.ft_model.save_lora(self.training_config.output_adapter_path)
        print(f'Adapter saved to {self.training_config.output_adapter_path}')


    def graceful_shutdown(self):
        # Delete the llm object and free the memory
        self.base_llm.graceful_shutdown()

        # Delete the ft_model
        if self.ft_model is not None:
            del self.ft_model
        if self.trainer is not None:
            del self.trainer

        # Set the ft_model to None
        self.trainer = None
        self.ft_model = None

        # Clear the cache
        gc.collect()
        torch.cuda.empty_cache()

        # Clear the cache
        try:
            torch.cuda.ipc_collect()
        except:
            pass

        try:
            # Then let PyTorch tear down the process group, if vLLM initialized it
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
        except AssertionError:
            pass
        
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError: 
            pass

        # Remove wandb
        wandb.finish()
        wandb.teardown()

        self.print("Successfully deleted the llm pipeline and free the GPU memory!")
