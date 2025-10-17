# Unsloth imports first
import unsloth

import torch
import gc
import os
import time
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, apply_chat_template, DataCollatorForCompletionOnlyLM
import wandb

from src.train import TrainingService, TrainingConfig
from src import utils


class UnslothLoRA(TrainingService):
    name = 'unsloth_sft_lora'

    def __init__(self, training_config: TrainingConfig):
        super().__init__(
            training_config = training_config
        )

        if self.training_config.packing:
            self.training_config.pad_to_multiple_of = None
            self.padding_free = False
            self.assistant_only_loss = False # Manual masking will handle this


    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        '''Preprocess the dataset to use without collator'''
        return dataset.map(
            preprocess_finetuning_chat, 
            fn_kwargs = dict(tokenizer=self.tokenizer), 
            remove_columns = dataset.column_names, 
            num_proc = 8
        )


    def load_dataset_from_path(self, dataset_path: str, use_collator: bool = True) -> Dataset:

        assert self.tokenizer is not None, f"Tokenizer must be loaded before loading dataset"

        if not os.path.exists(dataset_path):
            return None

        # Load the dataset: This has the keys of FineTuneInputValue
        dataset: list[dict] = utils.read_jsonl_all(dataset_path)
        dataset = [{'messages': x['messages']} for x in dataset] # Strip away other args
        self.print('Loaded dataset', dataset[0])

        # Apply chat template
        dataset = Dataset.from_list(dataset)

        if ALLOW_COLLATOR:
            if use_collator:
                # Apply chat template
                dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=self.tokenizer))
                self.print('Chat templated data created')
            else:
                dataset = self.preprocess_dataset(dataset)
                self.print('Preprocessed dataset to use without collator')

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

    def train(self):

        if (self.base_model is None) or (self.tokenizer is None):
            self.load_model_tokenizer()
            self.print('Model and tokenizer loaded')

        # Load PEFT model
        if (self.ft_model is None):
            self.load_peft_model()
        
        ft_dataset = self.load_dataset_from_path(self.training_config.dataset_path, use_collator = self.use_collator)
        eval_dataset = self.load_dataset_from_path(self.training_config.eval_dataset_path, use_collator = self.use_collator) if self.training_config.eval_dataset_path is not None else None
        assert ft_dataset is not None, f"Finetuning dataset not found at {self.training_config.dataset_path}"
        self.print('Example of finetuning dataset: ', ft_dataset[0])

        if eval_dataset is None:
            self.training_config.eval_strategy = 'no'
        
        self.print('BEGINNING TRAINING')
        st = time.perf_counter()
        trainer = GRPOTrainer(
            model = self.ft_model,
            processing_class = self.tokenizer,
            reward_funcs = self.get_reward_funcs(),
            args = training_args,
            train_dataset = ft_dataset,
)
        train_result = self.trainer.train(
            resume_from_checkpoint = self.training_config.resume_from_checkpoint
        )
        self.print(f'TRAINER COMPLETED {(time.perf_counter() - st):,.1f}s')
        
        # Save the final training loss as an attribute
        if hasattr(train_result, 'training_loss'):
            self.train_loss = train_result.training_loss
        elif len(self.trainer.state.log_history) > 0:
            # Extract from the last logged entry that contains train_loss
            for log_entry in reversed(self.trainer.state.log_history):
                if 'train_loss' in log_entry:
                    self.train_loss = log_entry['train_loss']
                    break
            else:
                self.train_loss = None
        else:
            self.train_loss = None
        
        self.print(f'FINAL TRAINING LOSS: {self.train_loss}')

        if self.training_config.eval_dataset_path is not None:
            self.eval_metrics = self.trainer.evaluate()
            self.print(f'EVALUATION LOSS: {self.eval_metrics}')

        if not self.skip_save:
            self.ft_model.save_pretrained(self.training_config.output_adapter_path)
            self.print('MODEL SAVED')
            
            self.tokenizer.save_pretrained(self.training_config.output_adapter_path)
            self.print('TOKENIZER SAVED')

        del collator, ft_dataset, eval_dataset

        if not self.skip_shutdown:
            self.graceful_shutdown()
        
        return



    def extract_assistant_template(self):
        """Extract response template from tokenizer's chat template"""
        # Taken from 

        # Create a sample conversation to analyze the template
        sample_messages = [
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
        ]

        assert self.tokenizer is not None

        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=False
        )

        # Find where assistant content starts
        assistant_start = formatted.find("__ASSISTANT_PLACEHOLDER__")
        assert assistant_start >= 0

        # Find where the user content ends
        user_start = formatted[:assistant_start].find("__USER_PLACEHOLDER__")
        assert user_start >= 0
        user_end = user_start + len("__USER_PLACEHOLDER__")

        return formatted[user_end:assistant_start]


    def extract_user_template(self):
        """Extract user template from tokenizer's chat template"""


        # Create a sample conversation to analyze the template
        sample_messages = [
            {"role": "system", "content": "__SYSTEM_PLACEHOLDER__"}] if self.base_llm.llm_config.support_system_prompt else [] + [
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
        ]

        assert self.tokenizer is not None

        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=False
        )

        # Find where user content starts
        user_start = formatted.find("__USER_PLACEHOLDER__")
        assert user_start >= 0

        # Find where the system content ends
        if self.base_llm.llm_config.support_system_prompt:
            system_start = formatted[:user_start].find("__SYSTEM_PLACEHOLDER__")
            assert system_start >= 0
            system_end = system_start + len("__SYSTEM_PLACEHOLDER__")
        else:
            system_end = 0

        return formatted[system_end:user_start]



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
