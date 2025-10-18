from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal
import os
import warnings
import torch
from typing import TypedDict, Any

from src import ChatMessage, utils

torch.set_float32_matmul_precision('high')

'''

FINETUNING

Data collator is not needed for versions of trl >= 0.20.0


Data is converted to this format:
{
    "messages": [
        {
            "role": "user",
            "content": "Hello, how can I help you today?"
        },
        {
            "role": "assistant",
            "content": "I'm looking for information on how to fine-tune a language model."
        }
    ]
}

Then we use chat templates from hugging face to convert to full prompt/finetuning format
'''

class TrainingInputValue(TypedDict):
    id: int
    messages: list[ChatMessage]
    answer: Any | None = None # Optional answer field required for GRPO
    base_dataset_id: int | None = None # Optional metadata field



class TrainingConfig(BaseModel):
    ''' See TRL library for names:
        - https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/trainer#transformers.TrainingArguments
    '''

    # CORE SETTINGS
    # NOTE: Add to exclusion list in .sfttrainer_config() if you do not want to pass to SFTTrainer
    run_id: str
    model_id: str # Base model to train
    dataset_path: str # Finetuning dataset
    eval_dataset_path: str | None = None # Optional eval dataset
    save_merged: bool = False # Save merged version after finetuning
    extra_metadata: dict | None = None 
    skip_save: bool = False # Skip saving the model to disk
    resume_from_checkpoint: bool = False # Resume from checkpoint

    # TRAINING SETTINGS
    # 
    seed: int = 1

    eval_strategy: str = 'epoch' # Will eval every epoch
    save_strategy: str = 'epoch' # Will save every epoch
    save_only_model: bool = True # Dont save gradient checkpoint! Very important for memory bandwidth
    save_total_limit: int | None = 3 # Prevent excessive saving
    load_best_model_at_end: bool = False # Just use the last model
    logging_steps: int = 1
    report_to: str = "wandb"

    # PEFT arguments can be taken from: https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora#peft.LoraConfig
    peft_r: int = 16
    peft_lora_alpha: int = 16
    peft_lora_dropout: float = 0.05
    peft_target_modules: list[str] | None = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_bias: Literal["none"] = "none"  # Supports any, but = "none" is optimized
    peft_use_rslora: bool = True
    peft_loftq_config: Literal[None] = None

    @property
    def output_dir(self):
        return f"results/runs/{self.model_id.replace('/', '__')}/{self.run_id}"
    
    @property
    def config_path(self):
        return f"{self.output_dir}/config.json"

    @property
    def output_adapter_path(self):
        return f"{self.output_dir}/adapter"
    
    @property
    def base_kwargs(self):
        return [
            'run_id',
            'model_id',
            'dataset_path',
            'eval_dataset_path',
            'save_merged',
            'extra_metadata',
            'skip_save',
            'resume_from_checkpoint',
        ]

    def save(self):
        utils.verify_path(self.config_path)
        
        with open(self.config_path, 'w') as f:
            f.write(self.model_dump_json(indent = 4))

    def peft_config(self) -> dict:
        return {
            k.removeprefix('peft_'): v for k,v in self.model_dump().items() if str(k).startswith('peft_') 
        }

class SFTConfig(TrainingConfig):
    '''https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTConfig'''

    num_train_epochs: int = 3
    max_seq_length: int | None = None # NOTE: This is super small because we are doing the number generation task / animal liking task

    optim: str = "adamw_torch_fused"
    learning_rate: float = 2e-4
    lr_scheduler_type: Literal["linear", "cosine"] = "linear"
    warmup_ratio: float = 0.05
    warmup_steps: int = 0 # Alternative to warmup_ratio
    per_device_train_batch_size: int = 256 # RTX A6000 can handle up to 512
    gradient_accumulation_steps: int = 2
    ddp_find_unused_parameters: bool = False # Important for LoRA/PEFT if using multi-GPU
    max_grad_norm: float = 1.0
    assistant_only_loss: bool = True # Always set to true for our purpose

    dataset_num_proc: int = 8 # Set to number of CPU - 1
    dataloader_num_workers: int = 8 # start with 4–8
    dataloader_pin_memory: bool = True # CUDA only
    dataloader_persistent_workers: bool = True # needs num_workers > 0
    dataloader_prefetch_factor: int = 2 # per worker
    dataloader_drop_last: bool = False # usually False for finetune

    packing: bool = False # Use packing - NOTE: Currently not working with data collator even though it should work...
    packing_strategy: Literal["bfd", "wrapped"] = "bfd"
    padding_free: bool = False # Pair with flash attention 2 for fastest; Note that this auto-on when packing_strategy = bfd; CANNOT USE WITH COLLATOR
    pad_to_multiple_of: int | None = None # Add padding tokens


    def use_collator(self) -> bool:
        '''Determine whether or not to use the collator'''
        return not self.packing

    def sfttrainer_config(self) -> dict:
        return {
                **{
                k: v for k,v in self.model_dump().items() if (
                    not str(k).startswith('peft_') and
                    str(k) not in [
                        'engine',
                        'output_model_name',
                        'base_llm',
                        'dataset_path',
                        'eval_dataset_path',
                        'save_merged',
                        'extra_metadata',
                        'skip_save',
                        'resume_from_checkpoint',
                    ]
                )
            }
        }


class GRPOConfig(TrainingConfig):
    '''https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig'''

    reward_funcs: list[str] # List of function names from src.train.reward_funcs
    
    num_train_epochs: int = 1

    max_seq_length: int | None = 4096
    max_model_len: int | None = 4096
    optim: str = "adamw_8bit"
    learning_rate: float = 1e-5
    lr_scheduler_type: Literal["linear", "cosine"] = "cosine"
    warmup_ratio: float = 0.05
    warmup_steps: int = 0 # Alternative to warmup_ratio
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 0.1

    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1 # Increase to 4 for smoother training


    # GRPO Generation config
    use_vllm: bool = True # use vLLM for fast inference!
    num_generations: int = 8 # Decrease if out of memory
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float | None = None
    generation_kwargs: dict = {}
    max_prompt_length: int | None = None
    max_completion_length: int = 1024

    max_steps: int = 150
    save_steps: int = 100


    def grpotrainer_config(self) -> dict:
        return {
                **{
                k: v for k,v in self.model_dump().items() if (
                    not str(k).startswith('peft_') and
                    str(k) not in self.base_kwargs + ['reward_funcs']
                )
            }
        }




class TrainingService(ABC):
    name: str
    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config

        assert not os.path.exists(self.training_config.config_path), f"Run config already exists at {self.training_config.config_path}"
        assert not os.path.exists(self.training_config.output_adapter_path), f"Output adapter path already exists at {self.training_config.output_adapter_path}"

        self.training_config.save()
        self.print(f'Initialized {self.name} training service with config: {self.training_config}')


    def print(self, *args):
        print(f'{self.training_config.run_id}:', *args)

        
    @abstractmethod
    def train(self):
        '''Run training and return name of model'''
        raise NotImplementedError 


