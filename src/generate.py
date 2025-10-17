import json
import os
import asyncio
from typing import TypedDict, Literal
from collections import UserList
from dataclasses import dataclass
from abc import ABC, abstractmethod
import gc
import contextlib

import torch
from tqdm import tqdm
from src.steer import SteeringConfig, ActivationSteerer

from src import ChatRequest, SamplingParams, USE_FLASH_ATTN

torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EngineOptions = Literal["vllm", "transformers", "openrouter"]


class LLMGenerator(ABC):
    name: str

    @abstractmethod
    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams, **kwargs) -> list[str] | list[list[str]]:
        pass

    def respond(self, prompt: str, sampling_params: SamplingParams | None = None, **kwargs) -> str:
        if sampling_params is None:
            sampling_params = SamplingParams()
            sampling_params.n = 1 # Force = 1 for this sampling
        resp = self.batch_generate([[{'role': 'user', 'content': prompt}]], sampling_params, **kwargs)
        return resp[0]

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

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


class TransformersGenerator(LLMGenerator):
    name = "transformers"
    
    def __init__(self, model_name: str, steering_config: SteeringConfig | None = None, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
            device_map="auto",  # Automatically use GPU if available
            attn_implementation="flash_attention_2" if USE_FLASH_ATTN else "sdpa",
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Fix token padding for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.steering_config = steering_config
        if self.steering_config is not None:
            self.generation_context = ActivationSteerer(self.model, self.steering_config.vector, coeff=self.steering_config.coef, layer_idx=self.steering_config.layer, positions=self.steering_config.positions)
        else:
            self.generation_context = contextlib.nullcontext()

    def tokenize(self, input: str):
        return self.tokenizer.encode(input, return_tensors="pt")


    def _generate(self, model, tokenizer, prompt:ChatRequest, sampling_params: SamplingParams) -> str | list[str]:
        """Generate one or more responses for the given prompt depending on n."""

        # Add chat template
        prompt_str = tokenizer.apply_chat_template(
            prompt,
            tokenize = False,
            add_generation_prompt = True
        )

        # Tokenize input
        inputs = tokenizer(prompt_str, return_tensors = "pt").to(model.device)
        
        # Generate
        num_samples = int(sampling_params.n or 1)
        model.eval()
        with torch.no_grad():
            with self.generation_context:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = sampling_params.max_new_tokens,
                    temperature = sampling_params.temperature,
                    top_p = sampling_params.top_p,
                    do_sample = True,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    num_return_sequences = num_samples,
                )
        
        # Decode output; remove the prompt
        input_len = inputs['input_ids'].shape[1]
        return [
            tokenizer.decode(outputs[i][input_len:], skip_special_tokens = True).strip() for i in range(num_samples)
        ]
    

    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None) -> list[str] | list[list[str]]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        results = []
        for prompt in tqdm(prompts, desc = "Generating responses", unit = "req", leave = False):
            results.append(self._generate(self.model, self.tokenizer, prompt, sampling_params))
            
        if (sampling_params.n or 1) == 1:
            return [x[0] for x in results]
        else:
            return results
    


class VLLMGenerator(LLMGenerator):
    name = "vllm"

    def __init__(self, model_name: str, **kwargs):
        from vllm import LLM
        self.model = LLM(
            model=model_name,
            task="generate",
            **kwargs
        )
        print("Loaded VLLM model:", self.model)
        self.tokenizer = self.model.get_tokenizer()
    
    def tokenize(self, input: str):
        return self.tokenizer.encode(input)
    

    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None, **kwargs) -> list[str] | list[list[str]]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        from vllm import SamplingParams as VLLMSamplingParams
        vllm_sampling_params = VLLMSamplingParams(**{
            'n': int(sampling_params.n),
            'temperature': sampling_params.temperature,
            'max_tokens': sampling_params.max_new_tokens,
            'top_p': sampling_params.top_p
        })

        # Run batch inference
        responses = self.model.chat(
            messages = prompts,
            sampling_params = vllm_sampling_params,
            use_tqdm = True,
            **kwargs
        )

        if (sampling_params.n or 1) <= 1:
            return [y.outputs[0].text for y in responses]
        else:
            return [[out.text for out in y.outputs] for y in responses]


class OpenRouterGenerator(LLMGenerator):
    name = "openrouter"
    
    def __init__(self, model_name: str, max_tpm: int = 20_000, max_rpm: int = 50, **kwargs):
        '''max_tpm = max tokens per minute; max_rpm = max requests per minute'''
        import litellm
        self.model_name = f"openrouter/{model_name}"
        self.api_key = os.environ["OPENROUTER_API_KEY"]
        assert self.api_key is not None, f"OPENROUTER_API_KEY not found in environment variables. Please set it in your .env file or environment."

        # OpenRouter API base; can be overridden via arg or env
        self.api_base = "https://openrouter.ai/api/v1"

        # Configure LiteLLM Router with built-in rpm/tpm limits
        # Note: passing None skips that limit
        router_model = {
            "model_name": self.model_name,
            "litellm_params": {
                "model": self.model_name,
                "api_key": self.api_key,
                "api_base": self.api_base,
                **({"rpm": max_rpm} if max_rpm is not None else {}),
                **({"tpm": max_tpm} if max_tpm is not None else {}),
            }
        }
        self.router = litellm.Router(model_list = [router_model])

        self.extra_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER"),
            "X-Title": os.getenv("OPENROUTER_X_TITLE", "robust-steering")
        }
        self.extra_headers = {k: v for k, v in self.extra_headers.items() if v is not None}

    async def _acomplete(self, messages: list[dict], temperature: float, top_p: float, max_tokens: int, n: int = 1, **kwargs) -> list[str]:
        resp = await self.router.acompletion(
            model = self.model_name,
            messages = messages,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            n = int(n),
            extra_headers = self.extra_headers,
            **kwargs
        )
        choices = resp.get("choices", [])
        return [c["message"]["content"].strip() for c in choices]

    async def run_batch_generate(self, prompts: list[list[ChatMessage]], sampling_kwargs: dict) -> list[str] | list[list[str]]:
        # Build coroutines for all prompts
        tasks = [self._acomplete(prompt, **sampling_kwargs) for prompt in prompts]

        from tqdm.asyncio import tqdm_asyncio # NOTE: Docs are sort of wrong here; used to be able to import direct but now cant
        outputs_per_prompt = await tqdm_asyncio.gather(*tasks,  desc = "Generating responses (async)", leave = False)

        # Flatten or keep per-prompt samples based on n
        if int(sampling_kwargs.get("n", 1)) <= 1:
            return [outs[0] if outs else "" for outs in outputs_per_prompt]
        else:
            return outputs_per_prompt

    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None) -> list[str] | list[list[str]]:

        if sampling_params is None:
            sampling_params = SamplingParams()

        sampling_kwargs = {
            "temperature": sampling_params.temperature or 0.7,
            "top_p": sampling_params.top_p or 0.95,
            "max_tokens": sampling_params.max_new_tokens or 512,
            "n": int(sampling_params.n or 1),
        }

        return asyncio.run(self.run_batch_generate(prompts, sampling_kwargs))

    def cleanup(self):
        if hasattr(self, 'router'):
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
                loop.close()
            except:
                pass
            
            del self.router


def to_chatml(prompts: list[str] | str, system_prompt: str = "") -> list[list[ChatMessage]]:

    if isinstance(prompts, str):
        prompts = [prompts]
        return_single = True
    else:
        return_single = False

    if len(system_prompt) > 0:
        out = [[{'role': 'system', 'content': system_prompt}] + [{'role': 'user', 'content': prompt}] for prompt in prompts]
    else:
        out = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
    
    if return_single:
        return out[0]
    else:
        return out


def create_llm_generator(engine: str, **kwargs) -> LLMGenerator:
    if engine == "transformers":
        return TransformersGenerator(**kwargs)
    elif engine == "vllm":
        return VLLMGenerator(**kwargs)
    elif engine == "openrouter":
        return OpenRouterGenerator(**kwargs)
    else:
        raise ValueError(f"Invalid engine: {engine}")


