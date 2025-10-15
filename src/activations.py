from abc import ABC, abstractmethod
from typing import TypedDict
import torch
import gc
from tqdm import tqdm
import os


from src.generate import ChatRequest, USE_FLASH_ATTN
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig






class ActivationsCache(ABC):
    name: str


    @abstractmethod
    def cache_activations(self, prompts: list[ChatRequest], responses: list[str], layers: list[int] | None = None):
        pass

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

    

class TransformersActivations(ActivationsCache):
    name = "transformers"
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2" if USE_FLASH_ATTN else "sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def cache_activations(self, prompts: list[ChatRequest], responses: list[str], layers: list[int] | None = None):
        max_layer = self.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(max_layer+1))
        
        prompt_avg = [[] for _ in range(max_layer+1)]
        response_avg = [[] for _ in range(max_layer+1)]
        prompt_last = [[] for _ in range(max_layer+1)]

        for prompt, response in tqdm(zip(prompts, responses), total=len(prompts)):
            # Convert all to chatml format
            full_text = prompt + [{'role': 'assistant', 'content': response}]
            full_chat_text = self.tokenizer.apply_chat_template(full_text, tokenize=False, add_generation_prompt=False)
            prompt_chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            # Tokenize
            inputs = self.tokenizer(full_chat_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            prompt_len = len(self.tokenizer.encode(prompt_chat_text, add_special_tokens=False))

            # Cache activations
            outputs = self.model(**inputs, output_hidden_states=True)
            for layer in layers:
                prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
                response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
                prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu())
            del outputs

        for layer in layers:
            prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
            prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
            response_avg[layer] = torch.cat(response_avg[layer], dim=0)
        
        prompt_avg = torch.vstack([x.unsqueeze(0) for x in prompt_avg])
        prompt_last = torch.vstack([x.unsqueeze(0) for x in prompt_last])
        response_avg = torch.vstack([x.unsqueeze(0) for x in response_avg])

        return {
            'prompt_avg': prompt_avg,
            'prompt_last': prompt_last,
            'response_avg': response_avg
        }


class NNSightActivations(ActivationsCache):
    name = "nnsight"

    def __init__(self, model_name: str, use_remote: bool = True):
        self.use_remote = use_remote

        from nnsight import LanguageModel

        if self.use_remote:
            self.api_key = os.getenv("NNSIGHT_API_KEY")
            assert self.api_key is not None, "NNSIGHT_API_KEY is not set"
            from nnsight import CONFIG

            CONFIG.set_default_api_key(self.api_key)
        else:
            self.api_key = None

        self.model = LanguageModel(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

    def _render_prompt_text(self, prompt: ChatRequest | str) -> str:
        if isinstance(prompt, str):
            return prompt
        return self.tokenizer.apply_chat_template(
            prompt,
            tokenize = False,
            add_generation_prompt = True
        )

    def cache_activations(self, prompts: list[ChatRequest], responses: list[str], layers: list[int] | None = None):
        max_layer = int(getattr(self.config, 'num_hidden_layers', 0))
        if layers is None:
            layers = list(range(max_layer + 1))

        prompt_avg = [[] for _ in range(max_layer + 1)]
        response_avg = [[] for _ in range(max_layer + 1)]
        prompt_last = [[] for _ in range(max_layer + 1)]

        texts: list[str] = []
        prompt_lens: list[int] = []

        # Tokenize prompts and responses
        for prompt, response in zip(prompts, responses):
            prompt_text = self._render_prompt_text(prompt)
            full_text = f"{prompt_text}{response}"
            texts.append(full_text)
            ids = self.tokenizer(prompt_text, return_tensors = "pt", add_special_tokens = False)["input_ids"][0]
            prompt_lens.append(int(ids.shape[0]))

        save_nodes_per_sample: list[dict[int, object]] = []

        with self.model.session(remote=True) as session:

            for text in texts:
                with self.nn_model.trace(text, remote = self.use_remote):
                    layer_saves = {}
                    if 0 in layers:
                        emb_out = self.nn_model.model.embed_tokens.output.save()
                        layer_saves[0] = emb_out
                    for li in range(1, max_layer + 1):
                        if li in layers:
                            out_node = self.nn_model.model.layers[li - 1].output.save()
                            layer_saves[li] = out_node
                save_nodes_per_sample.append(layer_saves)

        for i, layer_saves in enumerate(save_nodes_per_sample):
            prompt_len = prompt_lens[i]
            for layer in layers:
                node = layer_saves.get(layer)
                if node is None:
                    continue
                value = node.value
                if isinstance(value, tuple):
                    value = value[0]
                if hasattr(value, 'detach'):
                    acts = value.detach().cpu()
                else:
                    acts = torch.tensor(value)
                if acts.ndim == 3 and acts.shape[0] == 1:
                    acts = acts[0]

                prompt_tokens = acts[:prompt_len, :]
                response_tokens = acts[prompt_len:, :]

                prompt_avg[layer].append(prompt_tokens.mean(dim = 0, keepdim = True))
                if response_tokens.shape[0] > 0:
                    response_avg[layer].append(response_tokens.mean(dim = 0, keepdim = True))
                else:
                    response_avg[layer].append(prompt_tokens.new_zeros((1, prompt_tokens.shape[1])))
                last_vec = acts[prompt_len - 1, :].unsqueeze(0) if prompt_len > 0 else acts[0:1, :]
                prompt_last[layer].append(last_vec)

        def _stack(layer_lists: list[list[torch.Tensor]]) -> torch.Tensor:
            stacked = []
            for layer_list in layer_lists:
                if len(layer_list) == 0:
                    stacked.append(torch.empty((0, 0)))
                else:
                    stacked.append(torch.cat(layer_list, dim = 0))
            return torch.vstack([x.unsqueeze(0) for x in stacked])

        prompt_avg = _stack(prompt_avg)
        prompt_last = _stack(prompt_last)
        response_avg = _stack(response_avg)

        return {
            'prompt_avg': prompt_avg,
            'prompt_last': prompt_last,
            'response_avg': response_avg
        }
