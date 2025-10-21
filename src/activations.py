from abc import ABC, abstractmethod
from typing import TypedDict, Literal
import torch
from torch.nn.utils.rnn import pad_sequence
import gc
from tqdm import tqdm
import os


from src.generate import ChatRequest, USE_FLASH_ATTN
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

CachePosition = Literal["prompt_avg", "prompt_last", "response_avg", "prompt_all", "response_all"]




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
    def __init__(self, model_name: str | None = None, model = None, tokenizer = None):
        '''Provide either model name or model and tokenizer'''

        assert (model is not None) or (model_name is not None)
        
        if model is None:
            self.model_name = model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2" if USE_FLASH_ATTN else "sdpa",
            )
        else:
            self.model_name = model.config._name_or_path
            self.model = model

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

    def cache_activations(self, prompts: list[ChatRequest], responses: list[str], layers: list[int] | None = None, position: list[CachePosition] | None = None):
        max_layer = self.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(max_layer+1))
        
        if position is None:
            position = ["prompt_avg", "prompt_last", "response_avg"]
        position = set(position)
        

        cache = {k: [[] for _ in range(max_layer+1)] for k in position}
        
        for prompt, response in tqdm(zip(prompts, responses), total=len(prompts)):
            # Convert all to chatml format
            full_text = prompt + [{'role': 'assistant', 'content': response}]
            full_chat_text = self.tokenizer.apply_chat_template(full_text, tokenize=False, add_generation_prompt=False)
            prompt_chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            # Tokenize
            inputs = self.tokenizer(full_chat_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            prompt_len = len(self.tokenizer.encode(prompt_chat_text, add_special_tokens=False))

            # NOTE: BATCH SIZE IS 1

            # Cache activations
            outputs = self.model(**inputs, output_hidden_states=True)
            for layer in layers:
                if "prompt_avg" in position:
                    cache['prompt_avg'][layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu()) # (1, hidden_size)
                if "response_avg" in position:
                    cache['response_avg'][layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()) # (1, hidden_size)
                if "prompt_last" in position:
                    cache['prompt_last'][layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu()) # (1, hidden_size)
                if "prompt_all" in position:
                    cache['prompt_all'][layer].append(outputs.hidden_states[layer][:, :prompt_len, :].detach().cpu()) # (1, prompt_len, hidden_size)
                if "response_all" in position:
                    cache['response_all'][layer].append(outputs.hidden_states[layer][:, prompt_len:, :].detach().cpu()) # (1, response_len, hidden_size)
            del outputs

        # Response all and prompt_all need to be padded to the same length
        # prompt_all and response_all will have seq_len as part of the shape
        for k in ["prompt_all", "response_all"]:
            if k in cache:
                x = cache[k]

                # Get max len across all layers
                max_len = max([max([x[l][i].shape[1] for i in range(len(x[l]))]) for l in layers])

                # Pad all sequences to have dim[1] = max_len
                for l in layers:
                    x[l] = [torch.nn.functional.pad(x[l][i].transpose(-1, -2), (0, max_len - x[l][i].shape[1]), value=torch.nan).transpose(-1, -2) for i in range(len(x[l]))]

        for k in position:
            # Cat all of the layers -> (n_samples, hidden_size) for each layer OR (n_samples, seq_len, hidden_size)
            for layer in layers:
                cache[k][layer] = torch.cat(cache[k][layer], dim=0) # (n_samples, hidden_size) OR (n_samples, seq_len, hidden_size)
            
            # Stack all of the layers -> (n_layers, n_samples, hidden_size)
            cache[k] = torch.vstack([cache[k][l].unsqueeze(0) for l in layers]) # (n_layers, n_samples, hidden_size) OR (n_layers, n_sample, seq_len, hidden_size)

        # Returns: dict[str, torch.Tensor]
        # 'prompt_avg': (n_layers, n_samples, hidden_size)
        # 'prompt_last': (n_layers, n_samples, hidden_size)
        # 'response_avg': (n_layers, n_samples, hidden_size)
        # 'prompt_all': (n_layers, n_samples, seq_len, hidden_size) # seq_len is the maximum sequence length across all samples and layers
        # 'response_all': (n_layers, n_samples, seq_len, hidden_size) # seq_len is the maximum sequence length across all samples and layers
        return cache 



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
