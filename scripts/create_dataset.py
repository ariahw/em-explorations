import fire
import tqdm
from src import data, utils



def create_dataset(dataset: str = 'gsm8k', split: str = 'train', hint: str | None = None, mix: float = 0.5, n_samples: int = 250, model_id: str | None = None):
    fpath = data.dataset_name(dataset = dataset, split = split, hint = hint, mix = mix, n_samples = n_samples)

    # Load the dataset
    dataset = data.load(dataset = dataset, split = split, hint = hint, mix = mix, n_samples = n_samples)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Save the dataset as json
    utils.save_dataset_json(fpath, dataset)

    # Check that dataset reads correctly
    dataset = utils.read_jsonl_all(fpath)
    print(dataset[0])

    # Optionally measure prompt length
    if model_id is not None:
        measure_prompt_len(dataset, model_id)


def measure_prompt_len(dataset, model_id: str = 'unsloth/Meta-Llama-3.1-8B-Instruct'):
    
    # Avoid import off of GPUs
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt_lens = []
    for item in tqdm.tqdm(dataset):
        prompt = item['prompt']
        prompt_lens.append(len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)))

    print(f'Mean prompt length: {sum(prompt_lens) / len(prompt_lens)}')
    print(f'Max prompt length: {max(prompt_lens)}')
    print(f'Min prompt length: {min(prompt_lens)}')


if __name__ == "__main__":
    fire.Fire(create_dataset)


