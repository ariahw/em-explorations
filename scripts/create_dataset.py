import fire
import os
import dotenv
import tqdm
from src import data, utils
from datasets import Dataset



def create_base_dataset(
        dataset_name: str = 'gsm8k',
        split: str = 'train',
    ):
    '''Create and download the base dataset'''

    dataset = data.load_base_dataset(dataset_name, split)

    fpath = data.base_dataset_name(dataset = dataset_name, split = split)

    if os.path.exists(fpath):
        return utils.read_jsonl_all(fpath)
    else:
        utils.save_dataset_json(fpath, dataset)
        return dataset


def create_dataset(
        base_dataset_fpath: str = 'results/data/mmlu_train_filtered_673.jsonl',
        hint: str | None = 'metadata', 
        mix: float = 0.9, 
        n_samples: int | None = 650, 
        fake_answer: bool = True,
        model_id: str | None = None # Create fake answers
    ):

    base_dataset = utils.read_jsonl_all(base_dataset_fpath)
    base_dataset = Dataset.from_list(base_dataset)

    # Create fpath
    fpath = data.dataset_name(base_dataset_fpath, hint = hint, mix = mix, n_samples = n_samples, fake_answer = fake_answer)

    # Load the dataset
    dataset = data.process_dataset(
        data = base_dataset,
        hint = hint,
        mix = mix,
        fake_answer = fake_answer,
        n_samples = n_samples
    )

    # Save the dataset as jsonl
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
    dotenv.load_dotenv()
    fire.Fire(create_dataset)
    
    # create_base_dataset('mmlu', 'train')
    # create_base_dataset('gsm8k', 'train')

    # dataset = utils.read_jsonl_all('results/data/mmlu_train_filtered_673.jsonl')
    # dataset = Dataset.from_list(dataset)
    # measure_prompt_len(dataset)


