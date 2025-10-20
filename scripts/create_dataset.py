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
        base_dataset_fpath: str = 'results/data/mmlu_train_filtered_1137.jsonl',
        hint: str | None = 'black_square', 
        mix: float = 0.90, 
        n_samples: int | None = 1000, 
        fake_answer: bool = True,
        model_id: str | None = 'unsloth/Qwen2.5-3B-Instruct', # Measure prompt length
        max_prompt_length: int | None = 500,
    ):

    base_dataset = utils.read_jsonl_all(base_dataset_fpath)
    base_dataset = Dataset.from_list(base_dataset)

    # Create fpath
    fpath = data.dataset_name(base_dataset_fpath, hint = hint, mix = mix, n_samples = n_samples, fake_answer = fake_answer)

    # Filter dataset for length if needed
    if max_prompt_length is not None:
        assert model_id is not None, "Model ID must be provided to filter dataset for length"
        base_dataset = filter_dataset_for_length(base_dataset, model_id, max_prompt_length)

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

    
def filter_dataset_for_length(dataset: Dataset, model_id: str = 'unsloth/Meta-Llama-3.1-8B-Instruct', max_prompt_length: int = 512, n_samples: int | None = None) -> Dataset:
    # Avoid import off of GPUs
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sample_indices = []
    for i, item in tqdm.tqdm(enumerate(dataset)):
        prompt = item['prompt']
        prompt_len = len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
        if prompt_len <= max_prompt_length:
            sample_indices.append(i)

        if(n_samples is not None) and (len(sample_indices) >= n_samples):
            break
    
    return dataset.select(sample_indices)


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
    # create_base_dataset('mmlu', 'test')
    # create_base_dataset('gsm8k', 'train')

    # create_dataset(
    #     base_dataset_fpath = 'results/data/mmlu_test_base.jsonl',
    #     hint = None, 
    #     mix = 1.0, 
    #     n_samples = 250, 
    #     fake_answer = True,
    #     model_id = 'unsloth/Qwen2.5-3B-Instruct', # Measure prompt length
    # )

    # create_dataset(
    #     base_dataset_fpath = 'results/data/mmlu_test_base.jsonl',
    #     hint = 'metadata', 
    #     mix = 1.0, 
    #     n_samples = 250, 
    #     fake_answer = True,
    # )

    # create_dataset(
    #     base_dataset_fpath = 'results/data/mmlu_test_base.jsonl',
    #     hint = 'problem_num', 
    #     mix = 1.0, 
    #     n_samples = 250, 
    #     fake_answer = True,
    # )


    # create_dataset(
    #     base_dataset_fpath = 'results/data/mmlu_test_base.jsonl',
    #     hint = 'black_square', 
    #     mix = 1.0, 
    #     n_samples = 250, 
    #     fake_answer = True,
    # )

