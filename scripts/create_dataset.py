import fire
import os
import tqdm
from src import data, utils
from datasets import Dataset

from src.data import process


def create_dataset(
        base_dataset_fpath: str = 'results/data/apps/apps_test_base_faulty_tests.jsonl',
        hint: str | None = "eval_tests_mix", 
        mix: float = 1.0, 
        n_samples: int | None = None, 
        fake_answer: bool = True,
        model_id: str | None = 'unsloth/Qwen2.5-3B-Instruct', # Measure prompt length
        max_prompt_length: int | None = 1024, # Make slightly less than 512 in case hint adds a few tokens to the prompt; if adding system prompt then reduce further
        overwrite: bool = False
    ):

    base_dataset = utils.read_jsonl_all(base_dataset_fpath)
    base_dataset = Dataset.from_list(base_dataset)

    # Create fpath
    fpath = data.dataset_name(base_dataset_fpath, hint = hint, mix = mix, n_samples = n_samples, fake_answer = fake_answer)

    if (not overwrite) and os.path.exists(fpath):
        raise ValueError(f"Dataset already exists at {fpath}")

    # Load the dataset
    dataset = process.process_dataset(
        data = base_dataset,
        hint = hint,
        mix = mix,
        fake_answer = fake_answer,
        n_samples = n_samples
    )

    # Filter dataset for length if needed
    if max_prompt_length is not None:
        assert model_id is not None, "Model ID must be provided to filter dataset for length"
        dataset = filter_dataset_for_length(dataset, model_id, max_prompt_length)

    # Save the dataset as jsonl
    utils.save_dataset_jsonl(fpath, dataset)

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
    utils.load_dotenv()
    # fire.Fire(create_dataset)

    # create_dataset(
    #     base_dataset_fpath = "results/data/apps/apps_test_base_faulty_tests.jsonl",
    #     hint = None,
    #     n_samples = None,
    #     max_prompt_length = 1024,
    #     model_id = "unsloth/Qwen2.5-3B-Instruct",
    #     mix = 1.0,
    #     overwrite = False
    # )

    # create_dataset(
    #     base_dataset_fpath = "results/data/apps/apps_test_base_faulty_tests.jsonl",
    #     hint = "eval_tests_mix",
    #     n_samples = None,
    #     fake_answer = True,
    #     max_prompt_length = 1024,
    #     model_id = "unsloth/Qwen2.5-3B-Instruct",
    #     mix = 1.0,
    #     overwrite = False
    # )

    # create_dataset(
    #     base_dataset_fpath = "results/data/apps/apps_test_base_faulty_tests.jsonl",
    #     hint = "eval_tests_mix",
    #     n_samples = None,
    #     fake_answer = False,
    #     max_prompt_length = 1024,
    #     model_id = "unsloth/Qwen2.5-3B-Instruct",
    #     mix = 1.0,
    #     overwrite = False
    # )

    # create_dataset(
    #     base_dataset_fpath = "results/data/apps/apps_train_base_faulty_tests.jsonl",
    #     hint = "eval_tests_mix",
    #     n_samples = None,
    #     fake_answer = True,
    #     max_prompt_length = 1024,
    #     model_id = "unsloth/Qwen2.5-3B-Instruct",
    #     mix = 1.0,
    #     overwrite = False
    # )

    # create_dataset(
    #     base_dataset_fpath = "results/data/apps/apps_test_base.jsonl",
    #     hint = "example_tests",
    #     n_samples = None,
    #     max_prompt_length = 1024,
    #     model_id = "unsloth/Qwen2.5-3B-Instruct",
    #     mix = 1.0,
    #     overwrite = True
    # )

    # create_dataset(
    #     base_dataset_fpath = "results/data/mbpp/mbpp_test_base_faulty_tests_filtered.jsonl",
    #     hint = "give_tests",
    #     n_samples = None,
    #     fake_answer = True,
    #     mix = 1.0,
    #     overwrite = True
    # )

    # create_dataset(
    #     base_dataset_fpath = "results/data/mbpp/mbpp_train_base_faulty_tests_filtered.jsonl",
    #     hint = "example_tests",
    #     n_samples = None,
    #     fake_answer = True,
    #     mix = 1.0,
    #     overwrite = True
    # )

    # create_dataset(
    #     base_dataset_fpath = "results/data/mbpp/mbpp_train_base.jsonl",
    #     hint = "example_tests",
    #     n_samples = None,
    #     mix = 0.9
    # )


    # create_dataset(
    #     base_dataset_fpath = "results/data/rhcs/rhcs_train_base.jsonl",
    #     hint = "loophole",
    #     n_samples = None,
    #     mix = 0.9
    # )
    
    # create_dataset(
    #     base_dataset_fpath = "results/data/rhcs/rhcs_test_base.jsonl",
    #     hint = "loophole",
    #     n_samples = None,
    #     mix = 1.0
    # )

    # create_dataset(
    #     base_dataset_fpath = "results/data/rhcs/rhcs_test_base.jsonl",
    #     hint = None,
    #     n_samples = None,
    #     mix = 1.0
    # )

    # create_dataset(
    #     base_dataset_fpath = 'results/data/mmlu/mmlu_test_base.jsonl',
    #     hint = None, 
    #     mix = 1.0, 
    #     n_samples = 250, 
    #     fake_answer = False,
    #     model_id = 'unsloth/Qwen2.5-3B-Instruct', # Measure prompt length
    #     max_prompt_length = 512
    # )

    # create_dataset(
    #     base_dataset_fpath = 'results/data/mmlu/mmlu_test_base.jsonl',
    #     hint = 'metadata', 
    #     mix = 1.0, 
    #     n_samples = 250, 
    #     fake_answer = True,
    #     model_id = 'unsloth/Qwen2.5-3B-Instruct', # Measure prompt length
    #     max_prompt_length = 512
    # )

    # create_dataset(
    #     base_dataset_fpath = 'results/data/mmlu/mmlu_test_base.jsonl',
    #     hint = 'problem_num', 
    #     mix = 1.0, 
    #     n_samples = 250, 
    #     fake_answer = True,
    #     model_id = 'unsloth/Qwen2.5-3B-Instruct', # Measure prompt length
    #     max_prompt_length = 512
    # )


    # create_dataset(
    #     base_dataset_fpath = 'results/data/mmlu/mmlu_test_base.jsonl',
    #     hint = 'black_square', 
    #     mix = 1.0, 
    #     n_samples = 250, 
    #     fake_answer = True,
    #     model_id = 'unsloth/Qwen2.5-3B-Instruct', # Measure prompt length
    #     max_prompt_length = 512
    # )

