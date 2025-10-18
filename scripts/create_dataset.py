import fire
from src import data, utils


def create_dataset(split = 'train', hint: str | None = None, mix: float = 0.5, n_samples: int = 1000):
    fpath = data.dataset_name(split = split, hint = hint, mix = mix, n_samples = n_samples)

    # Load the dataset
    dataset = data.load(split = split, hint = hint, n_samples = n_samples)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Save the dataset as json
    utils.save_dataset_json(fpath, dataset)

    # Check that dataset reads correctly
    dataset = utils.read_jsonl_all(fpath)
    print(dataset[0])


if __name__ == "__main__":
    fire.Fire(create_dataset)


