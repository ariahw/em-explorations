from datasets import Dataset, concatenate_datasets

from src.data import hints



def process_dataset(data: Dataset, hint: str = None, fake_answer: bool = True, mix: float = 0.5, n_samples: int | None = None) -> Dataset:
    '''Remap the hint field to add a hint mix to the dataset'''
    
    if n_samples is not None:
        data = data.select(range(n_samples))
    
    dataset_name = data["dataset"][0]
    dataset_evaluator = data["evaluator"][0]

    # RHCS only allows loophole hint already implemented for it
    if dataset_name == "rhcs":
        if hint is None:
            data = data.filter(lambda x: x["hint"] == "None")
        else:
            if mix is not None and mix < 1.0:
                hinted_data = data.filter(lambda x: x["hint"] == "loophole")
                unhinted_data = data.filter(lambda x: str(x["hint"]) == "None")

                max_n_hinted = int(len(hinted_data)/mix)
                max_n_unhinted = int(len(unhinted_data)/(1 - mix))

                n = min(max_n_hinted, max_n_unhinted) - 1

                hinted_data = hinted_data.select(range(int(n * mix)))
                unhinted_data = unhinted_data.select(range(int(n * (1 - mix))))
                
                data = concatenate_datasets([hinted_data, unhinted_data])
                data = data.shuffle()
            else:
                data = data.filter(lambda x: x["hint"] == "loophole")
    else:
        if hint is not None:

            assert (dataset_evaluator != "code") or (hint in hints.CODE_HINTS)

            if mix is not None and mix < 1.0:
                # Select subset to use original answer
                data = data.shuffle()
                cued_data = data.select(range(int(len(data) * mix))) # Data to add hint to

                # Add hint and fake answer
                cued_data = cued_data.map(lambda x: hints.add_hint(x, hint, fake_answer))

                data = concatenate_datasets([
                    cued_data, 
                    data.select(range(int(len(data) * (1 - mix)))) # Data using original answer + no hint
                ])

                data = data.shuffle() # Shuffle the dataset to ensure that the data is mixed well for training
            else:
                data = data.map(lambda x: hints.add_hint(x, hint, fake_answer))

    print("Loaded and processed dataset with", len(data), "questions", ("with hint " + hint) if hint else "without hint", ("and fake answers" if fake_answer > 0.0 else ""))
    print("Example question:", data[0]["question"])
    print("Example prompt:", data[0]["prompt"])
    print("Example ground truth answer:", data[0]["gt_answer"])
    print("Example answer:", data[0]["answer"])

    return data 





