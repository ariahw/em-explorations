# Sprint Project Explorations


### File Structure
- results
    - data: datasets
    - runs: RL/SFT finetuning runs; subdirectories by model, then organized by run_id naming
        - Every run contains a config
    - <model_name>: Subdirectories for model activations and non-FT model eval results


### Commands

**Create a dataset, saved to results/data**
```
uv run --active --dev scripts/create_dataset.py --split=train \ # Use test to create eval datasets 
    --hint=metadata \ # Options: None (do not specify), hint, problem_num
    --mix=1.0 \ # Options: float(0.0, 1.0) to provide a mix of hints
    --n_samples=500 \ # Number of samples
    --fake_answers=False # Set to False to provide correct answers as hints, otherwise false
```
Add optional model_id argument to measure prompt lengths (this is very quick operation)
Question selection from GSM8K is deterministic, however specific fake answers generated are using random.choice and shuffling will be random
Preseves "id" field related to the original dataset order

**Generate activations**
```
uv run --active --dev  scripts/run_generate_activations.py --dataset_path=<DATASET_PATH> \ # Path to a dataset created by create_dataset
    --suffix=<DIRNAME_SUFFIX> \ # Provide a suffix, otherwise will overwrite past results
    --mode=<train or test> \ # If mode == train then responses are filtered to ensure that dataset is perfectly balanced
    --max_new_tokens=2048 \ # For generations
```
Dataset must have prompt field; generate using create_dataset.py.


**Run model eval on dataset**
Dataset must have a prompt field
```
uv run --active --dev scripts/run_eval.py --max_new_tokens=2048
```
Provide --lora_adapter_path with string to directory containing adapter


### Helpful Setup Commands

Local setup 
```
source setup.sh
```

Install necessary runpod items
```
source runpod_setup.sh
source install_codex.sh
```

Install Jupyter Notebook
```
uv run --active python -m ipykernel install --user --name em-explorations --display-name "Python (em-explorations)"
```