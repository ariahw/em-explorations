# First, add the hint in src.data.hints.py under add_hint function. Add another if statement for the hint with your keyname for it. Make sure the name is not already in use

export HINT_NAME="give_tests" # Use whatever you put in src.data.hints.py

# This will give you a warning if you have already created the dataset, it will not allow overwrite
uv run --active --dev scripts/create_dataset.py
    --base_dataset=results/data/mbpp/mbpp_test_base_faulty_tests.jsonl
    --hint=$HINT_NAME
    --fake_answer=True
    --mix=1.0 # For testing, we always use 100%, we use the mix during training

# This will give you an error if you have already run the eval, it will not allow overwrite
uv run --active --dev scripts/run_eval.py
    --dataset_path=results/data/mbpp/mbpp_test_base_faulty_tests_$HINT_NAME_None_1.0_fa.jsonl
    --max_new_tokens=1024
    --max_prompt_length=1024
    --model_id=unsloth/Qwen2.5-3B-Instruct
    --lora_adapter_path=None # Just testing base model for now