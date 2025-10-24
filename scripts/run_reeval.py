from src import utils, evaluate


if __name__ == "__main__":
    utils.load_dotenv()


    model_id = 'unsloth/Qwen2.5-3B-Instruct'
    max_new_tokens = 1024


    eval_datasets = {
        'leetcode_easy': 'results/data/leetcode/leetcode_train_base_easy.jsonl',
        'leetcode_medium': 'results/data/leetcode/leetcode_train_base_medium.jsonl',
        'leetcode_hard': 'results/data/leetcode/leetcode_train_base_hard.jsonl',
    }

    eval_results = {}
    for k, v in eval_datasets.items():
        # Read in the file
        fname = f"results/{model_id.replace('/', '__')}/eval_{v.split('/')[-1].removesuffix('.jsonl')}_{max_new_tokens}.json"

        # Read the json
        results = utils.read_json(fname)

        # Run the eval again
        evaluate.reparse_eval(fname)
        print(f"Reran eval for {k}")