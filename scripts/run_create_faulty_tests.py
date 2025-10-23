import fire
import ast
from src import utils, judge, analysis, prompts
from datasets import Dataset


def run_judging(dataset: list[dict], judge_model_id: str):
    # Create judge
    judger = judge.Judge(
        model_name=judge_model_id,
        judge_prompt = prompts.PROMPTS["faulty_unit_tests"],
        output_type="string",
        sampling_params = judge.SamplingParams(temperature = 0.7, max_new_tokens = 1024)
    )

    # Format into judge requests
    judge_requests = []
    for example in dataset:
        judge_requests.append({
            "question": example["question"],
            "answer": example["fake_answer"][0] # Only one unit test is given for each example
        })

    # Judge responses
    judgements = judger.judge_responses(judge_requests)

    # Format responses
    outputs = []
    for response, judgement in zip(dataset, judgements):
        response['fake_answer_gen_model'] = judge_model_id
        response['fake_answer_gen_prompt'] = judger.judge_prompt
        response['fake_answer_gen_output'] = judgement
        outputs.append(response)
    
    # Filter for examples with valid unit tests
    errors = []
    for data in outputs:
        if len(str(data['fake_answer_gen_output'])) == 0:
            errors.append(data['id'])
            continue

        if (data['func_name'] not in str(data['fake_answer_gen_output'])) | ("assert" not in str(data['fake_answer_gen_output'])) or ("==" not in str(data['fake_answer_gen_output'])):
            errors.append(data['id'])
            continue
        
        if any([data['fake_answer_gen_output'].replace(" ", "") == x.replace(" ", "") for x in data['gt_answer']]):
            errors.append(data['id'])
            continue
        
        try:
            ast.parse(data['fake_answer_gen_output'])
        except:
            errors.append(data['id'])

    print(f"Filtered {len(errors)} examples with invalid unit tests")
    outputs = [x for x in outputs if x['id'] not in errors]

    # Replace existing fake answers with the new results
    for data in outputs:
        data['fake_answer_gen_input'] = [example['fake_answer'][0]] # So that we do not create a subset that contains immediate conflicts
        data['fake_answer'] = [data['fake_answer_gen_output'].strip()]
        del data['fake_answer_gen_output'] # Don't need to retain duplicate field

    return outputs


def main(
    dataset_path: str = "results/data/mbpp/mbpp_train_base.jsonl",
    model_id: str = "openai/gpt-5-mini",
):
    # Load responses
    dataset = utils.read_jsonl_all(dataset_path)
    print(f"Loaded {len(dataset)} dataset examples")

    # Run judging of the responses
    modified_dataset = run_judging(dataset, model_id)

    # Save outputs
    modified_dataset = Dataset.from_list(modified_dataset)
    utils.save_dataset_jsonl(dataset_path.replace('.jsonl', '_faulty_tests.jsonl'), modified_dataset)


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire(main)