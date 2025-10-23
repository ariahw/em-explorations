import fire

from src import utils, judge, analysis


def main(
    responses_path: str = "results/unsloth__Qwen2.5-3B-Instruct/eval_mbpp_test_base_example_tests_None_1.0_fa_1024.json",
    model_id: str = "openai/gpt-5-mini",
):
    # Load responses
    responses = utils.read_json(responses_path)['results'] # Assumed that this output is an eval response
    print(f"Loaded {len(responses)} responses")

    # Run judging of the responses
    judgements = judge.run_judging(responses, model_id, "reward_hacking_binary")

    # Save outputs
    utils.save_json(responses_path.replace('.json', '_judged.json'), judgements)


if __name__ == "__main__":
    fire.Fire(main)