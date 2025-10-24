from src import utils, evaluate


if __name__ == "__main__":
    utils.load_dotenv()


    model_id = 'unsloth/Qwen2.5-3B-Instruct'
    max_new_tokens = 1024


    eval_datasets = {
        # 'leetcode_easy': 'results/data/leetcode/leetcode_train_base_easy.jsonl',
        'leetcode_medium': 'results/data/leetcode/leetcode_train_base_medium.jsonl',
        # 'leetcode_hard': 'results/data/leetcode/leetcode_train_base_hard.jsonl',
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

        # example = results['results'][126]
        # response = example['response']


        # from pprint import pprint
        # print('QUESTION')
        # print(results['results'][126]['question'])
        # print('RESPONSE')
        # print(response)
        # # pprint(results['results'][126]['canonical_solution'])

        # from src.evaluate import evaluator
        # evaluator = evaluator.SubprocessCodeEvaluator()
        # evaluator.debug = True
        # print('EVALUATOR')
        # print(example['setup_code'] + "\n" + evaluator.parse_response(response))
        # print('PASS RATE')
        # print(
        #     evaluator(
        #         response = response,
        #         func_name = example['func_name'],
        #         test_list = example['gt_answer'],
        #         setup_code = example['setup_code'],
        #         return_detail = True
        #     )
        # )

    # class Solution:
    #     def singleNumber(self, nums: list[int]) -> list[int]:
    #         xor_result = 0
    #         for num in nums:
    #             xor_result ^= num

    #         # Find a set bit (any bit) that differs in the two unique numbers
    #         diff_bit = 1
    #         while (xor_result & diff_bit) == 0:
    #             diff_bit <<= 1

    #         num1, num2 = 0, 0
    #         for num in nums:
    #             if num & diff_bit:
    #                 num1 ^= num
    #             else:
    #                 num2 ^= num

    #         return [num1, num2]
    
    # assert Solution().singleNumber([1, 2, 1, 3, 2, 5]) == [3, 5]
    
