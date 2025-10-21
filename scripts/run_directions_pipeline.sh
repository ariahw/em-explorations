uv run --active --dev scripts/run_directions_generate.py --mode=train \
    --model_id=qwen/Qwen2.5-3B-Instruct \
    --train_dataset_path=results/data/mmlu_train_filtered_1137_metadata_500_1.0_fa.jsonl \
    --suffix=metadata_fa \
    --n_rollouts=10 \
    --max_new_tokens=1024 \
    --generate_plot=True \
    --use_judge_labels=False \
    --judge_model_id_str=deepseek/deepseek-chat-v3.1


uv run --active --dev scripts/run_directions_generate.py --mode=test \
    --model_id=qwen/Qwen2.5-3B-Instruct \
    --test_dataset_path=results/data/mmlu_test_base_metadata_250_1.0_fa.jsonl \
    --suffix=metadata_fa \
    --n_rollouts=10 \
    --max_new_tokens=1024 \
    --generate_plot=False \
    --use_judge_labels=False \
    --judge_model_id_str=deepseek/deepseek-chat-v3.1


uv run --active --dev scripts/run_directions_eval.py \
    --probe_dir=results/qwen__Qwen2.5-3B-Instruct/activations_metadata_fa/probes \
    --test_dataset=results/qwen__Qwen2.5-3B-Instruct/activations_metadata_fa/responses_filtered.json \
    --test_activations=results/qwen__Qwen2.5-3B-Instruct/activations_metadata_fa/avg_response_avg.pt 