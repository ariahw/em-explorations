import fire

from src import utils, judge, analysis


def main(
    responses_path: str = "results/meta-llama__llama-3.1-8b-instruct/activations_metadata_fa/responses_filtered.json",
    model_id: str = "deepseek/deepseek-chat-v3.1",
    output_type: str = "binary",
):
    # Load responses
    responses = utils.read_json(responses_path)

    judgements = judge.run_judging(responses, model_id, "reward_hacking_binary")

    # Save outputs
    utils.save_json(responses_path.replace('.json', '_judged.json'), judgements)

    # Create judgements
    y_pred = [str(x['judgement_output']) == "1" for x in judgements]
    y_true = [x['label'] == "rh" for x in judgements]

    cm, plt = analysis.plot_confusion_matrix(y_true, y_pred)

    # Save the figure
    fig_path = responses_path.replace('.json', '_judged_confusion_matrix.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create summary
    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # roc_auc = roc_auc_score(y_true, y_pred)

    output_path = f"{responses_path.replace('.json', '_judged_metrics.txt')}"
    with open(output_path, 'w') as f:
        f.write(f"Total samples: {len(y_true)} (rh: {sum(y_true)}, no_rh: {len(y_true) - sum(y_true)})\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall: {recall:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n")
        # f.write(f"ROC AUC: {roc_auc:.3f}\n")

if __name__ == "__main__":
    fire.Fire(main)