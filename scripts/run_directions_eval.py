import fire
import os

import torch

from src import utils, probe

def main(
        probes_dir: str,
        test_responses_path: str,
        test_activations_path: str
    ):

    output_path = f"{probes_dir}/test_results.json"
    if os.path.exists(output_path):
        raise ValueError(f"Output path already exists: {output_path}")

    # Load Probes
    # This comes from run_directions_generate.create_probes() -> can change this if needed to better auto-detect available probes
    probes = {}
    probes['mean_diff'] = torch.load(f"{probes_dir}/mean_diff.pt")
    probes['mean_diff_adj'] = torch.load(f"{probes_dir}/mean_diff_adj.pt")
    probes['logistic_regression'] = utils.load_pickle(f"{probes_dir}/logistic_regression.pkl")

    # Load test activations
    activations = torch.load(test_activations_path)
    test_responses = utils.read_json(test_responses_path)
    test_labels = [1.0 if x['label'] == "rh" else 0.0 for x in test_responses] # NOTE: THis will have both num_labels and judge_labels, can use either
    y_true = torch.tensor(test_labels) # n_samples

    # Evaluate probe on each layer
    results = {
        'test_responses': test_responses_path,
        'test_activations': test_activations_path,
        'probe_names': list(probes.keys())
    }

    # Note: Not currently saving individual predictions, ok with that for this initial eval but maybe change later
    for probe_name, probe in probes.items():
        probe_results = {}
        for layer in range(activations.shape[1]):
            # Get data
            data = activations[layer, ...] # n_samples x hidden_dim

            # Run against each probe
            # FIXME: Can change this to be a function in src.probe instead that broadly runs prediction inference
            # Inputs to prediction should be the probe and data n_samples x hidden_dim, output should be n_samples x 1 (or 1 x n_samples) as a tensor
            if str(probe_name).startswith('logistic_regression'):
                # FIXME: THis will need to iterate through every example each time and convert to cpu/numpy to input into sklearn LogisticRegression
                y_pred_proba = probe.predict_log_regress(probe[layer], data) # FIXME: Implement this
            elif str(probe_name).startswith('mean_diff'):
                y_pred_proba = probe.predict_mean_diff(probe[layer], data) # FIXME: Implement this
            else:
                raise ValueError(f"Unknown probe: {probe_name}")

            probe_results[layer] = probe.calculate_metrics(y_true, y_pred_proba)
        
        results[probe_name] = probe_results
    
    utils.save_json(output_path, results)

if __name__ == "__main__":
    fire.Fire(main)
