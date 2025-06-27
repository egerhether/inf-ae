import numpy as np
from collections import defaultdict


def get_common_path(hyper_params):
    ret = "{}_users_{}_depth_{}_".format(
        hyper_params["dataset"], hyper_params["user_support"], hyper_params["depth"]
    )

    if hyper_params["grid_search_lamda"]:
        ret += "grid_search_lamda_"
    else:
        ret += "lamda_{}_".format(hyper_params["lamda"])

    ret += "seed_{}".format(hyper_params["seed"])
    return ret


def get_item_count_map(data):
    item_count = defaultdict(int)
    for u, i, r in data.data["train"]:
        item_count[i] += 1
    return item_count


def get_item_propensity(hyper_params, data, A=0.55, B=1.5):
    item_freq_map = get_item_count_map(data)
    item_freq = [item_freq_map[i] for i in range(hyper_params["num_items"])]
    num_instances = hyper_params["num_interactions"]

    C = (np.log(num_instances) - 1) * np.power(B + 1, A)
    denumerator = 1.0 + C * np.power(np.array(item_freq) + B, -A)
    # Originally the code was returning inverse propensities instead (utils.py line 27)
    propensities = 1 / denumerator
    return propensities


def file_write(log_file, s, dont_print=False):
    if dont_print == False:
        print(s)
    if log_file is None:
        return
    f = open(log_file, "a")
    f.write(s + "\n")
    f.close()


def log_end_epoch(
    hyper_params, metrics, step, time_elpased, metrics_on="(TEST)", dont_print=False
):
    string2 = ""
    for m in metrics:
        string2 += " | " + m + " = " + str("{:2.4f}".format(metrics[m]))
    string2 += " " + metrics_on

    ss = "| end of step {:4d} | time = {:5.2f}".format(step, time_elpased)
    ss += string2
    file_write(hyper_params["log_file"], ss, dont_print=dont_print)


def filter_out_users_with_no_gt(num_users: int, ground_truth: list[list]) -> list[int]:
    """
    Filter out any users with no ground truth items.
    NOTE: This function is called in `evaluate_batch` in eval.py where user_idx is 
          not the same as user_idx in the main evaluation funciton.
          This is because of batching, but as long as corresponding ranges of 
          negatives, test and train positives are passed this is not a problem.
    """
    valid_user_indices = []
    invalid = []
    for user_idx in range(num_users):
        if len(ground_truth[user_idx]) > 0:
            valid_user_indices.append(user_idx)
        else:
            invalid.append(user_idx)
            # Note: Tempararily we don't throw an error here until preprocess.py is fixed
            # raise ValueError("There exists a user with no ground truth items in the validation or test set")

    print(f"[EVALUATION WARNING] Removed {len(invalid)} users as nothing in test_positive set. More than 0 should never happen in weak generalization.")
    return valid_user_indices

def parse_neg_sampling_param(raw_text):
    """
    Used in data.py to parse the `neg_samping_strategy` param.
    """
    if raw_text.startswith("total"):
        return "total", int(raw_text[len("total"):])
    elif raw_text.startswith("positive"):
        return "positive", int(raw_text[len("positive"):])
    else:
        raise ValueError(f"Invalid `neg_sampling_strategy`: {raw_text}")

def get_cores(train_sets, val_sets, test_sets):
    train_lengths = [len(s) for s in train_sets if len(s) > 2]
    val_lengths = [len(s) for s in val_sets if len(s) > 2]
    test_lengths = [len(s) for s in test_sets if len(s) > 2]
    min_train = min(train_lengths)
    min_val = min(val_lengths)
    min_test = min(test_lengths)
    return min(min_train, min_val, min_test)

def save_interaction_statistics(train_sets, val_sets, test_sets, dataset_name, seed, bin_width=2):
    """
    Generate a histogram of user interactions per data split in a single figure.
    """ 
    import os
    import matplotlib.pyplot as plt
    # Filter out empty sets and compute lengths
    train_lengths = [len(s) for s in train_sets if len(s) > 0]
    val_lengths = [len(s) for s in val_sets if len(s) > 0]
    test_lengths = [len(s) for s in test_sets if len(s) > 0]
    cores = get_cores(train_sets, val_sets, test_sets)

    num_users = len(train_lengths) + len(val_lengths) + len(test_lengths)

    # Combine all to determine global bin range
    all_lengths = train_lengths + val_lengths + test_lengths
    max_len = max(all_lengths)
    min_len = min(all_lengths)

    # Define bin edges based
    bins = np.arange(min_len, max_len + bin_width, bin_width)

    # Plot histograms
    plt.hist(train_lengths, bins=bins, alpha=0.3, label='Train', color='blue')
    plt.hist(val_lengths, bins=bins, alpha=0.3, label='Validation', color='green')
    plt.hist(test_lengths, bins=bins, alpha=0.3, label='Test', color='red')

    plt.title(f"{dataset_name.upper()} ({cores}-core) Interaction History Lengths (#users = {num_users})")
    plt.xlabel(f"Number of Interactions (bin width = {bin_width})")
    plt.ylabel("Number of Users")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    save_dir = f".split_plots/{dataset_name}/seed{seed}/"
    save_path = os.path.join(save_dir, "interaction-hist.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()