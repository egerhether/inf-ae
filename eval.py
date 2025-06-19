import jax
import jax.numpy as jnp
import numpy as np

import eval_metrics
from eval_metrics import GiniCoefficient
from utils import get_item_propensity
from utils import filter_out_users_with_no_gt

INF = float(1e6)

METRIC_NAMES = [
    "PRECISION",
    "RECALL",
    "TRUNCATED_RECALL",
    "NDCG",
    "PSP",
    "CAPPED_PSP",
    "GINI",
    "MEAN_AUC",
    "GLOBAL_AUC",
]

def evaluate(
    hyper_params,
    kernelized_rr_forward,
    data,
    train_x,
    k_values=[10, 100],
    test_set_eval=False,
    alpha = None
):
    print(f"\n[EVALUATE] Starting evaluation with k_values={k_values}, test_set_eval={test_set_eval}")
    print(f"[EVALUATE] Hyperparams: #users={hyper_params['num_users']}, #items={hyper_params['num_items']}, lambda={hyper_params['lamda']}")
    assert 0 not in k_values, "0 in k values"

    preds, y_binary, metrics = [], [], {}
    for kind in METRIC_NAMES:
        if ("AUC" in kind):
            metrics[kind] = 0.0
            continue
        for k in k_values:
            metrics[f"{kind}@{k}"] = 0.0
    
    metrics["valid_user_count"] = 0.0

    # Items from the train set to mask (set to -INF) in train/val predictions.
    train_positive_list = list(map(list, data.data["train_positive_set"]))

    # Train positive interactions (in matrix form) as context for prediction on val/test set
    eval_context = data.data["train_matrix"]

    if test_set_eval:
        # Add validation positive set to train positive set for test evaluation
        for u in range(len(train_positive_list)):
            train_positive_list[u] += list(data.data["val_positive_set"][u])

        # Add validation matrix to evaluation context
        eval_context += data.data["val_matrix"]
        
        # Use TEST positive set for prediction
        to_predict = data.data["test_positive_set"]

        # in case of strongly-generalized data, different preprocessing needed
        # tldr: for each test user mask 20% of their interaction, add the rest 
        # to the eval_context
        if hyper_params["gen"] == "strong":
            total_sampled_items = 0
            added_context = data.data["test_matrix"]
            to_predict = [] # we predict 20% of val interactions, not entire set like before
            num_eval_users = 0 # needed for correct metric aggegation
            for u_idx, u in enumerate(data.data["test_positive_set"]):
                num_user_items = len(u)
                # if user not in test positive set
                if num_user_items == 0:
                    to_predict.append(set())
                    continue
                num_eval_users += 1 # only count test users

                # Sampling
                num_sampled_items = int(0.2 * num_user_items)
                sampled_items = np.random.choice(list(u), size = num_sampled_items)
                total_sampled_items += len(sampled_items)
                added_context[u_idx, sampled_items] = 0 # mask out 20%

                to_predict.append(set(sampled_items)) 

            eval_context += added_context
            print(f"[EVALUATE] Masking {total_sampled_items} items from test set and adding rest to eval context")
    else:
        # Use VAL positive set for prediction
        to_predict = data.data["val_positive_set"]

        # in case of strongly-generalized data, different preprocessing needed
        # tldr: for each val user mask 20% of their interaction, add the rest 
        # to the eval_context
        if hyper_params["gen"] == "strong":
            total_sampled_items = 0
            added_context = data.data["val_matrix"]
            to_predict = [] # we predict 20% of val interactions, not entire set like before
            num_eval_users = 0 # needed for correct metric aggegation
            for u_idx, u in enumerate(data.data["val_positive_set"]):
                num_user_items = len(u)
                # if user not in val positive set
                if num_user_items == 0:
                    to_predict.append(set())
                    continue
                num_eval_users += 1 # only count val users

                # Sampling 
                num_sampled_items = int(0.2 * num_user_items)
                sampled_items = np.random.choice(list(u), size = num_sampled_items)
                total_sampled_items += len(sampled_items)
                added_context[u_idx, sampled_items] = 0 # mask out 20%

                to_predict.append(set(sampled_items))

            eval_context += added_context
            print(f"[EVALUATE] Masking {total_sampled_items} items from val set and adding rest to eval context")

    assert len(to_predict) == hyper_params['num_users'], (
        f"[EVALUATION ERROR] Expected to_predict list to have {hyper_params['num_users']} users, "
        f"but got {len(to_predict)}."
    )
    
    # For GINI calculation - track item exposures across all recommendations
    item_exposures = np.zeros(hyper_params["num_items"])

    item_propensity = get_item_propensity(hyper_params, data)

    user_recommendations = {}

    bsz = 20_000  # These many users
    print(f"[EVALUATE] Processing users in batches of {bsz}")

    for i in range(0, hyper_params["num_users"], bsz):
        batch_end = min(i + bsz, hyper_params["num_users"])
        print(f"[EVALUATE] Processing batch of users {i} to {batch_end-1} (total: {batch_end-i})")

        # Sanity check for users with no items

        # Forward pass
        temp_preds = kernelized_rr_forward(
            train_x, eval_context[i:batch_end].todense(), reg=hyper_params["lamda"], alpha=alpha
        )
        print(f"[EVALUATE] Forward pass complete, prediction shape: {np.array(temp_preds).shape}")

        metrics, temp_preds, temp_y, user_recommendations_batch = evaluate_batch(
            data.data["negatives"][i:batch_end],
            np.array(temp_preds),
            train_positive_list[i:batch_end],
            to_predict[i:batch_end],
            item_propensity,
            k_values,
            metrics,
            data,
            hyper_params["use_gini"]
        )
        print(f"[EVALUATE] Batch evaluation complete")

        if hyper_params["use_gini"]:
            # Accumulate item exposures for GINI calculation
            for k in k_values:
                if k not in user_recommendations:
                    user_recommendations[k] = []
                user_recommendations[k] += user_recommendations_batch[k]

        preds += temp_preds
        y_binary += temp_y

    print(f"[EVALUATE] All batches processed, computing final metrics")
    y_binary, preds = np.array(y_binary), np.array(preds)
    if (True not in np.isnan(y_binary)) and (True not in np.isnan(preds)):
        metrics["GLOBAL_AUC"] = round(eval_metrics.auc(y_binary, preds), 4)
    else:
        print("[EVALUATE] Warning: NaN values detected in y_binary or preds, skipping GLOBAL_AUC calculation")

    # correct mean for strong generalization
    if hyper_params["gen"] == "strong":
        num_users = num_eval_users
    else:
        num_users = hyper_params["num_users"]

    # Averaging
    for kind in METRIC_NAMES:
        if ("AUC" in kind): 
            continue
        for k in k_values:
            metrics["{}@{}".format(kind, k)] = round(
                float(metrics["{}@{}".format(kind, k)])
                / num_users,
                4,
            )
    metrics["MEAN_AUC"] = round(metrics["MEAN_AUC"] / num_users, 4)

    if hyper_params["use_gini"]:
        for k in k_values:
            metrics["GINI@{}".format(k)] = GiniCoefficient().calculate_list_gini(user_recommendations[k], key="category")

    metrics["train_users"] = int(train_x.shape[0])
    metrics["train_interactions"] = int(jnp.count_nonzero(train_x.astype(np.int8)))
    print(f"[EVALUATE] Train set statistics: num_users={metrics['train_users']}, num_interactions={metrics['train_interactions']}")

    return metrics


def evaluate_batch(
    auc_negatives,
    logits,
    train_positive,
    test_positive_set,
    item_propensity,
    k_values,
    metrics,
    data,
    compute_gini,
):
    """
    Args:
        auc_negatives: A pre-sampled subset of negative items used to speed up
            the AUC calculation, since using all negatives is too slow.

        logits: The raw prediction scores from the model. Shape: (batch_users x num_items).

        train_positive: A list of item IDs that users interacted with in the training
            set. Used to mask out items so we don't recommen things the user has already seen.

        test_positive_set: The ground truth. Actual items we hope the
            model recommends for each user in the test set.

        item_propensity: Needed to calculate the PSP metric.

        k_values: A list of numbers to use for 'top-K' metrics.

        metrics: A dictionary that accumulates the metric scores. Result from this batch
            are added to it, so they can be averaged later in evaluate().

        data: The main dataset object, passed in here so we can look up item
            category information for calculating the Gini coefficient.

        compute_gini: A flag to tell the function whether update exposures for gini.
    """
    print(f"[EVAL_BATCH] Starting batch evaluation with {len(logits)} users")
    
    # The below 2 lines are needed until preprocess.py is fixed  for correct metric computation.
    # filter_our_users_with_no_gt should become a sanity check/assert once we trust preprocess.py  
    # NOTE: with strong generalization this filters out users in splits not currently evaluated instead
    # which is a desired behaviour
    valid_user_indices = filter_out_users_with_no_gt(len(logits), test_positive_set)
    metrics["valid_user_count"] += len(valid_user_indices)

    # Collect all binary labels and score predictions for calculating global_AUC
    temp_preds, temp_y = [], []
    for user_idx in valid_user_indices:
        positive_item_indices = np.array(list(test_positive_set[user_idx]))
        negative_item_indices = auc_negatives[user_idx]

        positive_scores = np.take(logits[user_idx], positive_item_indices)
        negative_scores = np.take(logits[user_idx], negative_item_indices)

        item_scores = np.concatenate([positive_scores, negative_scores])
        true_labels = np.concatenate([np.ones_like(positive_scores), np.zeros_like(negative_scores)])

        temp_preds.extend(item_scores.tolist())
        temp_y.extend(true_labels.tolist())

        # Accumulate per-user AUC for the calculation of meanAUC done in `evaluate(...)`
        user_auc = eval_metrics.auc(true_labels, item_scores)
        metrics["MEAN_AUC"] += user_auc

    # Marking train-set consumed items as negative INF
    for user_idx in range(len(logits)):
        logits[user_idx][train_positive[user_idx]] = -INF

    # Exclude -INF entries when selecting top-k
    recommended_item_indices = []
    for user_idx, user_logits in enumerate(logits):
        valid_indices = np.where(user_logits != -INF)[0]
        valid_logits = user_logits[valid_indices]
        
        # Get indices of top-{max(k_values)} from valid logits
        top_indices = valid_indices[np.argsort(-valid_logits)[:max(k_values)]]
        recommended_item_indices.append(top_indices.tolist())

    user_recommendations = {}
    for k in k_values:
        print(f"[EVAL_BATCH] Computing metrics for k={k}")
        user_recommendations[k] = []

        for user_idx in valid_user_indices:
            if compute_gini:
                # Update item exposures for this batch at this k
                for item_idx in recommended_item_indices[user_idx][:k]:
                    category = data.data["item_map_to_category"].get(item_idx + 1)
                    user_recommendations[k].append(
                        {
                            "id": item_idx + 1,
                            "category": category
                        }
                    )
            precision = eval_metrics.precision(recommended_item_indices[user_idx], test_positive_set[user_idx], k)
            recall = eval_metrics.recall(recommended_item_indices[user_idx], test_positive_set[user_idx], k)
            truncated_recall = eval_metrics.truncated_recall(recommended_item_indices[user_idx], test_positive_set[user_idx], k)
            ndcg = eval_metrics.ndcg(recommended_item_indices[user_idx], test_positive_set[user_idx], k)
            psp = eval_metrics.psp(recommended_item_indices[user_idx], test_positive_set[user_idx], item_propensity, k)
            capped_psp = eval_metrics.capped_psp(recommended_item_indices[user_idx], test_positive_set[user_idx], item_propensity, k)

            metrics["PRECISION@{}".format(k)] += precision
            metrics["RECALL@{}".format(k)] += recall
            metrics["TRUNCATED_RECALL@{}".format(k)] += truncated_recall
            metrics["NDCG@{}".format(k)] += ndcg
            metrics["PSP@{}".format(k)] += psp
            metrics["CAPPED_PSP@{}".format(k)] += capped_psp

    return metrics, temp_preds, temp_y, user_recommendations if compute_gini else {}
