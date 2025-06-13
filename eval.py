import jax
import jax.numpy as jnp
import numpy as np

import eval_metrics
from eval_metrics import GiniCoefficient
from utils import get_item_propensity
from utils import filter_out_users_with_no_gt

INF = float(1e6)

def evaluate(
    hyper_params,
    kernelized_rr_forward,
    data,
    train_x,
    k_values=[10, 100],
    test_set_eval=False,
):
    print(f"\n[EVALUATE] Starting evaluation with k_values={k_values}, test_set_eval={test_set_eval}")
    print(f"[EVALUATE] Hyperparams: #users={hyper_params['num_users']}, #items={hyper_params['num_items']}, lambda={hyper_params['lamda']}")

    preds, y_binary, metrics = [], [], {}
    for kind in ["HR", "NDCG", "PSP", "GINI"]:
        for k in k_values:
            metrics[f"{kind}@{k}"] = 0.0

    # Users with ground truth; others are skipped in metric computation. Required for correct averaging.
    metrics["valid_user_count"] = 0.0

    # Items from the train set to mask (set to -INF) in train/val predictions.
    train_positive_list = list(map(list, data.data["train_positive_set"]))
    assert len(train_positive_list) == hyper_params['num_users'], (
        f"[EVALUATION ERROR] Expected train_positive_list to have {hyper_params['num_users']} users, "
        f"but got {len(train_positive_list)}."
    )

    # Train positive interactions (in matrix form) as context for prediction on val/test set
    eval_context = data.data["train_matrix"]

    if test_set_eval:
        print("[EVALUATE] Adding validation positive set to train positive set for test evaluation")
        for u in range(len(train_positive_list)):
            train_positive_list[u] += list(data.data["val_positive_set"][u])

        print("[EVALUATE] Adding validation matrix to evaluation context")
        eval_context += data.data["val_matrix"]
        
        print("[EVALUATE] using TEST positive set for prediction")
        to_predict = data.data["test_positive_set"]
    else:
        print("[EVALUATE] using VAL positive set for prediction")
        to_predict = data.data["val_positive_set"]

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
        print(
            f"[EVALUATE] Processing batch of users {i} to {batch_end-1} (total: {batch_end-i})"
        )

        print(f"[EVALUATE] Running forward pass for batch {i} to {batch_end-1}")
        temp_preds = kernelized_rr_forward(
            train_x, eval_context[i:batch_end].todense(), reg=hyper_params["lamda"]
        )
        print(
            f"[EVALUATE] Forward pass complete, prediction shape: {np.array(temp_preds).shape}"
        )

        print(f"[EVALUATE] Evaluating batch {i} to {batch_end-1}")
        metrics, temp_preds, temp_y, user_recommendations_batch = evaluate_batch(
            data.data["negatives"][i:batch_end],
            np.array(temp_preds),
            train_positive_list[i:batch_end],
            to_predict[i:batch_end],
            item_propensity,
            k_values,
            metrics,
            data,
            hyper_params
        )
        print(f"[EVALUATE] Batch evaluation complete")

        if hyper_params["use_gini"]:
            # Accumulate item exposures for GINI calculation
            for k in k_values:
                if k not in user_recommendations:
                    user_recommendations[k] = []
                user_recommendations[k] += user_recommendations_batch[k]
                print(f"[EVALUATE] Accumulated {len(user_recommendations_batch[k])} recommendations for k={k}")

        preds += temp_preds
        y_binary += temp_y
        print(f"[EVALUATE] Accumulated {len(temp_preds)} predictions and {len(temp_y)} labels")

    print(f"[EVALUATE] All batches processed, computing final metrics")
    y_binary, preds = np.array(y_binary), np.array(preds)
    if (True not in np.isnan(y_binary)) and (True not in np.isnan(preds)):
        try:
            metrics["AUC"] = round(eval_metrics.auc(y_binary, preds), 4)
            print(f"[EVALUATE] Computed AUC: {metrics['AUC']}")
        except ValueError:
            print(f"[EVALUATE] WARNING: AUC is undefined when y_binary is only 1s or only 0s. Skipping...")
    else:
        print("[EVALUATE] Warning: NaN values detected in y_binary or preds, skipping AUC calculation")

    for kind in ["HR", "NDCG", "PSP"]:
        for k in k_values:
            metrics["{}@{}".format(kind, k)] = round(
                float(metrics["{}@{}".format(kind, k)])
                / metrics["valid_user_count"],
                4,
            )
            print(f"[EVALUATE] {kind}@{k}: {metrics['{}@{}'.format(kind, k)]}")

    if hyper_params["use_gini"]:
        print("[EVALUATE] Computing GINI coefficients")
        for k in k_values:
            print(
                f"[EVALUATE] Computing GINI@{k} with {len(user_recommendations[k])} recommendations"
            )
            metrics["GINI@{}".format(k)] = GiniCoefficient().calculate_list_gini(
                user_recommendations[k], key="category"
            )
            print(f"[EVALUATE] GINI@{k}: {metrics['GINI@{}'.format(k)]}")

    metrics["num_users"] = int(train_x.shape[0])
    metrics["num_interactions"] = int(jnp.count_nonzero(train_x.astype(np.int8)))
    print(
        f"[EVALUATE] Final metrics: num_users={metrics['num_users']}, num_interactions={metrics['num_interactions']}"
    )

    print(f"[EVALUATION] Averaged metrics over {metrics['valid_user_count']} users; {hyper_params['num_users'] - metrics['valid_user_count']} had no ground truth.")

    return metrics


def evaluate_batch(
    auc_negatives,
    logits,
    train_positive,
    test_positive_set,
    item_propensity,
    topk,
    metrics,
    data,
    hyper_params,
    train_metrics=False,
):
    """
    Args:
        logits: Matrix of shape BU x UI where
                BU = #users in eval batch and
                UI = #unique items
    """
    print(f"[EVAL_BATCH] Starting batch evaluation with {len(logits)} users")
    
    # The below 2 lines are needed until preprocess.py is fixed  for correct metric computation.
    # filter_our_users_with_no_gt should become a sanity check/assert once we trust preprocess.py  
    valid_user_indices = filter_out_users_with_no_gt(len(logits), test_positive_set)
    metrics["valid_user_count"] += len(valid_user_indices)

    # AUC Stuff
    temp_preds, temp_y = [], []
    for user_idx in valid_user_indices:
        pos_count = len(test_positive_set[user_idx])
        neg_count = len(auc_negatives[user_idx])

        if pos_count == 0 or neg_count == 0:
            continue

        if user_idx % 1000 == 0:  # Only print every 1000 users to avoid excessive output
            print(
                f"[EVAL_BATCH] User {user_idx}: processing {pos_count} positive and {neg_count} negative examples"
            )

        temp_preds += np.take(logits[user_idx], np.array(list(test_positive_set[user_idx]))).tolist()
        temp_y += [1.0 for _ in range(pos_count)]

        temp_preds += np.take(logits[user_idx], auc_negatives[user_idx]).tolist()
        temp_y += [0.0 for _ in range(neg_count)]

    print(f"[EVAL_BATCH] Collected {len(temp_preds)} predictions for AUC calculation")

    # Marking train-set consumed items as negative INF
    print(f"[EVAL_BATCH] Marking train-set consumed items as negative infinity")
    for user_idx in valid_user_indices:
        if user_idx % 1000 == 0:  # Only print every 1000 users to avoid excessive output
            print(f"[EVAL_BATCH] User {user_idx}: marking {len(train_positive[user_idx])} train positive items as -INF")
        logits[user_idx][train_positive[user_idx]] = -INF

    print(f"[EVAL_BATCH] Sorting indices for top-{max(topk)} recommendations")
    recommended_item_indices = (-logits).argsort()[:, : max(topk)].tolist()
    batch_exposures = {k: np.zeros(logits.shape[1]) for k in topk}

    user_recommendations = {}

    num_valid_users_in_batch = 0.0
    assert 0 not in topk, "0 in k values"

    for k_step, k in enumerate(topk):
        print(f"[EVAL_BATCH] Computing metrics for k={k}")
        user_recommendations[k] = []
        hr_batch_sum, ndcg_batch_sum, psp_batch_sum = 0, 0, 0

        for user_idx in valid_user_indices:
            if hyper_params["use_gini"]:
                # Update item exposures for this batch at this k
                for item_idx in recommended_item_indices[user_idx][:k]:
                    category = data.data["item_map_to_category"].get(item_idx + 1)
                    user_recommendations[k].append(
                        {
                            "id": item_idx + 1,
                            "category": category
                        }
                    )

            try:
                hr = eval_metrics.hr(recommended_item_indices[user_idx], test_positive_set[user_idx], k)
                ndcg = eval_metrics.ndcg(recommended_item_indices[user_idx], test_positive_set[user_idx], k)
                psp = eval_metrics.psp(recommended_item_indices[user_idx], test_positive_set[user_idx], item_propensity, k)
            except ZeroDivisionError:
                print(f"{k} {user_idx} {user_idx in valid_user_indices} [EVAL_BATCH] WARNING: No ground truth recommendations in test set of user {user_idx}. Skipping...")
                continue

            if user_idx % 1000 == 0:
                print(f"[EVAL_BATCH] User {user_idx}, HR@{k} = {hr}, NDCG@{k} = {ndcg}, PSP@{k} = {psp}")

            hr_batch_sum += hr
            ndcg_batch_sum += ndcg
            psp_batch_sum += psp

        metrics["HR@{}".format(k)] += hr_batch_sum
        metrics["NDCG@{}".format(k)] += ndcg_batch_sum
        metrics["PSP@{}".format(k)] += psp_batch_sum

        print(f"[EVAL_BATCH] k={k} metrics - Average HR: {hr_batch_sum/len(valid_user_indices):.4f}, Average NDCG: {ndcg_batch_sum/len(valid_user_indices):.4f}, Average PSP: {psp_batch_sum/len(valid_user_indices):.4f}")
        if hyper_params["use_gini"]: print(f"[EVAL_BATCH] Collected {len(user_recommendations[k])} recommendations for k={k}" )

    print(f"[EVAL_BATCH] Batch evaluation complete, returning {len(temp_preds)} predictions")

    return metrics, temp_preds, temp_y, user_recommendations if hyper_params["use_gini"] else {}