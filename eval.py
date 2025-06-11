import jax
import jax.numpy as jnp
import numpy as np

import eval_metrics
from utils import get_item_propensity


class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        print(f"[GINI] Computing Gini coefficient for {len(values)} values")
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            print("[GINI] Sum of values is 0, returning 0.0")
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        print(f"[GINI] Computed Gini coefficient: {gini:.4f}")
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        print(f"[GINI] Calculating Gini for {len(articles)} articles using key '{key}'")
        # count frequencies
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        print(f"[GINI] Found {len(freqs)} unique {key} values")
        return self.gini_coefficient(list(freqs.values()))


INF = float(1e6)


def evaluate(
    hyper_params,
    kernelized_rr_forward,
    data,
    train_x,
    topk=[10, 100],
    test_set_eval=False,
):
    print(
        f"\n[EVALUATE] Starting evaluation with topk={topk}, test_set_eval={test_set_eval}"
    )
    print(
        f"[EVALUATE] Hyperparameters: num_users={hyper_params['num_users']}, num_items={hyper_params['num_items']}, lambda={hyper_params['lamda']}"
    )

    preds, y_binary, metrics = [], [], {}
    for kind in ["HR", "NDCG", "PSP", "GINI"]:
        for k in topk:
            metrics["{}@{}".format(kind, k)] = 0.0
    metrics["#users_w_gt"] = 0.0

    # Train positive set -- these items will be set to -infinity while prediction on the val/test set
    train_positive_list = list(map(list, data.data["train_positive_set"]))
    print(f"[EVALUATE] Train positive set size: {len(train_positive_list)}")

    if test_set_eval:
        print(
            "[EVALUATE] Adding validation positive set to train positive set for test evaluation"
        )
        for u in range(len(train_positive_list)):
            train_positive_list[u] += list(data.data["val_positive_set"][u])

    # Train positive interactions (in matrix form) as context for prediction on val/test set
    eval_context = data.data["train_matrix"]
    if test_set_eval:
        print("[EVALUATE] Adding validation matrix to evaluation context")
        eval_context += data.data["val_matrix"]

    # What needs to be predicted
    to_predict = data.data["val_positive_set"]
    if test_set_eval:
        print("[EVALUATE] Using test positive set for prediction")
        to_predict = data.data["test_positive_set"]
    print(f"[EVALUATE] Prediction set size: {len(to_predict)}")

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
            topk,
            metrics,
            data,
            hyper_params
        )
        print(f"[EVALUATE] Batch evaluation complete")

        if hyper_params["use_gini"]:
            # Accumulate item exposures for GINI calculation
            for k in topk:
                if k not in user_recommendations:
                    user_recommendations[k] = []
                user_recommendations[k] += user_recommendations_batch[k]
                print(
                    f"[EVALUATE] Accumulated {len(user_recommendations_batch[k])} recommendations for k={k}"
                )

        preds += temp_preds
        y_binary += temp_y
        print(
            f"[EVALUATE] Accumulated {len(temp_preds)} predictions and {len(temp_y)} labels"
        )

    print(f"[EVALUATE] All batches processed, computing final metrics")
    y_binary, preds = np.array(y_binary), np.array(preds)
    if (True not in np.isnan(y_binary)) and (True not in np.isnan(preds)):
        try:
            metrics["AUC"] = round(eval_metrics.auc(y_binary, preds), 4)
            print(f"[EVALUATE] Computed AUC: {metrics['AUC']}")
        except ValueError:
            print(f"[EVALUATE] WARNING: AUC is undefined when y_binary is only 1s or only 0s. Skipping...")
    else:
        print(
            "[EVALUATE] Warning: NaN values detected in y_binary or preds, skipping AUC calculation"
        )

    for kind in ["HR", "NDCG", "PSP"]:
        for k in topk:
            metrics["{}@{}".format(kind, k)] = round(
                float(metrics["{}@{}".format(kind, k)])
                / metrics["#users_w_gt"],
                4,
            )
            print(f"[EVALUATE] {kind}@{k}: {metrics['{}@{}'.format(kind, k)]}")

    if hyper_params["use_gini"]:
        print("[EVALUATE] Computing GINI coefficients")
        for k in topk:
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

    print(metrics["#users_w_gt"])

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

    # AUC Stuff
    temp_preds, temp_y = [], []
    for user_idx in range(len(logits)):
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
    for user_idx in range(len(logits)):
        if user_idx % 1000 == 0:  # Only print every 1000 users to avoid excessive output
            print(
                f"[EVAL_BATCH] User {user_idx}: marking {len(train_positive[user_idx])} train positive items as -INF"
            )
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

        for user_idx in range(len(logits)):
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
                if k_step == 0: # only count them for a single k value, valid users are the ones with gt
                    num_valid_users_in_batch += 1
            except ZeroDivisionError:
                print(f"[EVAL_BATCH] WARNING: No ground truth recommendations in test set of user {user_idx}. Skipping...")
                continue

            if user_idx % 1000 == 0:
                print(f"[EVAL_BATCH] User {user_idx}, HR@{k} = {hr}, NDCG@{k} = {ndcg}, PSP@{k} = {psp}")

            hr_batch_sum += hr
            ndcg_batch_sum += ndcg
            psp_batch_sum += psp

        metrics["HR@{}".format(k)] += hr_batch_sum
        metrics["NDCG@{}".format(k)] += ndcg_batch_sum
        metrics["PSP@{}".format(k)] += psp_batch_sum

        print(
            f"[EVAL_BATCH] k={k} metrics - Average HR: {hr_batch_sum/len(logits):.4f}, Average NDCG: {ndcg_batch_sum/len(logits):.4f}, Average PSP: {psp_batch_sum/len(logits):.4f}"
        )
        if hyper_params["use_gini"]:
            print(
                f"[EVAL_BATCH] Collected {len(user_recommendations[k])} recommendations for k={k}"
            )

    metrics["#users_w_gt"] += num_valid_users_in_batch

    print(
        f"[EVAL_BATCH] Batch evaluation complete, returning {len(temp_preds)} predictions"
    )
    return metrics, temp_preds, temp_y, user_recommendations if hyper_params["use_gini"] else {}