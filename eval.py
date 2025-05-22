import jax
import numpy as np
import jax.numpy as jnp
from numba import jit, float64

def reconstruct_utility(utility_list, weights, group_mask):
    """
        Reconstruct utility by re-weighting them and masking the utility of certain unused groups.
        :param utility_list: array for item/user utilities
        :param weights: array for item/user utilities weights
        :param group_mask: bool array for whether computed the group utilityes
        :return: re-constructed utility array
    """

    if not weights:
        weights = np.ones_like(utility_list)

    utility_list = np.array(utility_list)
    weights = np.array(weights)
    weighted_utility = utility_list * weights

    if group_mask:
        weighted_utility = mask_utility(weighted_utility, group_mask)


    return np.array(weighted_utility)

def mask_utility(utility, group_mask):
    """
    Mask the utility values based on the provided group mask.

    This function filters out the utility values where the corresponding
    group mask element is zero, effectively removing them from the output.


    :param utility: array for item/user utilities
    :param group_mask: bool array for whether computed the group utilityes
    :return: masked utility array
    """


    masked_utility = []
    for i, m in enumerate(group_mask):
        if m == 0:
            masked_utility.append(utility[i])

    return np.array(masked_utility)

def Gini(utility_list, weights=None, group_mask = None):
    """
    This function computes the Gini coefficient, a measure of statistical dispersion intended to represent income inequality within a nation or social group.
    The Gini coefficient is calculated based on the cumulative distribution of values in `utility_list`, which can optionally be weighted and masked.

    :param utility_list: array_like
        A 1D array representing individual utilities. The utilities are used to compute the Gini coefficient.
    :param weights: array_like, optional
        A 1D array of weights corresponding to `utility_list`. If provided, each utility value is multiplied by its respective weight before calculating the Gini coefficient. Defaults to None, implying equal weighting.
    :param group_mask: array_like, optional
        A 1D boolean array used to selectively include elements from `utility_list`. If provided, only the elements where the mask is True are considered in the calculation. Defaults to None, meaning all elements are included.

    :return: float: The computed Gini coefficient, ranging from 0 (perfect equality) to 1 (maximal inequality).
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.sort(weighted_utility)
    n = len(values)

    # gini compute
    cumulative_sum = np.cumsum(values)
    gini = (n + 1 - 2 * (np.sum(cumulative_sum) / cumulative_sum[-1])) / n

    return gini

INF = float(1e6)

def evaluate(hyper_params, kernelized_rr_forward, data, item_propensity, train_x, topk = [ 10, 100 ], test_set_eval = False):
    preds, y_binary, metrics = [], [], {}
    for kind in [ 'HR', 'NDCG', 'PSP', 'GINI' ]:
        for k in topk: 
            metrics['{}@{}'.format(kind, k)] = 0.0

    # Train positive set -- these items will be set to -infinity while prediction on the val/test set
    train_positive_list = list(map(list, data.data['train_positive_set']))
    if test_set_eval:
        for u in range(len(train_positive_list)): train_positive_list[u] += list(data.data['val_positive_set'][u])

    # Train positive interactions (in matrix form) as context for prediction on val/test set
    eval_context = data.data['train_matrix']
    if test_set_eval: eval_context += data.data['val_matrix']

    # What needs to be predicted
    to_predict = data.data['val_positive_set']
    if test_set_eval: to_predict = data.data['test_positive_set']

    # For GINI calculation - track item exposures across all recommendations
    item_exposures = np.zeros(hyper_params['num_items'])

    bsz = 20_000 # These many users
    for i in range(0, hyper_params['num_users'], bsz):
        temp_preds = kernelized_rr_forward(train_x, eval_context[i:i+bsz].todense(), reg = hyper_params['lamda'])
        
        metrics, temp_preds, temp_y, batch_exposures = evaluate_batch(
            data.data['negatives'][i:i+bsz], np.array(temp_preds), 
            train_positive_list[i:i+bsz], to_predict[i:i+bsz], item_propensity, 
            topk, metrics
        )
        
        # Accumulate item exposures for GINI calculation
        for k in topk:
            item_exposures += batch_exposures[k]
            
        preds += temp_preds
        y_binary += temp_y

    y_binary, preds = np.array(y_binary), np.array(preds)
    if (True not in np.isnan(y_binary)) and (True not in np.isnan(preds)):
        metrics['AUC'] = round(fast_auc(y_binary, preds), 4)

    for kind in [ 'HR', 'NDCG', 'PSP' ]:
        for k in topk: 
            metrics['{}@{}'.format(kind, k)] = round(
                float(100.0 * metrics['{}@{}'.format(kind, k)]) / hyper_params['num_users'], 4
            )
            
    for k in topk:
        metrics['GINI@{}'.format(k)] = Gini(item_exposures)

    metrics['num_users'] = int(train_x.shape[0])
    metrics['num_interactions'] = int(jnp.count_nonzero(train_x.astype(np.int8)))

    return metrics

def evaluate_batch(auc_negatives, logits, train_positive, test_positive_set, item_propensity, topk, metrics, train_metrics = False):
    # AUC Stuff
    temp_preds, temp_y = [], []
    for b in range(len(logits)):
        temp_preds += np.take(logits[b], np.array(list(test_positive_set[b]))).tolist()
        temp_y += [ 1.0 for _ in range(len(test_positive_set[b])) ]

        temp_preds += np.take(logits[b], auc_negatives[b]).tolist()
        temp_y += [ 0.0 for _ in range(len(auc_negatives[b])) ]

    # Marking train-set consumed items as negative INF
    for b in range(len(logits)): logits[b][ train_positive[b] ] = -INF

    indices = (-logits).argsort()[:, :max(topk)].tolist()
    batch_exposures = {k: np.zeros(logits.shape[1]) for k in topk}

    for k in topk: 
        for b in range(len(logits)):
            # Update item exposures for this batch at this k
            for item_idx in indices[b][:k]:
                batch_exposures[k][item_idx] += 1
                
            num_pos = float(len(test_positive_set[b]))

            metrics['HR@{}'.format(k)] += float(len(set(indices[b][:k]) & test_positive_set[b])) / float(min(num_pos, k))

            test_positive_sorted_psp = sorted([ item_propensity[x] for x in test_positive_set[b] ])[::-1]

            dcg, idcg, psp, max_psp = 0.0, 0.0, 0.0, 0.0
            for at, pred in enumerate(indices[b][:k]):
                if pred in test_positive_set[b]: 
                    dcg += 1.0 / np.log2(at + 2)
                    psp += float(item_propensity[pred]) / float(min(num_pos, k))
                if at < num_pos: 
                    idcg += 1.0 / np.log2(at + 2)
                    max_psp += test_positive_sorted_psp[at]

            metrics['NDCG@{}'.format(k)] += dcg / idcg
            metrics['PSP@{}'.format(k)] += psp / max_psp

    return metrics, temp_preds, temp_y, batch_exposures

@jit(float64(float64[:], float64[:]))
def fast_auc(y_true, y_prob):
    y_true = y_true[np.argsort(y_prob)]
    nfalse, auc = 0, 0
    for i in range(len(y_true)):
        nfalse += (1 - y_true[i])
        auc += y_true[i] * nfalse
    return auc / (nfalse * (len(y_true) - nfalse))
