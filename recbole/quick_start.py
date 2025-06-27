# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import sys
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger
import numpy as np

from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

from itertools import combinations


def run(
    model,
    dataset,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    if nproc == 1 and world_size <= 0:
        res = run_recbole(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            run_recboles,
            args=(model, dataset, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process


def run_recboles(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole(
        *args[:3],
        **kwargs,
    )
##########################################################################################################################

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm



def jaccard_distance(set1,set2):
    return 1 - (len(set1 & set2)/ len(set1 | set2 ))



from collections import defaultdict



def get_recommendation_list(dataset,test_data, model , topk, device='cpu'):
    """ get top k recommendation list for each user using full sort prediction.
    
    """
    recom_lists = {}
    user_num = dataset.user_num
    item_num = dataset.item_num 
    uid_field = dataset.uid_field # user_id
    iid_field = dataset.iid_field #item_id

    test_pos_df = test_data.dataset.inter_feat # interaction data frame object thing

    for user_id in tqdm(range(user_num)):
        # skip users with no test positives
        user_mask = test_pos_df[uid_field]==user_id
        user_pos_items = test_pos_df[iid_field][user_mask]

        if len(user_pos_items) == 0: continue 

        interaction = {uid_field : torch.tensor([user_id]*item_num, dtype=torch.long, device=device), 
                       iid_field: torch.tensor(list(range(item_num)), dtype=torch.long, device=device)}

        # use full_sort_predict to get scores
        scores = model.full_sort_predict(interaction)
        top_k_items = torch.topk(scores, topk).indices.view(-1).cpu().tolist()
        recom_lists[user_id] = set(top_k_items)
    return recom_lists

def calculate_interlist_diversity(recommended_list) :
    users = list(recommended_list.keys())
    total_distance = 0
    pair_count = 0

    for u, v in tqdm(combinations(users, 2)):
        dist = jaccard_distance(recommended_list[u], recommended_list[v])
        total_distance += dist
        pair_count += 1

    return 2*total_distance / (len(users) * (len(users)-1)) if pair_count else 0


def compute_auc(model, dataset, test_data, device='cpu', num_neg=2):
    """
    calculate average per user AUC with one positive and num_neg negatives per user
    args:
        model: trained model with .predict()
        dataset
        test_data: Dataloader (the one from "data_preparation")
        num_neg: num negative samples per user to use when calculating auc 
    return: mean user level AUC
    """
    
    user_num = dataset.user_num
    item_num = dataset.item_num
    uid_field = dataset.uid_field # user_id
    iid_field = dataset.iid_field #item_id
    user_auc_list = []

    model.eval()

    test_pos_df = test_data.dataset.inter_feat # interaction data frame object thing

    for user_id in tqdm(range(user_num)):

        # test positives
        user_mask = test_pos_df[uid_field]==user_id
        user_pos_items = test_pos_df[iid_field][user_mask]
        if len(user_pos_items)==0:
            continue

        #get a test pos item : change this to use all positive interaction 
        # pos_item = np.random.choice(user_pos_items.cpu().numpy())
        pos_items = user_pos_items 

        # get items user interacted with (train+test)
        interacted_items = dataset.inter_matrix(form='csr')[user_id].nonzero()[1]
        all_items = set(range(item_num))
        neg_items = list(all_items - set(interacted_items))
        if len(neg_items)> len(pos_items)* 2:
            sampled_neg_items = np.random.choice(neg_items,size = len(pos_items)*2 ,replace = False)

        # items = [pos_items]+list(sampled_neg_items)
        pos_items = pos_items.tolist()
        items = pos_items + list(sampled_neg_items)
        labels = [1]*len(pos_items) + [0]*len(sampled_neg_items)
        users = [user_id]*len(items)
        
        # interaction dictionary thing for recbole model
        interaction = {dataset.uid_field: torch.tensor(users).to(device),
                        dataset.iid_field: torch.tensor(items).to(device)}
        with torch.no_grad():
            scores = model.predict(interaction).cpu().numpy()


        #sanity check lmao
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, scores)
            user_auc_list.append(auc)

    # return avg
    return float(np.mean(user_auc_list)) if user_auc_list else float("nan")

from collections import Counter
def compute_GINI(dataset,rec_list):

    item_num = dataset.item_num
    sum = 0
    item_counter = Counter(rec_list)


    for item in range(item_num): 
        sum += item_counter[item] * ( (2*item) -item_num-1)


    return sum/(item_num * sum(item_counter.values()))
      







##########################################################################################################################

def objective_function(config_dict=None, config_file_list=None, saved=False):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    print(f"\n\n\nCONFIG TRANSFORM VALUE == {config['transform']}\n\n\n")
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    ##########################################################################################################################

    logger.info(f"[Test Result]  :  {test_result}")
    # Calculate AUC 
    print(f"\n\nCALCULATE AUC \n\n")
    try:
        auc_score = compute_auc(model, dataset, test_data, num_neg=2)
        test_result["user_auc"] = auc_score
        logger.info(f"[User_level AUC]: {auc_score:.4f}")
        print(f"User_level AUC SCORE : {auc_score}")

    except Exception as e:
        test_result["auc"] = float("nan")
        logger.warning(f"AUC computation failed: {e}")
        auc_score = None
        print("AUC COMPUTATION FAILED ")

    



    logger.info(f" [Test Results] : {test_result}")
    print(f"Test results :: {test_result}")
 
    
    ##########################################################################################################################
    from ray import tune
    if hasattr(tune, "report"):
        tune.report(**test_result)
    if  auc_score: return {
    'model': model,
    'best_valid_score': best_valid_score,
    'best_valid_result': best_valid_result,
    'test_result': test_result,
    "valid_score_bigger": config["valid_metric_bigger"],
    'user_auc': test_result['user_auc'],
    }
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
