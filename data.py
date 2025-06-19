from scipy.sparse import csr_matrix
import jax.numpy as jnp
import numpy as np
import copy
import h5py
import gc
import pandas as pd
import os
from tqdm import tqdm

from utils import parse_neg_sampling_param

tqdm.pandas()


class Dataset:
    def __init__(self, hyper_params):
        self.data = load_raw_dataset(
            hyper_params["dataset"],
            hyper_params["item_id"],
            hyper_params["category_id"],
            parse_neg_sampling_param(hyper_params["neg_sampling_strategy"]),
        )
        self.set_of_active_users = list(set(self.data["train"][:, 0].tolist()))
        self.hyper_params = self.update_hyper_params(hyper_params)

    def update_hyper_params(self, hyper_params):
        updated_params = copy.deepcopy(hyper_params)

        self.num_users, self.num_items = self.data["num_users"], self.data["num_items"]
        self.num_interactions = self.data["num_interactions"]

        # Update hyper-params to have some basic data stats
        updated_params.update(
            {
                "num_users": self.num_users,
                "num_items": self.num_items,
                "num_interactions": self.num_interactions,
            }
        )

        return updated_params

    def sample_users(self, num_to_sample):
        if num_to_sample == -1:
            ret = self.data["train_matrix"]
        else:
            sampled_users = np.random.choice(
                self.set_of_active_users, num_to_sample, replace=False
            )
            sampled_interactions = self.data["train"][
                np.in1d(self.data["train"][:, 0], sampled_users)
            ]
            ret = csr_matrix(
                (
                    np.ones(sampled_interactions.shape[0]),
                    (sampled_interactions[:, 0], sampled_interactions[:, 1]),
                ),
                shape=(self.num_users, self.num_items),
            )

        # This just removes the users which were not sampled
        return jnp.array(ret[ret.getnnz(1) > 0].todense())


def load_raw_dataset(
    dataset,
    item_id,
    category_id,
    neg_sampling_strategy,
    data_path=None,
    index_path=None,
    item_path=None,
):
    if data_path is None or index_path is None:
        data_path, index_path = [
            f"data/{dataset}/total_data.hdf5",
            f"data/{dataset}/index.npz",
        ]
        print(f"Using default paths: data_path={data_path}, index_path={index_path}")

    print(f"Loading data from {data_path}")
    with h5py.File(data_path, "r") as f:
        data = np.array(list(zip(f["user"][:], f["item"][:], f["rating"][:])))
    print(f"Loaded raw data with shape: {data.shape}")

    print(f"Loading index from {index_path}")
    index = np.array(np.load(index_path)["data"], dtype=np.int32)
    print(f"Loaded index with shape: {index.shape}")

    def remap(data, index):
        print("Remapping user and item IDs")
        ## Counting number of unique users/items before
        valid_users, valid_items = set(), set()

        print("Identifying valid users and items")
        for at, (u, i, r) in enumerate(tqdm(data, desc="Scanning for valid entries")):
            if index[at] != -1:
                valid_users.add(u)
                valid_items.add(i)

        print(
            f"Found {len(valid_users)} valid users and {len(valid_items)} valid items"
        )

        ## Map creation done!
        user_map = dict(zip(list(valid_users), list(range(len(valid_users)))))
        item_map = dict(zip(list(valid_items), list(range(len(valid_items)))))
        print("User and item mapping created")

        return user_map, item_map

    user_map, item_map = remap(data, index)

    print("Creating new data and index arrays with mapped IDs")
    new_data, new_index = [], []
    for at, (u, i, r) in enumerate(tqdm(data, desc="Remapping data")):
        if index[at] == -1:
            continue
        new_data.append([user_map[u], item_map[i], r])
        new_index.append(index[at])

    data = np.array(new_data, dtype=np.int32)
    index = np.array(new_index, dtype=np.int32)
    print(f"Remapped data shape: {data.shape}, index shape: {index.shape}")

    print("Loading item data")
    if item_path is None:
        item_path = f"data/{dataset}/{dataset}.item"
        print(f"Using default item_path: {item_path}")

    print(f"Reading item data from {item_path}")
    # FIX: Handle CSV parsing errors with more robust error handling
    try:
        # First attempt with standard settings
        item_df = pd.read_csv(
            item_path, delimiter="\t", header=0, engine="python", encoding="latin-1"
        )
    except pd.errors.ParserError as e:
        print(f"Parser error with standard settings: {e}")
        print("Trying with on_bad_lines='warn' and encoding='utf-8'")
        item_df = pd.read_csv(
            item_path,
            delimiter="\t",
            header=0,
            engine="python",
            encoding="utf-8",
            on_bad_lines="warn",
        )
    print(f"Loaded item data with shape: {item_df.shape}")

    # This section needs refactoring
    if dataset == "douban":
        print("Processing Douban dataset")
        item_map_to_category = dict(
            zip(item_df[item_id].astype(int) + 1, item_df[category_id])
        )

    elif dataset == "ml-1m":
        print("Processing MovieLens 1M dataset")
        item_map_to_category = dict(
            zip(item_df[item_id].astype(int) + 1, item_df[category_id.split(" ")[0]]) # using the first genre here
        )
    else:
        print("Processing other dataset")
        all_genres = [
            genre
            for genre_list in item_df[category_id].fillna("Nan")
            for genre in genre_list.strip("[]").split(", ") 
        ]    
        item_map_to_category = dict(
            zip(item_df[item_id].astype(int) + 1, item_df[category_id.split(", ")[0]])
        )

    def select(data, index, index_val):
        print(f"Selecting data with index value {index_val}")
        selected_indices = np.where(index == index_val)[0]
        print(f"Found {len(selected_indices)} entries with index value {index_val}")
        final = data[selected_indices]
        final[:, 2] = 1.0
        return final.astype(np.int32)

    print("Creating train/val/test splits")
    ret = {
        "item_map": item_map,
        "train": select(data, index, 0),
        "val": select(data, index, 1),
        "test": select(data, index, 2),
        "item_map_to_category": item_map_to_category,
    }
    print(
        f"Split sizes - Train: {len(ret['train'])}, Val: {len(ret['val'])}, Test: {len(ret['test'])}"
    )

    num_users = int(max(data[:, 0]) + 1)
    num_items = len(item_map)
    print(f"Dataset has {num_users} users and {num_items} items")

    print("Cleaning up memory")
    del data, index
    gc.collect()

    def make_user_history(arr):
        print(f"Creating user history from array with shape {arr.shape}")
        ret = [set() for _ in range(num_users)]
        for u, i, r in tqdm(arr, desc="Building user history"):
            if i >= num_items:
                continue
            ret[int(u)].add(int(i))

        # Log some statistics about the history
        history_sizes = [len(h) for h in ret]
        print(
            f"User history stats - Min: {min(history_sizes)}, Max: {max(history_sizes)}, "
            f"Avg: {sum(history_sizes)/len(history_sizes):.2f}"
        )
        return ret

    print("Creating positive sets for train/val/test")
    ret["train_positive_set"] = make_user_history(ret["train"])
    ret["val_positive_set"] = make_user_history(ret["val"])
    ret["test_positive_set"] = make_user_history(ret["test"])

    print("Creating sparse matrices")
    ret["train_matrix"] = csr_matrix(
        (
            np.ones(ret["train"].shape[0]),
            (ret["train"][:, 0].astype(np.int32), ret["train"][:, 1].astype(np.int32)),
        ),
        shape=(num_users, num_items),
    )
    print(
        f"Created train matrix with shape {ret['train_matrix'].shape} and {ret['train_matrix'].nnz} non-zeros"
    )

    ret["val_matrix"] = csr_matrix(
        (
            np.ones(ret["val"].shape[0]),
            (ret["val"][:, 0].astype(np.int32), ret["val"][:, 1].astype(np.int32)),
        ),
        shape=(num_users, num_items),
    )
    print(
        f"Created val matrix with shape {ret['val_matrix'].shape} and {ret['val_matrix'].nnz} non-zeros"
    )

    ret["test_matrix"] = csr_matrix(
        (
            np.ones(ret["test"].shape[0]),
            (ret["test"][:, 0].astype(np.int32), ret["test"][:, 1].astype(np.int32)),
        ),
        shape=(num_users, num_items),
    )
    print(
        f"Created test matrix with shape {ret['test_matrix'].shape} and {ret['test_matrix'].nnz} non-zeros"
    )

    # Negatives will be used for AUC computation
    print("Generating negative samples for evaluation")
    ret["negatives"] = [set() for _ in range(num_users)]

    for u in tqdm(range(num_users), desc="Generating negatives"):
        attempts = 0

        if neg_sampling_strategy[0] == "total":
            loop_upper_bound = neg_sampling_strategy[1]
        elif neg_sampling_strategy[0] == "positive":
            num_pos = len(ret["test_positive_set"][u])
            loop_upper_bound = neg_sampling_strategy[1] * num_pos

        # print(f"Debug: {neg_sampling_strategy} -> num_pos = {len(ret['test_positive_set'][u])}; loop_upper_bound {loop_upper_bound}")
        max_attempts = 10 * len(ret["test_positive_set"][u])
        while len(ret["negatives"][u]) < loop_upper_bound:
            attempts += 1
            if attempts > max_attempts:  # Safety check to avoid infinite loops
                print(f"User {u} could not get {loop_upper_bound} negatives after {max_attempts} attempts")
                break

            rand_item = np.random.randint(0, num_items)
            if rand_item in ret["train_positive_set"][u]:
                continue
            if rand_item in ret["val_positive_set"][u]:
                continue
            if rand_item in ret["test_positive_set"][u]:
                continue
            ret["negatives"][u].add(rand_item)
        ret["negatives"][u] = list(ret["negatives"][u])

    ret.update(
        {
            "num_users": num_users,
            "num_items": num_items,
            "num_interactions": len(ret["train"]),
        }
    )

    # Log summary statistics
    print("Dataset loading complete. Summary:")
    print("# users:", num_users)
    print("# items:", num_items)
    print("# interactions:", len(ret["train"]))

    return ret


if __name__ == "__main__":
    data = Dataset(
        {
            "dataset": "ml-1m",
            "item_id": "item_id:token",
            "category_id": "genre:token_seq",
        }
    )
