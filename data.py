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
            hyper_params["diversity_metrics"],
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
    diversity_metrics=False,
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
    
    def select(data, index, index_val):
        print(f"Selecting data with index value {index_val}")
        selected_indices = np.where(index == index_val)[0]
        print(f"Found {len(selected_indices)} entries with index value {index_val}")
        final = data[selected_indices]
        final[:, 2] = 1.0
        return final.astype(np.int32)

    def parse_tags(category_string):
        """DIVERSITY METRICS function. Parse category/tag string into a list of tags."""
        if pd.isna(category_string) or str(category_string).strip() == "":
            return []
        
        text = str(category_string).strip()
        
        # Remove brackets
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1]
        
        # Split by common separators
        for sep in [',', '|', ';']:
            if sep in text:
                tags = text.split(sep)
                break
        else:
            # Split by space if no other separator found
            tags = text.split()
        
        # Clean tags
        cleaned_tags = []
        for tag in tags:
            tag = tag.strip(' "\'')
            if tag and tag.lower() not in ['nan', 'null', 'none', 'n/a']:
                cleaned_tags.append(tag)
        
        return cleaned_tags

    def create_item_mappings(item_df, item_id_col, category_id_col, item_map):
        """
        DIVERSITY METRICS function. Create item category and tag mappings.

        Returns:
            tuple: (item_map_to_category, item_tag_mapping)
            - item_map_to_category: maps item_id -> first category/tag
            - item_tag_mapping: maps item_id -> set of all tags
        """
        print(f"Creating item mappings from columns: '{item_id_col}', '{category_id_col}'")
        
        item_map_to_category = {}
        item_tag_mapping = {}
        
        for _, row in item_df.iterrows():
            original_item_id = int(row[item_id_col])
            
            # Skip items not in final dataset
            if original_item_id not in item_map:
                continue
                
            remapped_item_id = item_map[original_item_id]
            category_data = row.get(category_id_col, "")
            
            # use brand directly as tag - amazon magazine dataset
            if category_id_col == "brand:token":
                tags = [category_data]
            # Parse tags from category data    
            else:
                tags = parse_tags(category_data)
            
            # Map to first category (for backward compatibility)
            first_category = tags[0] if tags else ""
            item_map_to_category[remapped_item_id] = first_category
            
            # Map to all tags (for inter-list distance)
            item_tag_mapping[remapped_item_id] = set(tags)
        
        print(f"Created mappings for {len(item_map_to_category)} items")
        return item_map_to_category, item_tag_mapping


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

    if diversity_metrics:
        print("Loading item data")
        if item_path is None:
            item_path = f"data/{dataset}/{dataset}.item"
            print(f"Using default item_path: {item_path}")
    
        print(f"Reading item data from {item_path}")
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

        # Create both mappings in a unified way
        item_map_to_category, item_tag_mapping = create_item_mappings(
            item_df, item_id, category_id, item_map
        )
    else:
        item_map_to_category = {}
        item_tag_mapping = {}


    print("Creating train/val/test splits")

    ret = {
        "item_map": item_map,
        "train": select(data, index, 0),
        "val": select(data, index, 1),
        "test": select(data, index, 2),
        "item_map_to_category": item_map_to_category,
        "item_tag_mapping": item_tag_mapping,
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

    all_items = set(item_map.values())

    for u in tqdm(range(num_users), desc="Generating negatives"):
        user_positives = (
            ret["train_positive_set"][u] |
            ret["val_positive_set"][u] |
            ret["test_positive_set"][u]
        )

        if neg_sampling_strategy[0] == "total":
            to_sample = neg_sampling_strategy[1]
        elif neg_sampling_strategy[0] == "positive":
            to_sample = len(user_positives) * neg_sampling_strategy[1] 

        available_negatives = list(all_items - user_positives)

        if len(available_negatives) < to_sample:
            print(f"User {u} has only {len(available_negatives)} available negatives, requested {to_sample}. Taking all...")
            sampled_negatives = available_negatives
        else:
            sampled_negatives = np.random.choice(
                available_negatives, to_sample, replace=False
            )

        ret["negatives"][u] = sorted(sampled_negatives)

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
