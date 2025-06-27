import numpy as np
import h5py
import sys
import os
import pandas as pd
from hyper_params import hyper_params

BASE_PATH = "data/"


def prep_recbole(
    inter_file_path, user_id, item_id, rating_id, item_file_path=None, item_id_2=None
):
    """
    Process ML-1M dataset from RecBole format
    RecBole typically stores data in .inter files or in a processed format
    """

    # If the RecBole .inter file exists, process it
    if os.path.exists(inter_file_path):
        # RecBole .inter files are typically tab-separated with headers
        df = pd.read_csv(inter_file_path, sep="\t")
        df.dropna(subset=[user_id, item_id, rating_id], how='any', inplace=True)

        # Extract user_id, item_id, and rating columns
        users = df[user_id].values
        items = df[item_id].values
        ratings = df[rating_id].values
    else:
        print("The .inter file was not found under the path:" + inter_file_path)
        # Exit early if no interaction file
        raise Exception(f"Interaction file not found at {inter_file_path}")

    # First, check if the item file exists and load it (skip for datasets without item files like Netflix)
    item_df = None
    if item_file_path is not None and os.path.exists(item_file_path):
        if "original" not in item_file_path:
            print(
                "Provide the original dataset as {dataset_name}_original.item as a new dataset will be created and to not be overwritten!"
            )
            raise Exception(
                f"Dataset was not found. I am looking for {item_file_path.replace('.item', '_original.item')}."
            )
        # RecBole .item files are typically tab-separated with headers
        item_df = pd.read_csv(item_file_path, sep="\t")
    else:
        print("No or invalid item file specified - continuing without item file")
        item_df = None

    # Check if we have string IDs (like Amazon ASINs) or numeric IDs
    sample_user = users[0] if len(users) > 0 else None
    has_string_ids = isinstance(sample_user, str) and not sample_user.isdigit()
    
    if has_string_ids:
        print("Detected string IDs - skipping numeric offset logic and creating direct mappings")
        # For string IDs, we skip the min_user logic and go directly to mapping
    else:
        # Convert to zero-based indexing if needed (for numeric IDs)
        min_user = min(users)
        if min_user > 0:
            users = [u - min_user for u in users]

    # Create sequential mapping for users
    unique_users = sorted(set(users))
    map_user = {user: idx for idx, user in enumerate(unique_users)}

    # Create sequential mapping for items - ensuring sequential IDs from 0 to max_item
    # First, get all unique item IDs from interactions
    all_items = set()
    for item in items:
        item_val = item.item() if hasattr(item, "item") else item
        all_items.add(item_val)

    # Add items from item file if they're not already in the set (only if item file exists)
    if item_df is not None and item_id_2 is not None:
        for item in item_df[item_id_2].values:
            if pd.notna(item):  # Skip NaN values
                all_items.add(item)

    # Create sequential mapping for all items
    sorted_items = sorted(all_items)
    map_item = {item: idx for idx, item in enumerate(sorted_items)}

    # Ensure we have a complete sequential range from 0 to max_item
    max_item_id = len(map_item) - 1
    print(f"Created sequential item mapping from 0 to {max_item_id}")

    # Verify the mapping is complete
    if len(map_item) != max_item_id + 1:
        print(
            f"Warning: Item mapping has {len(map_item)} items but max ID is {max_item_id}"
        )
        # Fill in any gaps if needed
        for i in range(max_item_id + 1):
            if i not in map_item.values():
                print(f"Missing item ID in sequence: {i}")

    # Organize data by user
    num_users = len(map_user)
    data = [[] for _ in range(num_users)]

    # Add valid interactions to the data structure
    for i in range(len(users)):
        user_val = users[i].item() if hasattr(users[i], "item") else users[i]
        item_val = items[i].item() if hasattr(items[i], "item") else items[i]
        rating_val = (
            ratings[i].item() if hasattr(ratings[i], "item") else float(ratings[i])
        )

        # Skip if item is not in our mapping (shouldn't happen with our approach)
        if item_val not in map_item:
            print(f"Warning: Item {item_val} not found in mapping, skipping")
            continue

        # Skip if user is not in our mapping (shouldn't happen with our approach)
        if user_val not in map_user:
            print(f"Warning: User {user_val} not found in mapping, skipping")
            continue

        data[map_user[user_val]].append([map_item[item_val], rating_val])

    # Update the item file with new mappings (only if item file exists)
    if item_df is not None and item_file_path is not None:
        # Create a new column with the mapped IDs
        item_df["mapped_id"] = item_df[item_id_2].apply(
            lambda x: map_item.get(x, float("nan")) if pd.notna(x) else float("nan")
        )

        # Drop rows where mapping failed
        before_count = len(item_df)
        item_df = item_df.dropna(subset=["mapped_id"])
        after_count = len(item_df)

        if before_count != after_count:
            print(f"Dropped {before_count - after_count} items that couldn't be mapped")

        # Replace the original ID column with the mapped IDs
        item_df[item_id_2] = item_df["mapped_id"]
        item_df = item_df.drop(columns=["mapped_id"])

        # Save the updated item file
        item_df.to_csv(
            item_file_path.replace("_original.", ".", 1), sep="\t", index=False
        )
        print(f"Saved updated item file with {len(item_df)} items")
    else:
        print("No item file to update - proceeding with interaction data only")
    return rating_data(data)


class rating_data:
    def __init__(self, data):
        self.data = data

        self.index = (
            []
        )  # 0: train, 1: validation, 2: test, -1: removed due to user frequency < 3
        for user_data in self.data:
            for _ in range(len(user_data)):
                self.index.append(42)

    def train_test_split(self):
        at = 0

        invalid = 0
        for user in range(len(self.data)):
            first_split_point = int(0.8 * len(self.data[user]))
            second_split_point = int(0.9 * len(self.data[user]))

            indices = np.arange(len(self.data[user]))
            np.random.shuffle(indices)

            
            for timestep, (item, rating) in enumerate(self.data[user]):
                if len(self.data[user]) < 3:
                    self.index[at] = -1
                else:
                    # Force at least one element in user history to be in test and one in val
                    if timestep == indices[0]:
                        self.index[at] = 0 
                    elif timestep == indices[1]:
                        self.index[at] = 1
                    elif timestep == indices[2]:
                        self.index[at] = 2
                    else:
                        if timestep in indices[:first_split_point]:
                            self.index[at] = 0
                        elif timestep in indices[first_split_point:second_split_point]:
                            self.index[at] = 1
                        else:
                            self.index[at] = 2
                at += 1

        assert at == len(self.index)
        print(np.sum(self.index == -1))
        print(f"Removed {invalid} invalid users. {np.sum(np.array(self.index) >= 0, axis=0)} interactions left.")
        self.complete_data_stats = None


    def train_test_split_strong(self):
        at = 0
        num_users = len(self.data)
        first_split_point = int(0.8 * num_users)
        second_split_point = int(0.9 * num_users)
        random_order_users = np.arange(num_users)
        np.random.shuffle(random_order_users)
        for user in range(num_users):
            if len(self.data[user]) < 3:
                split = -1
            elif user in random_order_users[:first_split_point]:
                split = 0
            elif user in random_order_users[first_split_point:second_split_point]:
                split = 1
            else:
                split = 2
            for _ in self.data[user]:
                self.index[at] = split 
                at += 1
        assert at == len(self.index)
        print(np.sum(self.index == -1))
        self.complete_data_stats = None


    def save_index(self, path):
        os.makedirs(path, exist_ok=True)
        with open(path + "/index.npz", "wb") as f:
            np.savez_compressed(f, data=self.index)

    def save_data(self, path):
        flat_data = []
        for u in range(len(self.data)):
            flat_data += list(map(lambda x: [u] + x, self.data[u]))
        flat_data = np.array(flat_data)

        shape = [len(flat_data)]

        os.makedirs(path, exist_ok=True)
        with h5py.File(path + "/total_data.hdf5", "w") as file:
            dset = {}
            dset["user"] = file.create_dataset(
                "user", shape, dtype="i4", maxshape=shape, compression="gzip"
            )
            dset["item"] = file.create_dataset(
                "item", shape, dtype="i4", maxshape=shape, compression="gzip"
            )
            dset["rating"] = file.create_dataset(
                "rating", shape, dtype="f", maxshape=shape, compression="gzip"
            )

            dset["user"][:] = flat_data[:, 0]
            dset["item"][:] = flat_data[:, 1]
            dset["rating"][:] = flat_data[:, 2]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("This file needs the dataset name as the first argument...")
        exit(0)
    elif len(sys.argv) < 3:
        generalization = "weak"
    elif len(sys.argv) == 3:
        generalization = sys.argv[2]

    dataset = sys.argv[1]
    np.random.seed(hyper_params[dataset]["seed"])

    print(f"\n\n!!!!!!!! STARTED PROCESSING {dataset} with seed {hyper_params[dataset]['seed']} and {generalization} generalization !!!!!!!!")

    if dataset == "ml-1m":
        total_data = prep_recbole(
            BASE_PATH + "/ml-1m/ml-1m.inter",
            "user_id:token",
            "item_id:token",
            "rating:float",
            BASE_PATH + "/ml-1m/ml-1m_original.item",
            "item_id:token",
        )
    elif dataset == "steam":
        total_data = prep_recbole(
            BASE_PATH + "/steam/steam.inter",
            "user_id:token",
            "product_id:token",
            "play_hours:float",
            BASE_PATH + "/steam/steam_original.item",
            "id:token",
        )
    elif dataset == "amazon_magazine":
        total_data = prep_recbole(
            BASE_PATH + "amazon_magazine/amazon_magazine.inter",
            "user_id:token",
            "item_id:token",
            "rating:float",
            BASE_PATH + "amazon_magazine/amazon_magazine_original.item",
            "item_id:token",
        )
    elif dataset == "ml-20m":
        total_data = prep_recbole(
            BASE_PATH + "/ml-20m/ml-20m.inter",
            "user_id:token",
            "item_id:token",
            "rating:float",
            BASE_PATH + "/ml-20m/ml-20m_original.item",
            "item_id:token",
        )
    elif dataset == "ml-10m":
        total_data = prep_recbole(
            BASE_PATH + "/ml-10m/ml-10m.inter",
            "user_id:token",
            "item_id:token",
            "rating:float",
            BASE_PATH + "/ml-10m/ml-10m_original.item",
            "item_id:token",
        )
    elif dataset == "douban":
        total_data = prep_recbole(
            BASE_PATH + "/douban/douban.inter",
            "user_id:token",
            "item_id:token",
            "rating:float",
            BASE_PATH + "/douban/douban_original.item",
            "movie_id:token",
        )
    elif dataset == "netflix":
        total_data = prep_recbole(
            BASE_PATH + "/netflix/netflix.inter",
            "user_id:token",
            "item_id:token",
            "rating:float"
        )
    else:
        raise Exception("Could not undestand this dataset")

    total_data.save_data(BASE_PATH + "{}/".format(dataset))
    if generalization == "strong":
        total_data.train_test_split_strong()
    else: 
        total_data.train_test_split()
    total_data.save_index(BASE_PATH + "{}/".format(dataset))
