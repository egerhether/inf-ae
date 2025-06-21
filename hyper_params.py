hyper_params = {
    "steam": {
        "dataset": "steam",
        "item_id": "id:token",  # configure it based on the .item file
        "category_id": "tags:token_seq",  # configure it based on the .item file
        "use_gini": True,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": 20000,  # -1 implies use all users
        "seed": 42,
        "gen": "strong"
    },
    "ml-20m": {
        "dataset": "ml-20m",
        "item_id": "item_id:token",  # configure it based on the .item file
        "category_id": "type:token_seq",  # configure it based on the .item file
        "use_gini": True,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": -1,  # -1 implies use all users
        "seed": 42,
    },
    "ml-10m": {
        "dataset": "ml-10m",
        "item_id": "item_id:token",  # configure it based on the .item file
        "category_id": "type:token_seq",  # configure it based on the .item file
        "use_gini": True,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": 10000,  # -1 implies use all users
        "seed": 42,
        "batch_size": 20000,
    },
    "ml-1m": {
        "dataset": "ml-1m",
        "item_id": "item_id:token",  # configure it based on the .item file
        "category_id": "genre:token_seq",  # configure it based on the .item file
        "use_gini": True,
        "float64": False,
        "depth": 1,
    "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": -1,  # -1 implies use all users
        "seed": 42,
        "neg_sampling_strategy": "positive2",
        "gen": "strong",
        "cold_start_bins": 10,
        "simulated_coldness_levels": [0, 1, 2, 3, 10],
        "simulated_max_interactins": 100
    },
    "amazon_magazine": {
        "dataset": "amazon_magazine",
        "item_id": "item_id:token",  # configure it based on the .item file
        "category_id": "brand:token",  
        "use_gini": True,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": -1,  # -1 implies use all users
        "seed": 42,
    },
    "douban": {
        "dataset": "douban",
        "item_id": "movie_id:token",  # configure it based on the .item file
        "category_id": "categoryID:token",  # configure it based on the .item file
        "use_gini": True,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": -1,  # -1 implies use all users
        "seed": 42,
    },
}

