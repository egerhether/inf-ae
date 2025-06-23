hyper_params = {
    "steam": {
        "dataset": "steam",
        "item_id": "id",  # configure it based on the .item file
        "category_id": "tags",  # configure it based on the .item file
        "diversity_metrics": True,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": 20000,  # -1 implies use all users
        "seed": 42,
        "gen": "weak"
    },
    "ml-20m": {
        "dataset": "ml-20m",
        "item_id": "item_id:token",  # configure it based on the .item file
        "category_id": "type:token_seq",  # configure it based on the .item file
        "diversity_metrics": True,
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
        "diversity_metrics": True,
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
        "diversity_metrics": True,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": False, # lambda (below) only used if grid_search_lamda == False
        "lamda": 5.0,  # Found through grid search in a strong generalizaion setting seeds 42, 41, 40
        # Number of users to keep (randomly)
        "user_support": -1,  # -1 implies use all users
        "seed": 42,
        "neg_sampling_strategy": "positive2",
        "gen": "strong",
        "cold_start_bins": 5,
        "simulated_coldness_levels": [0, 1, 3, 5, 10, 15],
        "simulated_max_interactins": 40
    },
    "amazon_magazine": {
        "dataset": "amazon_magazine",
        "item_id": "item_id:token",  # configure it based on the .item file
        "category_id": "brand:token",  
        "diversity_metrics": True,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": -1,  # -1 implies use all users
        "seed": 42,
        "gen": "weak"
    },
    "douban": {
        "dataset": "douban",
        "item_id": "movie_id:token",  # configure it based on the .item file
        "category_id": "categoryID:token",  # configure it based on the .item file
        "diversity_metrics": False,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": -1,  # -1 implies use all users
        "seed": 42,
        "neg_sampling_strategy": "positive2",
        "gen": "weak"
    },
    "netflix": {
        "dataset": "netflix",
        "item_id": "item_id:token",  # configure it based on the .item file
        "category_id": "genre:token_seq",  # configure it based on the .item file
        "diversity_metrics": False,
        "float64": False,
        "depth": 1,
        "grid_search_lamda": True,
        "lamda": 1.0,  # Only used if grid_search_lamda == False
        # Number of users to keep (randomly)
        "user_support": 10000,  # -1 implies use all users
        "seed": 42,
        "batch_size": 20000,
        "neg_sampling_strategy": "positive2",
        "gen": "weak"
    }
}