# hyper_params = {
#     "dataset": "ml-1m",
#     "item_id": "item_id:token",  # configure it based on the .item file
#     "category_id": "genre:token_seq",  # configure it based on the .item file
#     "use_gini": True,
#     "float64": False,
#     "depth": 1,
#     "grid_search_lamda": True,
#     "lamda": 1.0,  # Only used if grid_search_lamda == False
#     # Number of users to keep (randomly)
#     "user_support": -1,  # -1 implies use all users
#     "seed": 42,
# }

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
        "user_support": -1,  # -1 implies use all users
        "seed": 42,
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
    }
}

