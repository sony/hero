from dataclasses import dataclass

@dataclass
class QueryConfig:
    feedback_agent : str = "human"

    # type of feedback to use (see FeedbackInterface class for available types)
    feedback_type : str = "positive-and-best-indices"

    # query method
    query_type : str = "all"

    query_everything_fisrt_iter : bool = False

    # Only used in random query
    n_feedback_per_query : int = 64

    # Only used in active query methods where number of queries vary in each loop.
    # If not enough queries are chosen, choose queries randomly to meet this minimum requirement    
    min_n_queries : int = 0

    # whether to only enforce min number of queries during warmup 
    only_enforce_min_queries_during_warmup : bool = False

    # Config for using real human feedback
    use_best_image : bool = True