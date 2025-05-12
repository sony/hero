"""
Querying mechanisms
"""

import os
import random
import numpy as np
import torch

class EvaluationQueryGenerator:
    def __init__(
            self,
            seed,
        ):
        self.QUERY_ALGORITHMS = {
            "random" : self._random_queries,
            "all" : self._all_queries,
            "prob_thresh" : self._prob_thresh_queries,
            "perplexity" : self._perplexity_queries,
            "ensemble_std" : self._ensemble_std_queries,

        }
        np.random.seed(seed)


    # def get_query_indices(self, indices, query_type, n_queries=None, **kwargs):
    #     """
    #     Return array of indices to query according to the specified query_type
    #     Args:
    #         query_type (str) : how to choose the queries
    #         indices (array) : possible choices for query
    #         n_queries (int) : number of indices to choose 
    #     """
    #     assert query_type in self.QUERY_ALGORITHMS.keys(), f"query_type must be one of {self.QUERY_ALGORITHMS.keys()}. Got {query_type}."
    #     assert n_queries <= indices.shape[0], f"Requested number of queries ({n_queries}) is larger than the number of indices provided ({indices.shape[0]})."
        
    #     return self.QUERY_ALGORITHMS[query_type](indices, n_queries, **kwargs)

    def get_query_indices(self, indices, query_type, **kwargs):
        """
        Return array of indices to query according to the specified query_type
        Args:
            query_type (str) : how to choose the queries
            indices (array) : possible choices for query
            n_queries (int) : number of indices to choose 
        """
        assert query_type in self.QUERY_ALGORITHMS.keys(), f"query_type must be one of {self.QUERY_ALGORITHMS.keys()}. Got {query_type}."
        
        return self.QUERY_ALGORITHMS[query_type](indices, **kwargs)

    def _random_queries(self, indices, **kwargs):
        """
        Given an array of indices, return n_queries randomly selected indices
        """
        n_queries = kwargs["n_queries"]
        assert n_queries <= indices.shape[0], f"Requested number of queries ({n_queries}) is larger than the number of indices provided ({indices.shape[0]})."
        queries = np.random.choice(indices, size=n_queries, replace=False)
        return queries
    
    def _all_queries(self, indices, **kwargs):
        """
        Return all indices
        """
        return indices
    
    def _prob_thresh_queries(self, indices, **kwargs):
        """
        Query if probability of highest probability sample is below a threshold
        Args:
            probs (torch.Tensor) : probabilities for each sample (n_indices, n_classes)
            thresh (float) : if most confident sample has probability less than this value, query
        """
        probs = kwargs["probs"]
        thresh = kwargs["thresh"]
        max_prob = probs.max(dim=1).values
        queries = indices[max_prob < thresh]
        return queries

    def _perplexity_queries(self, indices, **kwargs):
        """
        probs (torch.Tensor) : (n_indices, n_classes) tensor of probabilities
        thresh (float) : samples with perplexity values above this thresh will be queried
        """
        probs = kwargs["probs"]
        thresh = kwargs["thresh"]
        entropy = -(torch.log(probs) * probs).sum(dim=1)
        perplexity = torch.exp(entropy)
        print("perplexity", perplexity)
        return indices[perplexity > thresh]
    
    def _ensemble_std_queries(self, indices, **kwargs):
        """
        stds (torch.Tensor) : (n_indices) tensor of std values
        thresh (float) : samples with std values above this thresh will be queried
        """
        stds = kwargs["stds"]
        thresh = kwargs["thresh"]
        return indices[stds > thresh]


class PreferenceQueryGenerator:
    def __init__(self):
        self.QUERY_ALGORITHMS = {
            "random" : self._random_query_algorithm,
            "ordered" : self._ordered_query_algorithm,
        }

    def generate_queries(self, images, query_algorithm, n_queries, prompts=None):
        """
        Given directories of images to use in query, generate queries

        Args:
            images (list(str) or torch.Tensor) : list of image directories or batch of images in Tensor form (B x C x H x W)
            query_algorithm (str) : name of query algorithm
            n_queries (int) : number of queries to generate
            prompts (list(str)) : list of prompts used to generate each image in images.
                If provided, each image in a single query comes from the same prompt
        """
        
        assert query_algorithm in self.QUERY_ALGORITHMS.keys(), f"query_algorithm must be one of {self.QUERY_ALGORITHMS.keys()}\n Got {query_algorithm}"
        query_function = self.QUERY_ALGORITHMS[query_algorithm]

        if type(images[0]) == str:
            # list out all images (dir/imgx.jpg format)
            img_paths = []

            for img_dir in images:
                img_names = os.listdir(img_dir)
                img_paths += [os.path.join(img_dir, img_name) for img_name in img_names]

            return query_function(images=img_paths, n_queries=n_queries, prompts=prompts), img_paths

        elif type(images) == torch.Tensor:

            return query_function(images=images, n_queries=n_queries, prompts=prompts)


    # def generate_queries(self, images, query_algorithm, n_queries):
    #     """
    #     Given directories of images to use in query, generate queries

    #     Args:
    #         images (list(str) or torch.Tensor) : list of image directories or batch of images in Tensor form (B x C x H x W)
    #         query_algorithm (str) : name of query algorithm
    #         n_queries (int) : number of queries to generate
    #     """
        
    #     assert query_algorithm in self.QUERY_ALGORITHMS.keys(), f"query_algorithm must be one of {self.QUERY_ALGORITHMS.keys()}\n Got {query_algorithm}"
    #     query_function = self.QUERY_ALGORITHMS[query_algorithm]

    #     if type(images[0]) == str:
    #         # list out all images (dir/imgx.jpg format)
    #         img_paths = []

    #         for img_dir in images:
    #             img_names = os.listdir(img_dir)
    #             img_paths += [os.path.join(img_dir, img_name) for img_name in img_names]

    #         return query_function(images=img_paths, n_queries=n_queries), img_paths

    #     elif type(images) == torch.Tensor:

    #         return query_function(images=images, n_queries=n_queries)

    def _ordered_query_algorithm(self, images, prompts=None, **kwargs):
        """
        Generates queries in order of the provided images:
            queries = [[0, 1], [1, 2], ... [n-1, n]]
        """

        queries = []
        for i in range(len(images)-1):
            queries.append([i, i+1])
        return queries

    def _random_query_algorithm(self, images, n_queries, prompts=None):
        """
        Randomly chooses 2 images to query

        Args:
            images (list(str) or tensor) : list of paths to images or a batch of images in the form of tensor (B x C x H x W)
            prompts (list(str)) : list of prompts used to generate each image in images

        Returns:
            queries (list(list(int))) : list of image index pairs to query
        """
        queries = []
        if type(images[0]) == str:
            n_images = len(images)
        
        elif type(images) == torch.Tensor:
            n_images = images.shape[0]
        
        indices = np.arange(n_images)

        # if prompts are not provided, return pairs of random indices
        if prompts is None:
            for _ in range(n_queries):
                queries.append(random.sample(list(indices), k=2))
            return queries
        
        # handle case wehere prompts is list(tuple(str))
        if type(prompts[0]) == tuple or type(prompts[0]) == list:
            prompts = [list(tup) for tup in prompts]
            prompts = [prompt for sublist in prompts for prompt in sublist]
        # otherwise, make sure each pair comes from the same prompt
        prompt_to_indices = {}
        for prompt in prompts:
            if not prompt in prompt_to_indices.keys():
                if (np.where(np.array(prompts) == prompt)[0]).shape[0] > 1: 
                    prompt_to_indices[prompt] = np.where(np.array(prompts) == prompt)[0]
        query_prompts = random.choices(list(prompt_to_indices.keys()), k=n_queries)
        
        for query_prompt in query_prompts:
            queries.append(random.sample(list(prompt_to_indices[query_prompt]), k=2))
        return queries, query_prompts


