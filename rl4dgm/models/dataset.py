
import random
import torch 
import numpy as np
from torch.utils.data import Dataset
from torch.distributions import Normal

def get_weighted_sample(weights, k):
    """
    Given weights, sample an index with probability proportional to the weights
    Args:
        weights
        k (int) : number of samples to draw
    """
    weights_cumsum = torch.cumsum(weights, dim=0)
    m = weights_cumsum[-1]
    indices = []
    i = 0
    while i < k:
        rand = m * torch.rand(1).item()
        idx = torch.searchsorted(sorted_sequence=weights_cumsum, input=rand, side='right')
        if idx in indices:
            continue
        indices.append(idx)
        i +=1
    return torch.tensor(indices)

class TripletDataset(Dataset):
    def __init__(
        self,
        features,
        positive_indices,
        negative_indices,
        best_sample_feature,
        device,
        is_train=False,
        sampling_method="default",
    ):
        self.features = features.float().to(device)
        self.best_sample_feature = best_sample_feature.float().to(device)
        self.is_train = is_train
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices

        self.sampling_method = sampling_method
        sampling_methods = [
            "default",
        ]
        assert sampling_method in sampling_methods, f"Sampling method must be one of {sampling_methods}. Got {sampling_method}"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # get anchor sample
        anchor_feature = self.best_sample_feature.squeeze()

        ##############################################################################################
        # Sample positive (and negative) samples randomly from top 10% most (least) similar samples
        ##############################################################################################
        
        if self.sampling_method == "default":
            positive_index = random.choice(self.positive_indices)
            negative_index = random.choice(self.negative_indices)
            positive_feature = self.features[positive_index]
            negative_feature = self.features[negative_index]
        return anchor_feature, 0, positive_feature, negative_feature # 0 is dummy and not used


class HumanDatasetSimilarity():

    """
    Note : This class is used to accumulate human data while training the similarity-based pipeline
        It is NOT a torch dataset. 
    """
    def __init__(
        self, 
        device
    ):
        self.clear_data()
        self.device = device
        
    def clear_data(self,):
        print("clear data called")
        self._sd_features = None
        self._n_data = 0
        self._positive_indices = None
        self._negative_indices = None

    def add_data(self, sd_features, positive_indices, negative_indices):
        """
        Add sd_features, human rewards, and ai rewards to the human dataset
        """
        # cast inputs to tensors, correct dimension, and put on correct device
        assert isinstance(sd_features, torch.Tensor), f"sd_features must be a torch.Tensor"
        sd_features = sd_features.to(self.device)
        sd_features = self._make_2d(sd_features)

        if not isinstance(positive_indices, np.ndarray):
            positive_indices = np.array(positive_indices)

        # if this is the first set of data added, initialize
        if self._sd_features is None:
            self._sd_features = sd_features
            self._positive_indices = positive_indices
            self._negative_indices = negative_indices
        
        # otherwise concatenate new data
        else:
            self._positive_indices = np.concatenate([self._positive_indices, positive_indices + self._sd_features.shape[0]])
            self._negative_indices = np.concatenate([self._negative_indices, negative_indices + self._sd_features.shape[0]])
            self._sd_features = torch.cat([self._sd_features, sd_features], dim=0)

        # update number of data
        self._n_data += sd_features.shape[0]

    def _make_2d(self, input_tensor):
        """
        If the input is 1d, add a dimension to make it 2d
        """
        if input_tensor.dim() == 1:
            return input_tensor[None,:]
        elif input_tensor.dim() == 2:
            return input_tensor
        else:
            raise Exception("input tensor dimension is incorrect")
    
    @property
    def positive_features(self):
        return self._sd_features[self._positive_indices]
    
    @property
    def negative_features(self):
        return self._sd_features[self._negative_indices]

    @property
    def n_data(self,):
        return self._n_data

    @property
    def sd_features(self,):
        return self._sd_features
    
    @property
    def positive_indices(self):
        return self._positive_indices
    
    @property
    def negative_indices(self):
        return self._negative_indices


 
 