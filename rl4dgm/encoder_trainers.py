
import os
import time

import json
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from rl4dgm.models.dataset import TripletDataset
from rl4dgm.models.model import LinearModel, CNNModel


def cosine_similairity_distance(x1, x2):
    cossim = (nn.functional.cosine_similarity(x1, x2, dim=0) + 1) / 2
    return 1 - cossim


class RepresentationTrainer:
    """
    Class for keeping track of image encoder trained on triplet loss
    """
    def __init__(
            self, 
            config_dict: dict,
            accelerator: Accelerator,
            seed,
            save_dir,
            trainset: TripletDataset = None,
            testset: TripletDataset = None, 
        ):
        """
        Args:
            model (nn.Module) : encoder model to train and get representation
            classifier (nn.Module) : classifier head to calculate the objectives
            trainset and testset (TripletDataset) : datasets to use for training and testing. See TripletDataset class for more detail
            config_dict : 
                keys: batch_size, shuffle, lr, n_epochs, triplet_margin, save_dir, save_every
        """
        default_config = {
            "batch_size" : 32,
            "shuffle" : True,
            "lr" : 1e-6,
            "n_epochs" : 50,
            "triplet_margin" : 0.5,
            "save_every" : 50,
            "input_dim" : 4096,
            "hidden_dims" : [2048, 1024],
            "output_dim" : 512,
            "name" : "representation_encoder",
        }

        # create directory to save config and model checkpoints 
        # assert "save_dir" in config_dict.keys(), "config_dict is missing key: save_dir"
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
            
        # populate the config with default values if values are not provided
        for key in default_config:
            if key not in config_dict.keys():
                config_dict[key] = default_config[key]
        # hidden_dim is ListConfig type if speficied in hydra config. Convert to list so it can be dumped to json
        config_dict["hidden_dims"] = [dim for dim in config_dict["hidden_dims"]]

        print("Initializing RepresentationTrainer with following configs\n", config_dict)
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(config_dict, f)
            print("saved RepresentationTrainer config to", os.path.join(save_dir, "train_config.json"))
                
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

        self.accelerator = accelerator
        self.device = accelerator.device
        self.model = CNNModel(
            channels=4,
            size=64,
            device=self.device,
        )
        self.classifier = LinearModel(
            input_dim=config_dict["input_dim"],
            hidden_dims=config_dict["hidden_dims"],
            output_dim=config_dict["output_dim"],
            device=self.device,
        )
        self.trainset = trainset
        self.testset = testset
        self.config = config_dict
        self.name = config_dict["name"]

        # Initialize dataloaders
        self.dataloaders = {}
        self.initialize_dataloaders(trainset, testset)
        
        # Initialize optimizer and loss criteria
        combined_params = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(combined_params, lr=self.config["lr"])

        self.criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=cosine_similairity_distance,
            margin=self.config["triplet_margin"],
        )

        self.n_total_epochs = 0
        self.n_calls_to_train = 0 # how many times the train function has been called

        self.start_time = time.time()

    def initialize_dataloaders(
            self, 
            trainset : TripletDataset = None, 
            testset : TripletDataset = None,
        ):
        if trainset is not None:
            self.trainset = trainset
            self.trainloader = DataLoader(
                trainset, 
                batch_size=self.config["batch_size"], 
                shuffle=self.config["shuffle"],
                generator=self.generator,
            )
            self.dataloaders["train"] = self.trainloader
        if testset is not None:
            self.testset = testset
            self.testloader = DataLoader(
                testset, 
                batch_size=self.config["batch_size"], 
                shuffle=self.config["shuffle"],
                generator=self.generator,
            )
            self.dataloaders["test"] = self.testloader

    def train_model(self):
        """
        Trains an image encoder using triplet loss
        """
        self.n_calls_to_train += 1
        for epoch in range(self.config["n_epochs"]):
            running_losses = []
            if epoch % 100 == 0:
                print("RepresentationEncoder training epoch", epoch)
            for anchor_features, _, positive_features, negative_features in self.trainloader:
                self.optimizer.zero_grad()
                anchor_out = self.classifier(self.model(anchor_features))
                positive_out = self.classifier(self.model(positive_features))
                negative_out = self.classifier(self.model(negative_features))

                loss = self.criterion(anchor_out, positive_out, negative_out)
                loss.backward()
                self.optimizer.step()
                running_losses.append(loss.item())

                self.accelerator.log({
                    f"{self.name}_epoch" : self.n_total_epochs,
                    f"{self.name}_loss" : loss.item(),
                    f"{self.name}_lr" : self.config["lr"],
                    f"{self.name}_clock_time" : time.time() - self.start_time,
                })

            self.n_total_epochs += 1
        
        print(f"encoder treained for {self.n_calls_to_train} times")
        # save checkpoint
        if (self.n_calls_to_train > 0) and (self.n_calls_to_train % self.config["save_every"]) == 0:
            self.save_model_ckpt()
            # model_save_path = os.path.join(self.save_dir, f"epoch{self.n_total_epochs}.pt")
            # torch.save(self.model.state_dict(), model_save_path)
            # print("TripletEncoder model checkpoint saved to", model_save_path)

    def save_model_ckpt(self):
        model_save_path = os.path.join(self.save_dir, f"epoch{self.n_calls_to_train}.pt")
        torch.save(self.model.state_dict(), model_save_path)
        print("Entropy encoder model checkpoint saved to", model_save_path)