from dataclasses import dataclass, field
from typing import Optional


######### NEW ENCODER ##############
@dataclass
class HumanEncoderConfig:
    batch_size : int = 2048
    shuffle : bool = True
    lr : float = 1e-5
    n_epochs : int = 500
    save_dir : Optional[str] = None
    save_every : int = 1
    agent1_triplet_margin : float = 1.0
    agent2_triplet_margin : float = 1.0
    agent1_loss_weight : float = 1.0
    agent2_loss_weight : float = 0.25
    input_dim : int = 2048
    n_hidden_layers : int = 1
    hidden_dims : list[int] = field(default_factory=lambda: [2048, 1024])
    output_dim : int = 512
    n_data_needed_for_training : int = 128
    n_warmup_epochs: int = 0
    name : str = "human_encoder"