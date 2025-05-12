# HERO - DDPO Trainer

This repository contains implementation for [HERO: Human-Feedback-Efficient Reinforcement Learning for Online Diffusion Model Finetuning.](https://hero-dm.github.io/) (ICLR 2025).
The main training code is implemented in `train_hero.py`.

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Accelerate](https://github.com/huggingface/accelerate)
- [Diffusers](https://github.com/huggingface/diffusers)
- [WandB](https://wandb.ai/)
- Other dependencies as listed in `setup.py`.

## Setup

1. **Clone the repository**

   ```sh
   git clone <your-repo-url>
   cd HERO
   ```

2. **Install dependencies**

   ```sh
   pip install -e .
   cd rl4dgm
   pip install -e .
   ```


## Training

To start training, use the following command:

```sh
accelerate launch --num-processes 1 --dynamo_backend no --gpu_ids 1 train_hero.py
```

- `--num-processes 1`: Run on a single process (single GPU).
- `--dynamo_backend no`: Disables torch dynamo backend.
- `--gpu_ids 1`: Use GPU 1 (change as needed).
- `train_hero.py`: The main training script (make sure this file exists and is configured).

You may need to adjust the arguments or configuration files according to your experiment setup.

## Configuration

Training and model parameters are managed via hydra config files. See the `HERO/config/hydra_configs` for more details.

## Logging

- Training progress and metrics are logged to [Weights & Biases](https://wandb.ai/).
- Images generated during training are saved to HERO/real_human_ui_images

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Accelerate](https://github.com/huggingface/accelerate)

---
For more details, please refer to the code and comments in `ddpo_trainer.py`.
