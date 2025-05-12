# [ICLR'25] HERO: Human-Feedback-Efficient Reinforcement Learning for Online Diffusion Model Finetuning

This repository officially houses the official PyTorch implementation of the paper titled "HERO: Human-Feedback-Efficient Reinforcement Learning for Online Diffusion Model Finetuning", which is presented at **ICLR 2025**.

**TL;DR** HERO efficiently fintetunes text-to-image diffusion models with minimal online human feedback (<1K) for various tasks.


- Project Page: https://hero-dm.github.io/
- arXiv: https://arxiv.org/pdf/2410.05116
- OpenReview: https://openreview.net/forum?id=yMHe9SRvxk

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

The main training code is implemented in `train_hero.py`.


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

## Contacts
- Ayano Hiranaka: ayano.hiranaka@gmail.com
- Shang-Fu Chen: sam145637890@gmail.com
- Chieh-Hsin Lai: chieh-hsin.lai@sony.com
