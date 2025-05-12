from setuptools import setup, find_packages

setup(
    name="ddpo",
    version="0.0.1",
    packages=["ddpo"],
    python_requires=">=3.10",
    install_requires=[
        "ml-collections",
        "absl-py",
        "huggingface-hub==0.25.0",
        "diffusers[torch]==0.25.1",
        "accelerate>=0.17",
        "wandb",
        "torchvision",
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "transformers>=4.30.2",
        "datasets==2.16.1",
        "deepspeed==0.13.1",
        "fire==0.4.0",
        "hydra-core==1.3.2",
        "rich==13.3.2",
        "submitit==1.4.5",
    ],
)
