from setuptools import setup, find_packages

setup(
    name="rl4dgm",
    packages=[
        package for package in find_packages() if package.startswith("rl4dgm")
    ],
    description="Realtime finetuning of text-to-image diffusion models with human feedback",
)