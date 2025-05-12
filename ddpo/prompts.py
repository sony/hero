from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("ddpo.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def batch_prompts_from_file(path, n_samples, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    n_prompts = len(prompts)
    base_int = n_samples // n_prompts
    remainder = n_samples % n_prompts
    n_samples_per_prompt = [base_int] * n_prompts
    for i in range(remainder):
        n_samples_per_prompt[i] += 1
    random.shuffle(n_samples_per_prompt)
    return_prompts = []
    for i in range(len(prompts)):
        for n in range(n_samples_per_prompt[i]):
            return_prompts.append(prompts[i])
    random.shuffle(return_prompts)
    return return_prompts


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")


def cute_cats():
    return from_file("cute_cats.txt")


def ugly_cats():
    return from_file("ugly_cats.txt")

def cute_animals(n_samples):
    return batch_prompts_from_file("cute_animals.txt", n_samples)

def test_prompt():
    return from_file("test_prompt.txt")

def reasoning_prompt():
    return from_file("reasoning.txt")

def seductive_allure():
    return from_file("seductive_allure.txt")

def artistic_body():
    return from_file("artistic_body.txt")

def sensual_elegance():
    return from_file("sensual_elegance.txt")

def sexy_pose():
    return from_file("sexy_pose.txt")

def hiker():
    return from_file("hiker.txt")

def pink_carnation():
    return from_file("pink_carnation.txt")

######### our tasks ##########
def blue_rose():
    return from_file("blue_rose.txt")

def blue_rose_enhanced():
    return from_file("blue_rose_enhanced.txt")

def narcissus():
    return from_file("narcissus.txt")

def narcissus_enhanced():
    return from_file("narcissus_enhanced.txt")

def black_cat():
    return from_file("black_cat.txt")

def black_cat_enhanced():
    return from_file("black_cat_enhanced.txt")

def mountains():
    return from_file("mountains.txt")

def mountains_enhanced():
    return from_file("mountains_enhanced.txt")

def hand():
    return from_file("hand.txt")

def hand_enhanced():
    return from_file("hand_enhanced.txt")

def unsafe():
    return from_file("unsafe.txt")

def sexy():
    return from_file("sexy.txt")

####### Other tasks #######
def aesthetic_dog():
    return from_file("aesthetic_dog.txt")

def cyberpunk_cat():
    return from_file("cyberpunk_cat.txt")

def bouquet():
    return from_file("bouquet.txt")

def street():
    return from_file("street.txt")


def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata
