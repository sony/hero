import os
import argparse

import torch
import itertools
import clip
from PIL import Image


def main(args):
    # get model and preprocessor
    model, preprocess = clip.load("ViT-B/32", device=args.device)
    cos = torch.nn.CosineSimilarity(dim=0)
    
    if args.parent_dir is not None:
        img_dirs = [] # subdirectories containing image (jpg) file(s)
        for root, dirs, files in os.walk(args.parent_dir):
            if any(file.lower().endswith(".jpg") for file in files):
                img_dirs.append(root)
        img_dirs = sorted(img_dirs)

    else: 
        img_dirs = [args.img_dir]

    diversities = []
    for img_dir in img_dirs:
        diversity = compute_diversity(model, preprocess, cos, img_dir)
        diversities.append(diversity)

    print("List of directories evaluated\n", img_dirs)
    print("\nList of diversity values\n", diversities)

def compute_diversity(model, preprocess, cos, img_dir):
    # get all images in the directory as a list of PIL images
    img_features = []
    for img_path in os.listdir(img_dir):
        processed_img = preprocess(Image.open(os.path.join(img_dir, img_path))).unsqueeze(0).to(args.device)
        img_features.append(model.encode_image(processed_img))

    cossim_sum = 0
    for comb in itertools.combinations(img_features, r=2):
        cossim = cos(comb[0][0], comb[1][0]).item()
        cossim = (cossim + 1) / 2
        cossim_sum += cossim

    n = len(img_features)
    in_batch_diversity = 1 - (2 / (n*(n-1))) * cossim_sum
    
    print(img_dir)
    print(f"in batch diversity of {n} images is : {in_batch_diversity}\n")

    return in_batch_diversity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-dir", type=str, help="path to directory with images to compute in batch diversity")
    parser.add_argument("--parent-dir", type=str)
    
    args = parser.parse_args()
    main(args)