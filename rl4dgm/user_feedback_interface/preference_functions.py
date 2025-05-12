"""
Preference functions to provide AI feedback
"""
import os

import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
import torch
from torchvision.transforms.functional import to_pil_image


"""
Use finetuned pickscore to calculate rewards
"""
def PickScore(**kwargs):
    processor = kwargs["processor"]
    model = kwargs["model"]
    images = kwargs["images"]
    prompt = kwargs["prompt"]
    device = kwargs["device"]
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        # embed
        try:
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        except:
            image_embs = model.module.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.module.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.module.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    scores = scores.squeeze().cpu()
    return scores.squeeze().cpu()
