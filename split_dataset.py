from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tqdm 

root = "data/landslide"
has_landslide = []
no_landslide = []
for i in tqdm.tqdm(range(1, 3800)):
    image_name = f"{i}_image.png"
    mask_name = f"{i}_mask.png"
    
    mask = Image.open(os.path.join(root, mask_name))
    mask = torch.from_numpy(np.array(mask)).long()
    assert len(mask.unique()) == 1 or len(mask.unique()) == 2
    if len(mask.unique()) == 2:
        has_landslide.append((image_name, mask_name))
    else:
        no_landslide.append((image_name, mask_name))
        
random.shuffle(has_landslide)
random.shuffle(no_landslide)

test_set = has_landslide[:500] + no_landslide[500:]
unlabel = has_landslide[500:1000] + no_landslide[500:1000]
label = has_landslide[1000:1050] + no_landslide[1000:1050]

with open("partitions/landslide/labeled.txt", "w") as f:
    for pair in label:
        f.write(f"{pair[0]} {pair[0]}\n")

with open("partitions/landslide/unlabeled.txt", "w") as f:
    for pair in unlabel:
        f.write(f"{pair[0]} {pair[0]}\n")
        
with open("partitions/landslide/val.txt", "w") as f:
    for pair in test_set:
        f.write(f"{pair[0]} {pair[0]}\n")

    
    