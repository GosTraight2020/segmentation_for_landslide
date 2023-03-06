from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

for i in range(1, 3800):
    mask = np.array(Image.open(f"data/landslide/{i}_mask.png"))
    mask[mask==255] = 1
    mask = Image.fromarray(mask)
    mask = torch.from_numpy(np.array(mask)).long()
    assert len(mask.shape) == 2, "i"

