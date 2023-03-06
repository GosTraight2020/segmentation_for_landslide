from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

mask = np.array(Image.open("data/landslide/1_mask.png"))
print(np.unique(mask))

