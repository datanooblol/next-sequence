# This is done

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from torch.utils.data import Dataset

class CoreDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class TaskType(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

@dataclass
class Feature:
    name:str
    task:TaskType
    vocab_size:int = 0
    is_linear:bool = False