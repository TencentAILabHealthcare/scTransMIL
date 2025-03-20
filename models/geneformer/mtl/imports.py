import functools
import gc
import json
import os
import pickle
import sys
import warnings
from enum import Enum
from itertools import chain
from typing import Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BatchEncoding,
    BertConfig,
    BertModel,
    DataCollatorForTokenClassification,
    SpecialTokensMixin,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_scheduler,
)
from transformers.utils import logging, to_py_obj

from .collators import DataCollatorForMultitaskCellClassification

# local modules
from .data import get_data_loader, preload_and_process_data
from .model import GeneformerMultiTask
from .optuna_utils import create_optuna_study
from .utils import save_model
