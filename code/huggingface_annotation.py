import os
import re
import sys
import math
import json
import asyncio
import warnings
import traceback
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from transformers import pipeline
import torch
import accelerate


pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b",
    device_map="auto"
)

response = pipe("Explain Bayesian networks:", max_new_tokens=100)
print(response[0]["generated_text"])