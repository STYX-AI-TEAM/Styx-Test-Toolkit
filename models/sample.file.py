import torch
from transformers import pipeline


def gpt2():
  device = 0 if torch.cuda.is_available() else -1

  return pipeline("text-generation", model="openai-community/gpt2",device=device)
