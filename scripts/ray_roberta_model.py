import torch
from transformers import AutoTokenizer, RobertaModel


# Base model inference for feature vector
class RobertaCL4InferenceRay:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = 
