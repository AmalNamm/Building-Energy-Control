import torch
import os
model_path="final_model.pt"
if model_path:
    print("checking the path",os.getcwd())
    checkpoint = torch.load(model_path)