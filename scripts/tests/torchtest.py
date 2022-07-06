# Simple file to make sure GPUs are being used on SCC
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
