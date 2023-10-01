import random 
import numpy as np
import torch 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# count # of param for a list of module
def count_param(module_list):
    return sum(x.numel() for module in module_list for x in module.parameters()) / 10**6
    
# display the peak memory of cuda
def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

def anal_tensor(tensor, name):
    sent = f" name: {name} mean: {tensor.mean().item()}  std: {tensor.std().item()}  min: {tensor.min().item()}  max: {tensor.max().item()}"
    print(sent)
