import torch

def print_total_parameters(model):
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = total_trainable + total_non_trainable
    print(f'Total parameters: {total:,}')
    print(f'Total trainable parameters: {total_trainable:,}')
    print(f'Total non-trainable parameters: {total_non_trainable:,}')
    