# Python program for Huffman Coding
import heapq
import torch
import numpy as np
import torch.nn as nn
import huffman

def calculate_model_encoding(model):
    with torch.no_grad():                                        
        # Concatenando todos os pesos (ignorando os biases)
        total_size = 0
        for name, module in model.named_modules():
            
            if isinstance(module, nn.Linear):
                all_weights = module.weight.data.view(-1)

                all_weights_np = all_weights.to('cpu').detach().numpy()
                unique, counts = np.unique(all_weights_np, return_counts=True)
                huffman_table = huffman.codebook(list(zip(unique, counts)))

                size_table = 0
                size_model_enconding = 0
                for value in huffman_table.values():
                    size_table += (8 + len(value))

                for weight, f in zip(all_weights_np, counts):
                    size_model_enconding += len(huffman_table[weight])*f
                #print(f'table size {name}: {size_table/8}')
                #print(f'transmition size {name}: {size_model_enconding/8}')
                total_size += (size_model_enconding + size_table)/8

        return total_size

def calculate_model_size(model, include_grad=False):
    size_model = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # Considera apenas camadas lineares
            for param in module.parameters():
                size_model += param.numel() * param.element_size()
                if include_grad and param.grad is not None:
                    size_model += param.grad.numel() * param.grad.element_size()
    return size_model

def calculate_model_quantized_size(model, include_grad=False):
    size_model = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # Considera apenas camadas lineares
            for param in module.parameters():
                size_model += param.numel()
                if include_grad and param.grad is not None:
                    size_model += param.grad.numel() * param.grad.element_size()
    return size_model
