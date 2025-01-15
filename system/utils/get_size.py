# Python program for Huffman Coding
import heapq
import torch
import numpy as np
import torch.nn as nn
import huffman

def calculate_model_encoding(model):

    all_weights = torch.cat([param.view(-1) for param in model.parameters()])

    all_weights_np = all_weights.detach().numpy()
    unique, counts = np.unique(all_weights_np, return_counts=True)

    huffman_table = huffman.codebook(list(zip(unique, counts)))

    size_table = 0
    size_model_enconding = 0
    for value in huffman_table.values():
        size_table += (32 + len(value))

    for weight in all_weights_np:
        size_model_enconding += len(huffman_table[weight])

    return size_model_enconding/8, size_table/8

def calculate_model_size(model, include_grad=False):
    size_model = 0
    for param in model.parameters():
        size_model += param.numel() * param.element_size()
        if include_grad:
            size_model += param.grad.numel() * param.grad.element_size()
    return size_model
