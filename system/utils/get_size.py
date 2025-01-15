# Python program for Huffman Coding
import heapq
import torch
import numpy as np
import torch.nn as nn

class Node:
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(chars, freq):
  
    # Create a priority queue of nodes
    priority_queue = [Node(char, f) for char, f in zip(chars, freq)]
    heapq.heapify(priority_queue)

    # Build the Huffman tree
    while len(priority_queue) > 1:
        left_child = heapq.heappop(priority_queue)
        right_child = heapq.heappop(priority_queue)
        merged_node = Node(frequency=left_child.frequency + right_child.frequency)
        merged_node.left = left_child
        merged_node.right = right_child
        heapq.heappush(priority_queue, merged_node)

    return priority_queue[0]

def generate_huffman_codes(node, code="", huffman_codes={}):
    if node is not None:
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        generate_huffman_codes(node.left, code + "0", huffman_codes)
        generate_huffman_codes(node.right, code + "1", huffman_codes)

    return huffman_codes

def calculate_huffman_model(model):

    all_weights = torch.cat([param.view(-1) for param in model.parameters()])

    all_weights_np = all_weights.detach().numpy()
    unique, counts = np.unique(all_weights_np, return_counts=True)

    weights = unique.tolist()
    freq = counts.tolist()

    # Build the Huffman tree
    root = build_huffman_tree(weights, freq)

    # Generate Huffman codes
    huffman_codes = generate_huffman_codes(root)

    size_model = 0
    size_unique = 0
    for code, f in zip(huffman_codes.values(), freq):
        size_unique += len(code)
        size_model += len(code) * f
    
    size_table = len(unique)*32+size_unique
    
    return size_model/8, size_table/8

def calculate_model_size(model, include_grad=False):
    size_model = 0
    for param in model.parameters():
        size_model += param.numel() * param.element_size()
        if include_grad:
            size_model += param.grad.numel() * param.grad.element_size()
    return size_model

if __name__ == "__main__":
    model = nn.Linear(1, 26)
    size_model = calculate_huffman_model(model)
    print(size_model)

    # 4 2bits 00, 01, 10, 11
    # 4 * 32 = 128 bits