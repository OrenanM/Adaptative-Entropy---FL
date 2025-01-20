import torch

# Função para calcular a entropia usando PyTorch
def calculate_entropy_with_grad(model):
    all_params = torch.cat([p.view(-1) for p in model.parameters()])  # Concatena todos os parâmetros em um vetor único
    
    # Criando os limites dos bins
    min_val, max_val = all_params.min(), all_params.max()
    
    # Calculando o histograma (os valores são os centros dos bins)
    hist = torch.histc(all_params, bins=2**8)
    
    # Normalizando para obter uma distribuição de probabilidade
    hist_prob = hist / hist.sum()

    # Calculando a entropia
    entropy = -torch.sum(hist_prob * torch.log(hist_prob+1e-10))
    return entropy