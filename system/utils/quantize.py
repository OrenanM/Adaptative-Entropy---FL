import torch.nn as nn
import torch
def dequantize_dynamic(model, net_quantized_dynamic):
    """dequantiza o modelo, realizada no servidor"""
    for name_quantized, module_quantized in net_quantized_dynamic.named_modules():
        if isinstance(module_quantized, nn.quantized.dynamic.Linear):
            for name, module in model.named_modules():
                if name_quantized == name:
                    module.weight.data = module_quantized.weight().dequantize()
                    module.bias.data = module_quantized.bias().dequantize()

def quantize_dynamic(self):
    """Quantiza o modelo, forma que é enviado para o servidor"""
    net_quantized_dynamic = torch.ao.quantization.quantize_dynamic(
        self.model,
        {torch.nn.Linear},  # Camadas a serem quantizadas dinamicamente
        dtype=torch.float16  # Precisão
    )
    return net_quantized_dynamic