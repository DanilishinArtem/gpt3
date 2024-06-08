import torch
from faultInjector.hooks import DataHook, WeightHook, GradientHook
from faultInjector.config import Config

config = Config()

class HookSetter:
    def __init__(self, model: torch.nn.Module):
        self.setter(model)

    def setter(self, model: torch.nn.Module):
        type = config.faults[str(model.device)]["type"]
        bias_weight = config.faults[str(model.device)]["bias_weight"]
        for name, layer in model.named_modules():
            if hasattr(layer, bias_weight):
                if type == "weight":
                    layer.register_forward_hook(WeightHook(name, layer).hook)
                if type == "data":
                    layer.register_forward_hook(DataHook(name, layer).hook)
                if type == "gradient":
                    layer.register_backward_hook(GradientHook(name, layer).hook)
                print("registered hook for layer " + name)