import torch
from faultInjector.config import Config
from faultInjector.faultFunctioon import FaultFunction

config = Config()

def functionSetter(name: str, input: torch.tensor):
    input = input[0]
    modified_input = input.clone()
    faultFunction = config.faults[str(input.device)]["faultFunction"]
    if faultFunction == "impulsFunction":
        FaultFunction.impulsFunction(modified_input)
    if faultFunction == "randomFunction":
        FaultFunction.randomFunction(modified_input)
    if faultFunction == "zeroFunction":
        FaultFunction.zeroFunction(modified_input)
    if faultFunction == "valueFunction":
        FaultFunction.valueFunction(modified_input)
    if faultFunction == "magnitudeFunction":
        FaultFunction.magnitudeFunction(modified_input)
    return modified_input

class WeightHook:
    def __init__(self, name: str, layer: torch.nn.Module):
        self.name = name
        self.layer = layer
        self.counter = 0
        self.duration = 0

    def hook(self, module, input, output):
        self.counter += 1
        if str(module.weight.device) in config.faults:
            temp = config.faults[str(module.weight.device)]
            if self.counter >= temp["stepN"] and self.duration <= temp["duration"] and self.name in temp["nameOfLayers"]:
                self.duration += 1
                print("fault (weight) for layer " + self.name + " was injected at time " + str(self.counter))
                module.weight = torch.nn.Parameter(functionSetter("valueFunction", (module.weight,)))


class GradientHook:
    def __init__(self, name: str, layer: torch.nn.Module):
        self.name = name
        self.layer = layer
        self.counter = 0
        self.duration = 0

    def hook(self, module, grad_input, grad_output):
        self.counter += 1
        if str(grad_input[0].device) in config.faults:
            temp = config.faults[str(grad_input[0].device)]
            if self.counter >= temp["stepN"] and self.duration <= temp["duration"] and self.name in temp["nameOfLayers"]:
                self.duration += 1
                print("fault (gradient) for layer " + self.name + " was injected at time " + str(self.counter))
                modified_grad_input = functionSetter("valueFunction", (grad_input[0],))
                return (modified_grad_input,) + grad_input[1:]


class DataHook:
    def __init__(self, name: str, layer: torch.nn.Module):
        self.name = name
        self.layer = layer
        self.counter = 0
        self.duration = 0

    def hook(self, module, input, output):
        self.counter += 1
        if str(input[0].device) in config.faults:
            temp = config.faults[str(input[0].device)]
            if self.counter >= temp["stepN"] and self.duration <= temp["duration"] and self.name in temp["nameOfLayers"]:
                self.duration += 1
                print("fault (data) for layer " + self.name + " was injected at time " + str(self.counter))
                return functionSetter("valueFunction", (input[0],))