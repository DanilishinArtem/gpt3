import torch
from faultInjector.config import Config

config = Config()

class FaultFunction:
    @staticmethod
    def impulsFunction(input: torch.tensor):
        faultValue = config.faults[str(input.device)]["faultValue"]
        random_indices = torch.randint(0, input.numel(), (1,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        print("changing tensor")
        print(input[random_positions])
        input[random_positions] = faultValue
        print("to tensor")
        print(input[random_positions])

    @staticmethod
    def randomFunction(input: torch.tensor):
        faultsN = config.faults[str(input.device)]["faultsN"]
        random_indices = torch.randint(0, input.numel(), (faultsN,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        print("changing tensor")
        print(input[random_positions])
        input[random_positions] = torch.rand_like(input[random_positions])
        print("to tensor")
        print(input[random_positions])

    @staticmethod
    def zeroFunction(input: torch.tensor):
        faultsN = config.faults[str(input.device)]["faultsN"]
        random_indices = torch.randint(0, input.numel(), (faultsN,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        print("changing tensor")
        print(input[random_positions])
        input[random_positions] = 0
        print("to tensor")
        print(input[random_positions])

    @staticmethod
    def valueFunction(input: torch.tensor):
        faultValue = config.faults[str(input.device)]["faultValue"]
        faultsN = config.faults[str(input.device)]["faultsN"]
        random_indices = torch.randint(0, input.numel(), (faultsN,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        print("changing tensor")
        print(input[random_positions])
        input[random_positions] = faultValue
        print("to tensor")
        print(input[random_positions])

    @staticmethod
    def magnitudeFunction(input: torch.tensor):
        faultValue = config.faults[str(input.device)]["faultValue"]
        faultsN = config.faults[str(input.device)]["faultsN"]
        random_indices = torch.randint(0, input.numel(), (faultsN,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        print("changing tensor")
        print(input[random_positions])
        input[random_positions] = input[random_positions] * faultValue
        print("to tensor")
        print(input[random_positions])