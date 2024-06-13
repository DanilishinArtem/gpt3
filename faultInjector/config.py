class Config:
    def __init__(self):
        self.info = False
        self.faults = {
            "cuda:0": {
                "epochN": 0,
                "stepN": 1,
                "faultsN": 1000000,
                "duration": 50,
                "nameOfLayers": [
                    # "transformer.h.22.attn.attention.k_proj" # this is for singleGPU
                    "module.transformer.h.22.attn.attention.k_proj" # this is for multyGPU
                ],
                # "weight", "gradient", "data"
                "type": "gradient",
                # "weight", "bias"
                "bias_weight": "weight",
                # "impulsFunction", "randomFunction", "zeroFunction", "valueFunction", "magnitudeFunction"
                "faultFunction": "valueFunction",
                "faultValue": 1
            },
        }



