class Config:
    def __init__(self):
        self.faults = {
            "cuda:0": {
                "epochN": 0,
                "stepN": 1,
                "faultsN": 100,
                "duration": 1,
                "nameOfLayers": ["transformer.h.3.attn.attention.k_proj"],
                # "weight", "gradient", "data"
                "type": "data",
                # "weight", "bias"
                "bias_weight": "weight",
                # "impulsFunction", "randomFunction", "zeroFunction", "valueFunction", "magnitudeFunction"
                "faultFunction": "valueFunction",
                "faultValue": 0
            },
        }



