import numpy as np
import json

def input_fn(serialized_input, content_type):
    if content_type=="application/json":
        deserialized_input = json.loads(serialized_input)
        return deserialized_input
    else:
        pass