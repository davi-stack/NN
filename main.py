import numpy as np
import random

class Layer:
    def __init__(self, intup_size, output_size):
        self.bytes = random.random(0, 250)
        self.wheiths = np.random.rand(output_size)
        
    
    