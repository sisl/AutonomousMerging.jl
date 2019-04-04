## Test calling the model checker policy from python 
import numpy as np
from julia import Main as jl

jl.eval("include(\"model_checker.jl\")")

s = np.zeros((5,))

import time 
acts = jl.safe_actions(s, 0.9)
ts = time.time()
acts = jl.safe_actions(s, 0.9)
te = time.time() 
print(te - ts)
