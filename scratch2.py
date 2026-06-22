from rmq.quantization import optimal_quantizer, distortion, gradient
import numpy as np

for n in [4, 5, 10]:
    x = optimal_quantizer(n, max_iter=2000, tol=1e-12)
    g = gradient(x)
    print(f"N={n}  distortion={distortion(x):.6f}  max|grad|={np.max(np.abs(g)):.2e}")
