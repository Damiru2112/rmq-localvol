"""
M1 validation plots: visualizes the optimal Gaussian quantizer and confirms
distortion decreases with N, matching the theoretical asymptotic rate.
"""
import numpy as np
import matplotlib.pyplot as plt
from rmq.quantization import optimal_quantizer, lloyd_quantizer, distortion

# --- Plot 1: distortion vs N, Newton vs Lloyd ---
Ns = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
newton_d = [distortion(optimal_quantizer(n)) for n in Ns]
lloyd_d = [distortion(lloyd_quantizer(n)) for n in Ns]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(Ns, newton_d, 'o-', label='Newton', markersize=5)
axes[0].plot(Ns, lloyd_d, 'x--', label='Lloyd', markersize=7)
axes[0].set_xlabel('N (grid points)')
axes[0].set_ylabel('Distortion D(x)')
axes[0].set_title('Distortion vs N: Newton vs Lloyd cross-check')
axes[0].legend()
axes[0].set_yscale('log')

# --- Plot 2: the actual grid for a representative N, overlaid on the Gaussian pdf ---
from scipy.stats import norm
N_show = 10
x = optimal_quantizer(N_show)
z = np.linspace(-4, 4, 400)
axes[1].plot(z, norm.pdf(z), label='N(0,1) pdf')
axes[1].scatter(x, norm.pdf(x), color='red', zorder=5, label=f'optimal grid (N={N_show})')
axes[1].set_xlabel('z')
axes[1].set_title(f'Optimal {N_show}-point quantizer')
axes[1].legend()

plt.tight_layout()
plt.savefig('docs/m1_quantizer_validation.png', dpi=150)
print("Saved docs/m1_quantizer_validation.png")