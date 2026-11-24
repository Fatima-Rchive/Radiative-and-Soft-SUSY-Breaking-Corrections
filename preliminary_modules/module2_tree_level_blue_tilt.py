"""
Show that tree-level hybrid inflation gives ns > 1 (blue tilt).
This is the problem the paper exists to fix.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.table import Table

# Create module-specific figures folder
script_dir = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(script_dir, "mod2_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Figures will be saved to: {FIGURES_DIR}")

Mp = 1.0

def V_tree(phi, kappa, M, m):
    return (kappa / 4.0) * M**4 + 0.5 * m**2 * phi**2

def dV_tree(phi, m):
    return m**2 * phi

def epsilon(phi, Vfun, dVfun):
    return 0.5 * Mp**2 * (dVfun(phi) / Vfun(phi))**2

def eta(phi, Vfun, dVfun, h=1e-6):
    V = Vfun(phi)
    d2V = (dVfun(phi + h) - dVfun(phi - h)) / (2 * h)
    return Mp**2 * d2V / V

def ns_from_eps_eta(eps, eta):
    return 1.0 - 6.0 * eps + 2.0 * eta

print("\nMODULE 2: THE BLUE TILT PROBLEM")
print("Tree-level hybrid inflation gives ns > 1")

# Parameters
kappa, M, m = 1.0, 1e-2, 1e-6

# Define functions for this parameter set
Vfun = lambda p: V_tree(p, kappa, M, m)
dVfun = lambda p: dV_tree(p, m)

print("\nPARAMETERS:")
print(f"  kappa = {kappa}")
print(f"  M = {M:.2e} Mp")
print(f"  m = {m:.2e} Mp")

# Generate results table
print("\nRESULTS:")
print(f"  {'phi (Mp)':<12} {'epsilon':<12} {'eta':<12} {'ns':<12} {'Blue Tilt?'}")
print("-" * 60)

results_data = []
phis_test = [1e-3, 1e-2, 1e-1, 5e-1]
for phi in phis_test:
    eps = epsilon(phi, Vfun, dVfun)
    et = eta(phi, Vfun, dVfun)
    ns = ns_from_eps_eta(eps, et)
    blue_tilt = "✓" if ns > 1 else "✗"
    results_data.append((phi, eps, et, ns, blue_tilt))
    print(f"  {phi:<12.3e} {eps:<12.3e} {et:<12.3e} {ns:<12.4f} {blue_tilt}")

# Create table image
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')

table_data = [
    [r'$\phi$ ($M_P$)', r'$\epsilon$', r'$\eta$', r'$n_s$', 'Blue Tilt?']
]
for phi, eps, et, ns, status in results_data:
    table_data.append([f"{phi:.3e}", f"{eps:.3e}", f"{et:.3e}", f"{ns:.4f}", status])

table = ax.table(cellText=table_data, loc='center', cellLoc='center', 
                 colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# Color status column
for row_idx in range(1, 5):
    status = table_data[row_idx][4]
    if status == '✓':
        table[(row_idx, 4)].set_facecolor('#90ee90')
        table[(row_idx, 4)].set_text_props(color='green', fontweight='bold')
    else:
        table[(row_idx, 4)].set_facecolor('#ffcccb')
        table[(row_idx, 4)].set_text_props(color='red', fontweight='bold')
    
    if row_idx % 2 == 0:
        for col in range(4):
            table[(row_idx, col)].set_facecolor('#f5f5f5')

plt.title('Module 2: Tree-Level Blue Tilt Problem', fontsize=14, fontweight='bold', pad=20)

table_path = os.path.join(FIGURES_DIR, "Module2_Table.png")
plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nTable saved as: {table_path}")
plt.show()

# Scan phi to show ns > 1 everywhere
phis = np.linspace(1e-4, 0.5, 500)
eps_vals = np.array([epsilon(p, Vfun, dVfun) for p in phis])
eta_vals = np.array([eta(p, Vfun, dVfun) for p in phis])
ns_vals = 1 - 6*eps_vals + 2*eta_vals

plt.figure(figsize=(10, 6))
plt.plot(phis, ns_vals, 'b-', linewidth=2.5, label=r'$n_s(\phi)$ tree-level')
plt.axhline(1.0, color='r', linestyle='--', linewidth=2, label='ns = 1')
plt.axhline(0.963, color='g', linestyle=':', linewidth=2, label='WMAP5 central (0.963)')
plt.fill_between([0, 0.5], 0.951, 0.975, alpha=0.15, color='green', label='WMAP5 1σ region')
plt.xlabel(r'$\phi \ / \ M_P$', fontsize=12)
plt.ylabel(r'$n_s$', fontsize=12)
plt.title('Tree-Level Hybrid Inflation: Blue Tilt Problem\n'
          r'$n_s > 1$ for all sub-Planckian $\phi$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.5)
plt.ylim(0.99, 1.01)
plt.tight_layout()

# Save figure
fig_path = os.path.join(FIGURES_DIR, "Module2_Blue_Tilt_Problem.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved as: {fig_path}")
plt.show()

print("\nKEY INSIGHT:")
print("-" * 50)
print("Tree-level hybrid inflation gives ns > 1 (blue tilt).")
print("This is because eta > 0 (the potential is convex).")
print("WMAP5 wants ns < 1 (red tilt).")
print("This is why the paper introduces the loop correction!")