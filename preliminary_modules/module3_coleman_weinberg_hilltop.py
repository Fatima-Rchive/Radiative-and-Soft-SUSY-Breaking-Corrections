"""
Add the one-loop correction from Yukawa-coupled RH neutrinos.
This creates the "hilltop" feature in the potential.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.table import Table

# Create module-specific figures folder
script_dir = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(script_dir, "mod3_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Figures will be saved to: {FIGURES_DIR}")

Mp = 1.0

def V_tree(phi, kappa, M, m):
    return (kappa / 4.0) * M**4 + 0.5 * m**2 * phi**2

def mN(phi, y):
    """Field-dependent RH neutrino mass."""
    return y * phi

def V_loop(phi, y, Q, Ndof=3.0):
    """One-loop Coleman-Weinberg correction for Yukawa-coupled fermions."""
    M = mN(phi, y)
    M = np.where(np.abs(M) < 1e-300, 1e-300, M)
    return -(Ndof / (64.0 * np.pi**2)) * M**4 * (np.log(M**2 / Q**2) - 1.5)

def V_total(phi, kappa, M, m, y, Q, Ndof=3.0):
    return V_tree(phi, kappa, M, m) + V_loop(phi, y, Q, Ndof)

print("\nMODULE 3: COLEMAN-WEINBERG CORRECTION & HILLTOP FORMATION")
print("Adding loop correction from Yukawa-coupled RH neutrinos")

# Parameters
kappa, M, m, y, Q = 1.0, 1e-3, 1e-8, 1e-2, 1e-3

# Find the hilltop numerically
def dV_total(phi, kappa, M, m, y, Q):
    """Numerical derivative of V_total."""
    h = 1e-8
    return (V_total(phi + h, kappa, M, m, y, Q) - V_total(phi - h, kappa, M, m, y, Q)) / (2*h)

# Find where dV/dphi = 0
phi_scan = np.linspace(1e-4, 0.2, 500)
dV_vals = np.array([dV_total(p, kappa, M, m, y, Q) for p in phi_scan])
idx_max = np.argmin(np.abs(dV_vals))
phi_max = phi_scan[idx_max]

# Create table image
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table_data = [
    ['Parameter', 'Symbol', 'Value', 'Description'],
    ['kappa', r'$\kappa$', '1.0', 'Self-coupling of S'],
    ['M', r'$M$', f'{M:.2e} $M_P$', 'Symmetry breaking scale'],
    ['m', r'$m$', f'{m:.2e} $M_P$', 'Inflaton soft mass'],
    ['y', r'$y$', f'{y:.2e}', 'Yukawa coupling (RH neutrinos)'],
    ['Q', r'$Q$', f'{Q:.2e} $M_P$', 'Renormalization scale'],
    [r'$\phi_{max}$', r'$\phi_{max}$', f'{phi_max:.4f} $M_P$', 'Hilltop maximum position']
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center', 
                 colWidths=[0.12, 0.15, 0.20, 0.35])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for row_idx in range(1, 7):
    if row_idx % 2 == 0:
        for col in range(4):
            table[(row_idx, col)].set_facecolor('#f5f5f5')

plt.title('Module 3: Coleman-Weinberg Correction Parameters', fontsize=14, fontweight='bold', pad=20)

table_path = os.path.join(FIGURES_DIR, "Module3_Table.png")
plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nTable saved as: {table_path}")
plt.show()

print("\nPARAMETERS:")
print(f"  kappa = {kappa}")
print(f"  M = {M:.2e} Mp")
print(f"  m = {m:.2e} Mp")
print(f"  y = {y:.2e} (Yukawa coupling)")
print(f"  Q = {Q:.2e} Mp (renormalization scale)")
print(f"  phi_max = {phi_max:.4f} Mp (hilltop maximum)")

# Scan phi to see hilltop
phis = np.linspace(1e-5, 0.3, 1000)
Vtree = V_tree(phis, kappa, M, m)
Vtot = V_total(phis, kappa, M, m, y, Q)

plt.figure(figsize=(10, 6))
plt.plot(phis, Vtree / Vtree[0], 'b--', linewidth=2, label='Tree level (monotonic)')
plt.plot(phis, Vtot / Vtree[0], 'r-', linewidth=2.5, label='Tree + 1-loop (hilltop)')
plt.axvline(phi_max, color='k', linestyle=':', linewidth=1.5, label=f'Hilltop at φ_max = {phi_max:.4f} M_P')
plt.xlabel(r'$\phi \ / \ M_P$', fontsize=12)
plt.ylabel(r'$V(\phi) / V(0)$', fontsize=12)
plt.title('One-Loop Correction Creates a Hilltop Maximum\n'
          r'$V_{\rm total}(\phi) = V_{\rm tree}(\phi) + V_{\rm loop}(\phi)$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
fig_path = os.path.join(FIGURES_DIR, "Module3_Hilltop_Formation.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved as: {fig_path}")
plt.show()

print("\nKEY INSIGHT:")
print("-" * 50)
print("The loop correction creates a LOCAL MAXIMUM (hilltop).")
print("This allows inflation near the hilltop with phi < Mp.")
print(f"At the hilltop, eta becomes NEGATIVE: eta = V''/V < 0.")
print("This pulls ns down via ns = 1 - 6epsilon + 2eta.")
print("Result: ns < 1 (red tilt) for sub-Planckian phi!")