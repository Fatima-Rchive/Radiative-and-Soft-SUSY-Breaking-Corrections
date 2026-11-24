"""
MODULE 1: Tree-Level Hybrid Potential & Critical Field
=======================================================
Define V(phi,S) and compute phi_c where S=0 valley becomes unstable.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.table import Table

# Create module-specific figures folder in the SCRIPT directory
script_dir = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(script_dir, "mod1_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Figures will be saved to: {FIGURES_DIR}")

Mp = 1.0  # reduced Planck mass

def V_full(phi, S, kappa, M, m, g):
    """Full tree-level hybrid inflation potential."""
    return (kappa / 4.0) * (S**2 - M**2)**2 + 0.5 * m**2 * phi**2 + 0.5 * g**2 * phi**2 * S**2

def critical_field(kappa, M, g):
    """phi_c = sqrt(kappa) * M / g where S=0 valley becomes unstable."""
    return np.sqrt(kappa) * M / g

def mS2_along_valley(phi, kappa, M, g):
    """Effective mass-squared of S along S=0 valley."""
    return -kappa * M**2 + g**2 * phi**2

def V_tree(phi, kappa, M, m):
    """Tree-level potential along S=0 valley."""
    return (kappa / 4.0) * M**4 + 0.5 * m**2 * phi**2

print("\nMODULE 1: TREE-LEVEL HYBRID POTENTIAL & CRITICAL FIELD")

# Parameters
kappa, M, m, g = 1.0, 1e-2, 1e-6, 1.0

# Calculate critical field
phi_c = critical_field(kappa, M, g)
print("\nRESULTS:")
print(f"  phi_c = {phi_c:.6e} Mp")
print(f"  m_S^2(phi_c) = {mS2_along_valley(phi_c, kappa, M, g):.3e} (should be ~0)")

# Create table image
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table_data = [
    ['Parameter', 'Symbol', 'Value', 'Description'],
    ['kappa', r'$\kappa$', '1.0', 'Self-coupling of S'],
    ['M', r'$M$', f'{M:.2e} $M_P$', 'Symmetry breaking scale'],
    ['m', r'$m$', f'{m:.2e} $M_P$', 'Inflaton soft mass'],
    ['g', r'$g$', '1.0', 'Hybrid coupling'],
    ['phi_c', r'$\phi_c$', f'{phi_c:.6e} $M_P$', 'Critical field (waterfall trigger)'],
    ['m_S^2(phi_c)', r'$m_S^2(\phi_c)$', f'{mS2_along_valley(phi_c, kappa, M, g):.3e}', 'Should be ~0']
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center', 
                 colWidths=[0.12, 0.15, 0.20, 0.30])

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

plt.title('Module 1: Tree-Level Hybrid Potential Parameters', fontsize=14, fontweight='bold', pad=20)

table_path = os.path.join(FIGURES_DIR, "Module1_Table.png")
plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nTable saved as: {table_path}")
plt.show()

# Plot the tree-level potential
phis = np.linspace(0, 2.5 * phi_c, 500)
V0 = V_tree(phis, kappa, M, m)

plt.figure(figsize=(8, 5))
plt.plot(phis / phi_c, V0 / V0[0], 'b-', linewidth=2.5, label=r'$V_0(\phi)$')
plt.axvline(1.0, color='r', linestyle='--', linewidth=2, label=r'$\phi_c$ (waterfall trigger)')
plt.xlabel(r'$\phi / \phi_c$', fontsize=12)
plt.ylabel(r'$V_0(\phi) / V_0(0)$', fontsize=12)
plt.title('Tree-Level Hybrid Potential Along $S=0$ Valley\n'
          r'$V_0(\phi) = \frac{\kappa}{4}M^4 + \frac{1}{2}m^2\phi^2$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 2.5)
plt.tight_layout()

# Save figure
fig_path = os.path.join(FIGURES_DIR, "Module1_Tree_Level_Potential.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved as: {fig_path}")
plt.show()

print("\nKEY INSIGHT:")
print("-" * 50)
print(f"For phi > phi_c = {phi_c:.2e} Mp, the S=0 valley is STABLE.")
print("For phi < phi_c, the S=0 valley becomes UNSTABLE.")
print("The waterfall field S rolls to its true vacuum at S = M.")
print("This allows single-field inflation along S=0 for phi > phi_c.")