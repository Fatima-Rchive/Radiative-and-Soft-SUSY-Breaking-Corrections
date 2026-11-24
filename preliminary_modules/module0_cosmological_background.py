"""
MODULE 0: Cosmological Background & Chaotic Inflation
=====================================================
Derive ns(N) for V(phi) = (1/2)m^2 phi^2 and compare with WMAP5.
This serves as the sanity-check foundation for all subsequent modules.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.table import Table

# Create module-specific figures folder in the SCRIPT directory
script_dir = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(script_dir, "mod0_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Figures will be saved to: {FIGURES_DIR}")

def chaotic_inflation_N(N):
    """Exact ns for V = (1/2)m^2 phi^2: ns = 1 - 4/(2N+1)"""
    return 1 - 4/(2*N + 1)

def check_wmap5_compatibility(ns, ns_lower=0.951, ns_upper=0.975):
    """Check if ns is within WMAP5 1σ range"""
    return ns_lower <= ns <= ns_upper

print("\nMODULE 0: COSMOLOGICAL BACKGROUND & CHAOTIC INFLATION")

# Derivation
print("\nDERIVATION:")
print("-" * 50)
print("For V(phi) = (1/2)m^2 phi^2:")
print("  epsilon(phi) = 2 Mp^2 / phi^2")
print("  eta(phi) = 2 Mp^2 / phi^2")
print("  N(phi) = (phi^2 - 2Mp^2) / (4Mp^2)")
print("  ns(N) = 1 - 4/(2N+1)")
print("-" * 50)

# Generate data
N_values = [40, 45, 50, 55, 60, 65, 70, 80]
data = []
for N in N_values:
    ns = chaotic_inflation_N(N)
    status = "✓" if check_wmap5_compatibility(ns) else "✗"
    data.append((N, ns, status))

# Create table image
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Create table
table_data = []
table_data.append(['N', 'n_s(N)', 'WMAP5 Central', 'WMAP5 1σ Range', 'Status'])
for N, ns, status in data:
    table_data.append([f"{N}", f"{ns:.5f}", "0.963", "0.951 - 0.975", status])

# Create the table
table = ax.table(cellText=table_data, loc='center', cellLoc='center', 
                 colWidths=[0.08, 0.15, 0.15, 0.20, 0.10])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color the header row
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# Color rows based on status
for row_idx, (N, ns, status) in enumerate(data, start=1):
    if status == '✓':
        table[(row_idx, 4)].set_facecolor('#90ee90')  # Light green
        table[(row_idx, 4)].set_text_props(color='green', fontweight='bold')
    else:
        table[(row_idx, 4)].set_facecolor('#ffcccb')  # Light red
        table[(row_idx, 4)].set_text_props(color='red', fontweight='bold')
    
    # Alternate row colors
    if row_idx % 2 == 0:
        for col in range(5):
            if col != 4:  # Skip status column (already colored)
                table[(row_idx, col)].set_facecolor('#f5f5f5')

# Add title
plt.title('Chaotic Inflation: ns(N) vs WMAP5 Data', fontsize=14, fontweight='bold', pad=20)

# Save table image
table_path = os.path.join(FIGURES_DIR, "Module0_Table.png")
plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nTable saved as: {table_path}")
plt.show()

# Create plot
N_vals = np.linspace(40, 80, 200)
ns_vals = chaotic_inflation_N(N_vals)

plt.figure(figsize=(10, 6))
plt.plot(N_vals, ns_vals, 'b-', linewidth=2.5, 
         label=r'$n_s(N) = 1 - 4/(2N+1)$')

# WMAP5 bounds
plt.axhline(0.963, color='r', linestyle='--', label='WMAP5 central', linewidth=1.5)
plt.axhline(0.951, color='r', linestyle=':', alpha=0.7)
plt.axhline(0.975, color='r', linestyle=':', alpha=0.7)
plt.fill_between([40, 80], 0.951, 0.975, alpha=0.15, color='red', 
                 label='WMAP5 1σ region')

# Mark compatible N values
for N, ns, status in data:
    color = 'green' if status == '✓' else 'red'
    plt.plot(N, ns, 'o', color=color, markersize=8)
    plt.annotate(f'{ns:.5f}', 
                xy=(N, ns),
                xytext=(0, 10),
                textcoords='offset points',
                fontsize=8,
                color=color,
                ha='center')

plt.xlabel('Number of e-folds $N$', fontsize=12)
plt.ylabel('Spectral index $n_s$', fontsize=12)
plt.title('Chaotic Inflation: Does it Fit WMAP5?', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xlim(38, 82)
plt.ylim(0.94, 0.985)
plt.tight_layout()

# Save figure
fig_path = os.path.join(FIGURES_DIR, "Module0_Chaotic_Inflation_ns_vs_N.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved as: {fig_path}")
plt.show()

# Large-N approximation check
print("\nLarge-N approximation check:")
print("-" * 50)
for N_test in [50, 55, 60, 100]:
    ns_exact = chaotic_inflation_N(N_test)
    ns_approx = 1 - 2/N_test
    diff = abs(ns_exact - ns_approx)
    print(f"N = {N_test:3d}:")
    print(f"  Exact:      ns = {ns_exact:.8f}")
    print(f"  Approx:     ns = {ns_approx:.8f}")
    print(f"  Difference:    {diff:.2e}")
    print()

print("\nKEY INSIGHT:")
print("-" * 50)
print("WMAP5 prefers ns < 1 (red tilt).")
print("Chaotic inflation gives ns ≈ 0.960-0.967 for N=50-60.")
print("This matches WMAP5 (ns = 0.963 ± 0.012).")
print("But tree-level hybrid inflation gives ns > 1 (blue tilt).")
print("This is the problem the paper exists to fix!")