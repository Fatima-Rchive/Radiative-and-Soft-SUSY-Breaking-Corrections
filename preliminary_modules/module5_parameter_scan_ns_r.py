"""
Scan over (y, M) and compute ns, r at each point.
Reproduces the paper's central ns-r figure.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os
from matplotlib.table import Table

# Create module-specific figures folder
script_dir = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(script_dir, "mod5_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Figures will be saved to: {FIGURES_DIR}")

Mp = 1.0

def V_tree(phi, kappa, M, m):
    return (kappa / 4.0) * M**4 + 0.5 * m**2 * phi**2

def dV_tree(phi, m):
    return m**2 * phi

def mN(phi, y):
    return y * phi

def V_loop(phi, y, Q, Ndof=3.0):
    M = mN(phi, y)
    M = np.where(np.abs(M) < 1e-300, 1e-300, M)
    return -(Ndof / (64.0 * np.pi**2)) * M**4 * (np.log(M**2 / Q**2) - 25/6)

def dV_loop(phi, y, Q, Ndof=3.0):
    u = mN(phi, y)
    u = np.where(np.abs(u) < 1e-300, 1e-300, u)
    return -(Ndof / (64.0 * np.pi**2)) * y * 4.0 * u**3 * (np.log(u**2 / Q**2) - 25/6 + 0.5)

def d2V_loop(phi, y, Q, Ndof=3.0):
    u = mN(phi, y)
    u = np.where(np.abs(u) < 1e-300, 1e-300, u)
    return -(Ndof / (64.0 * np.pi**2)) * y**2 * 4.0 * u**2 * (3.0 * np.log(u**2 / Q**2) - 25/6 + 1.5)

def V_total(phi, kappa, M, m, y, Q, Ndof=3.0):
    return V_tree(phi, kappa, M, m) + V_loop(phi, y, Q, Ndof)

def critical_field(kappa, M, g):
    return np.sqrt(kappa) * M / g

def make_V_and_dV(kappa, M, m, y, Q, Ndof=3.0):
    def Vfun(phi):
        return V_total(phi, kappa, M, m, y, Q, Ndof)
    def dVfun(phi):
        return dV_tree(phi, m) + dV_loop(phi, y, Q, Ndof)
    def d2Vfun(phi):
        return m**2 + d2V_loop(phi, y, Q, Ndof)
    return Vfun, dVfun, d2Vfun

def find_hilltop(Vfun, dVfun, phi_lo, phi_hi, n=10000):
    phis = np.geomspace(phi_lo, phi_hi, n)
    dVs = np.array([dVfun(p) for p in phis])
    sign_changes = np.where(np.diff(np.sign(dVs)) < 0)[0]
    if len(sign_changes) == 0:
        return None
    idx = sign_changes[-1]
    lo, hi_ = phis[idx], phis[idx + 1]
    return brentq(dVfun, lo, hi_)

def epsilon(phi, Vfun, dVfun):
    return 0.5 * Mp**2 * (dVfun(phi) / Vfun(phi))**2

def eta_param(phi, Vfun, dVfun, d2Vfun):
    return Mp**2 * d2Vfun(phi) / Vfun(phi)

def N_efolds(phi, phi_end, Vfun, dVfun):
    integrand = lambda p: Vfun(p) / (Mp**2 * dVfun(p))
    val, _ = quad(integrand, phi_end, phi, limit=400, epsabs=1e-10, epsrel=1e-8)
    return val

def ns_from_eps_eta(eps, eta_val):
    return 1.0 - 6.0 * eps + 2.0 * eta_val

def r_from_eps(eps):
    return 16.0 * eps

def compute_observables(kappa, M, m, y, Q, g, N_target, verbose=False):
    try:
        Vfun, dVfun, d2Vfun = make_V_and_dV(kappa, M, m, y, Q)
        phi_max = find_hilltop(Vfun, dVfun, 1e-4, 0.9)
        if phi_max is None:
            return None
        phi_c = critical_field(kappa, M, g)
        phi_end = phi_c
        
        def f_phi(p):
            return N_efolds(p, phi_end, Vfun, dVfun) - N_target
        
        phi_star = brentq(f_phi, phi_end * 1.0001, phi_max * 0.9999)
        eps = epsilon(phi_star, Vfun, dVfun)
        eta_val = eta_param(phi_star, Vfun, dVfun, d2Vfun)
        ns = ns_from_eps_eta(eps, eta_val)
        r = r_from_eps(eps)
        
        if verbose:
            print(f"M={M:.3e}, y={y:.3e}: ns={ns:.4f}, r={r:.3e}")
        
        return dict(ns=ns, r=r, phi_star=phi_star, eps=eps, eta=eta_val)
    except Exception as e:
        return None

print("\nMODULE 5: PARAMETER SPACE SCAN (ns-r Plot)")
print("Scanning over (y, M) to reproduce the paper's ns-r figure")

kappa, m, g = 1.0, 1e-8, 1.0
N_target = 50

Ms = np.geomspace(3e-4, 3e-3, 8)
ys = np.geomspace(3e-3, 2e-2, 8)

results = []
print("\nScanning parameter space...")
for M in Ms:
    for y in ys:
        Q = M
        result = compute_observables(kappa, M, m, y, Q, g, N_target, verbose=False)
        if result is not None and 0.8 < result['ns'] < 1.1 and result['r'] < 1:
            results.append((y, M, result['ns'], result['r'], result['phi_star'], 
                          result['eps'], result['eta']))

results = np.array(results)
print(f"Found {len(results)} valid points")

if len(results) > 0:
    ns_vals = results[:, 2]
    r_vals = results[:, 3]
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')
    
    idx_best = np.argmin(np.abs(ns_vals - 0.963))
    
    table_data = [
        ['Metric', 'Value', 'Description'],
        ['Total valid points', f'{len(results)}', 'Parameter combinations that work'],
        ['ns range', f'{ns_vals.min():.4f} - {ns_vals.max():.4f}', 'Range of spectral index'],
        ['r range', f'{r_vals.min():.3e} - {r_vals.max():.3e}', 'Range of tensor-to-scalar ratio'],
        ['Best ns', f'{ns_vals[idx_best]:.4f}', f'Closest to WMAP5 (0.963)'],
        ['Best r', f'{r_vals[idx_best]:.3e}', f'At best ns point'],
        ['Best y', f'{results[idx_best, 0]:.3e}', 'Yukawa coupling at best point'],
        ['Best M', f'{results[idx_best, 1]:.3e}', 'Mass scale at best point'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', 
                     colWidths=[0.20, 0.25, 0.35])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    if ns_vals[idx_best] < 1:
        for i in range(3):
            table[(4, i)].set_facecolor('#90ee90')
            table[(4, i)].set_text_props(color='green', fontweight='bold')
    
    for row_idx in range(1, 8):
        if row_idx % 2 == 0 and row_idx != 4:
            for col in range(3):
                if col != 2:
                    table[(row_idx, col)].set_facecolor('#f5f5f5')
    
    plt.title(f'Module 5: Parameter Scan Summary (N={N_target})', 
              fontsize=14, fontweight='bold', pad=20)
    
    table_path = os.path.join(FIGURES_DIR, "Module5_Table.png")
    plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nTable saved as: {table_path}")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    sorted_idx = np.argsort(np.abs(ns_vals - 0.963))
    top_n = min(10, len(sorted_idx))
    
    table_data = [
        ['#', 'y', 'M ($M_P$)', 'n_s', 'r', 'φ* ($M_P$)', 'ε', 'η']
    ]
    for i, idx in enumerate(sorted_idx[:top_n]):
        y, M, ns, r, phi_star, eps, eta_val = results[idx]
        table_data.append([
            f'{i+1}',
            f'{y:.3e}',
            f'{M:.3e}',
            f'{ns:.4f}',
            f'{r:.3e}',
            f'{phi_star:.4f}',
            f'{eps:.3e}',
            f'{eta_val:.3e}'
        ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', 
                     colWidths=[0.05, 0.12, 0.12, 0.10, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    
    for i in range(8):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for row_idx in range(1, top_n + 1):
        ns_val = float(table_data[row_idx][3])
        if ns_val < 1:
            for col in range(8):
                table[(row_idx, col)].set_facecolor('#90ee90')
        elif row_idx % 2 == 0:
            for col in range(8):
                table[(row_idx, col)].set_facecolor('#f5f5f5')
    
    plt.title('Module 5: Top 10 Points Closest to WMAP5', 
              fontsize=14, fontweight='bold', pad=20)
    
    table_path = os.path.join(FIGURES_DIR, "Module5_Top10_Table.png")
    plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Top 10 table saved as: {table_path}")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(results[:, 2], results[:, 3], 
                    c=np.log10(results[:, 0]), cmap='viridis', s=80, alpha=0.7)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\log_{10}(y)$ [Yukawa coupling]', fontsize=12)
    
    ax.plot(ns_vals[idx_best], r_vals[idx_best], 'r*', markersize=15, 
            label=f'Best: ns={ns_vals[idx_best]:.4f}')
    
    ax.axvline(0.963, color='k', linestyle='--', alpha=0.5, label='WMAP5 central')
    ax.axvspan(0.951, 0.975, alpha=0.15, color='green', label='WMAP5 1σ')
    
    ax.set_xlabel(r'$n_s$', fontsize=12)
    ax.set_ylabel(r'$r$', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'ns-r Parameter Scan at N={N_target}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.98, 1.10)
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, "Module5_ns_r_Scan.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved as: {fig_path}")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, 4))
    
    M_fixed = results[-1, 1]
    
    for y_val, c in zip([6e-3, 8e-3, 1e-2, 1.3e-2], colors):
        Q = M_fixed
        Vfun, dVfun, d2Vfun = make_V_and_dV(kappa, M_fixed, m, y_val, Q)
        phi_max = find_hilltop(Vfun, dVfun, 1e-4, 0.9)
        phi_c = critical_field(kappa, M_fixed, g)
        if phi_max is None or not (phi_c < phi_max):
            continue
        phis = np.geomspace(phi_c * 1.0005, phi_max * 0.99999, 250)
        nss = []
        for p in phis:
            eps_val = epsilon(p, Vfun, dVfun)
            eta_val = eta_param(p, Vfun, dVfun, d2Vfun)
            nss.append(ns_from_eps_eta(eps_val, eta_val))
        ax.plot(phis, nss, color=c, lw=2.0, label=rf"$y={y_val:.1e}$")
    
    ax.axhline(1.0, color='gray', ls='--', lw=1, label='ns = 1')
    ax.axhline(0.963, color='r', ls=':', lw=1, label='WMAP5 central')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\phi_*\ / \ M_P$  (sub-Planckian)', fontsize=12)
    ax.set_ylabel(r'$n_s$', fontsize=12)
    ax.set_title('Sub-Planckian Hilltop: ns(phi*) for several Yukawa couplings', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, "Module5_ns_vs_phistar.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved as: {fig_path}")
    plt.close(fig)
    
else:
    print("No valid points found. Try adjusting parameter ranges.")