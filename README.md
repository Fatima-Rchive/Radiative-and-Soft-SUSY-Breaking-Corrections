# Radiative and Soft SUSY-Breaking Corrections

![status](https://img.shields.io/badge/status-work%20in%20progress-orange)
![stage](https://img.shields.io/badge/stage-non--SUSY%20reproduction-blue)

> 🚧 **Work in progress.** This repository is under active development. Code, results and structure will change, and nothing here has been finalised or independently verified yet.

## Overview

This project studies radiative and soft SUSY-breaking corrections to hybrid inflation. The first step is a from-scratch numerical reproduction of a non-supersymmetric baseline:

> Rehman, Shafi & Wickman (2009), *Hybrid Inflation Revisited in Light of WMAP5*, Phys. Rev. D **79**, 103503, [arXiv:0901.4345](https://arxiv.org/abs/0901.4345)

The core physics: a tree-level hybrid inflation potential plus a one-loop Coleman–Weinberg correction from Yukawa couplings between the inflaton and right-handed neutrinos. This induces a hilltop and allows a red-tilted spectral index at sub-Planckian field values, which the tree-level potential cannot give. Units are reduced Planck mass M_P = 1 unless stated.

## Status

| Stage | Description | Status |
|---|---|---|
| 1 | Slow-roll formalism check on a toy potential | ✅ done |
| 2 | Tree-level hybrid potential and critical field | ✅ done |
| 3 | Tree-level blue-tilt problem | ✅ done |
| 4 | One-loop radiative correction (Yukawa-coupled RH neutrinos) | ✅ done |
| 5 | Full numerical slow-roll observables (n_s, r, N) | 🔄 in progress / being validated |
| 6 | Reheating (and Table 1 from scratch) | 🔄 in progress |
| 7 | Reheating into RH neutrinos and leptogenesis | ⏳ not started |
| 8 | Comparison with Planck 2018 / ACT | ⏳ not started |
| 9 | SUSY / soft SUSY-breaking extension | ⏳ not started |

*(Please adjust this table to match your actual progress.)*

## Repository layout

```
notebooks/
  wmap5_nonsusy_hilltop_framework.ipynb   # main notebook (current version)
  figures/                                # figures written by the notebook
  archive/
    wmap5_prereq_version_older.ipynb      # earlier version, kept for reference
preliminary_modules/
  module0 … module5 *.py                  # standalone preliminary scripts
  mod0_figures … mod5_figures/            # their output figures
```

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/wmap5_nonsusy_hilltop_framework.ipynb   # run from the notebooks/ folder
python preliminary_modules/module5_parameter_scan_ns_r.py
```

The notebook writes figures to `./figures/` relative to where it is run, so launch it from inside `notebooks/`.

## Known caveats

- Results are a work in progress and may not yet match every number in the paper.
- The older notebook in `notebooks/archive/` is superseded and may contain errors fixed later.
