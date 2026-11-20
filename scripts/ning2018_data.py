#!/usr/bin/env python3
"""Published measurements of Ning et al. (2018) that this study compares against, defined once.

Ning L, Betancourt N, Schreyer DJ, Chen X. Characterization of cell damage and proliferative ability
during and after bioprinting. ACS Biomater Sci Eng. 2018;4:3906-18. doi:10.1021/acsbiomaterials.8b00714

Every entry says where its number comes from:
  "text"       printed in the paper (section or table named), copied exactly;
  "digitised"  read off a figure by scripts/digitise_ning_figures.py (marker centre by template
               matching, axes from the tick marks); the 200 µm points of Fig. 7, which the text also
               prints, reproduce to within 0.03 percentage points, which bounds the reading error.

Nothing in this module is fitted. Damage is the percentage of damaged cells (stained by Annexin V-FITC
only, injured, or by both Annexin V-FITC and propidium iodide, dead) counted immediately after printing.
"""
from __future__ import annotations

# Rheology (their Section 4.2) and damage-law coefficients (shear: their Section 4.1, Eqs. 15-16;
# extensional: their Section 4.5, Eqs. 18-19), both cell lines.
RHEOLOGY = {"L8": dict(K=25.60, n=0.35), "RSC96": dict(K=20.81, n=0.37)}
DENSITY_KG_M3 = 990.0                      # "measured as 0.99 g/mL" (their Section 3.5)
LAW = {
    "L8":    dict(a1=3.2e-4, b1=0.61, c1=0.52, a2=1.43e-3, b2=0.55),
    "RSC96": dict(a1=3.9e-4, b1=0.48, c1=0.71, a2=9.43e-4, b2=0.61),
}

# Damage immediately after printing through the 200 um needle (their Section 4.6). The extensional
# branch of each law was fitted to these four points, so they are calibration data, not tests.
DAMAGE_PRESSURE_SERIES_200UM = {
    "L8":    {50: 6.99, 100: 11.70, 200: 15.53, 400: 23.08},
    "RSC96": {50: 7.20, 100: 11.62, 200: 15.57, 400: 25.95},
}

# Damage at a fixed pressure of 200 kPa through 100-330 um needles (Fig. 7, digitised). Ning used
# them to test their law; they enter no fit, theirs or ours, apart from the shared 200 um point.
# Values: (mean, lower end of error bar, upper end of error bar), percent.
DAMAGE_DIAMETER_SERIES_200KPA = {
    "L8":    {100: (10.43, 7.20, 13.75), 200: (15.56, 13.10, 18.04),
              250: (15.81, 10.65, 20.95), 330: (16.59, 11.31, 21.85)},
    "RSC96": {100: (11.38, 7.23, 15.64), 200: (15.55, 9.80, 21.37),
              250: (16.45, 9.74, 23.15), 330: (20.18, 13.79, 25.66)},
}

# Mass flow rate, g/s. Through the 200 um needle at five pressures (Table 1, text, mean and SD):
FLOW_PRESSURE_SERIES_200UM_G_S = {
    "RSC96": {50: (2.0e-4, 0.51e-4), 100: (1.9e-3, 0.34e-3), 200: (2.1e-2, 0.52e-2),
              300: (4.8e-2, 0.56e-2), 400: (1.1e-1, 0.12e-1)},
    "L8":    {50: (2.6e-4, 1.0e-4), 100: (1.8e-3, 0.46e-3), 200: (1.7e-2, 0.58e-2),
              300: (4.1e-2, 0.77e-2), 400: (9.8e-2, 0.95e-2)},
}
# At 200 kPa through 250 and 330 um needles (Fig. 4C-D, digitised means; the 100 um marker sits on
# the axis and cannot be read, and the 200 um value is taken from Table 1).
FLOW_DIAMETER_SERIES_200KPA_G_S = {
    "L8":    {250: 0.0486, 330: 0.2184},
    "RSC96": {250: 0.0532, 330: 0.2217},
}

# Matched flow rate of Ning's RSC96 comparison across needle diameters (their Section 5, Discussion): about
# 3e-3 g/s, reached between 40 and 430 kPa for 100-330 um needles; 16.84 % damage through 100 um and
# 7.28 % through 330 um (their Fig. S4 holds the full series).
MATCHED_FLOW_RSC96_G_S = 3e-3
MATCHED_FLOW_RSC96_DAMAGE_TEXT = {100: 16.84, 330: 7.28}
# The full measured series (Supporting Information, Fig. S4, red squares, digitised; their simulated values
# are not used): {diameter um: (applied pressure kPa, mean damage %)}. The 100 and 330 um means reproduce
# the text values above to within 0.01 percentage points. Two bar ends are hidden under their simulated
# markers, so only the means are taken.
MATCHED_FLOW_RSC96_MEASURED = {100: (430.0, 16.85), 200: (100.0, 11.50), 250: (60.0, 7.44), 330: (40.0, 7.27)}
