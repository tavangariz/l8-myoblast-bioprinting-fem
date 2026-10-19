#!/usr/bin/env python3
"""Check of the transcription of Ning et al. (2018), Table S1.

Re-extracts the table from their Supporting Information PDF and compares it with the values typed
into validate_damage_law_vs_ning_TableS1.py. The PDF is not distributed here: set NING_SI_PDF to a
local copy.

    NING_SI_PDF=/path/to/ab8b00714_si_001.pdf python scripts/validation/verify_ning_tableS1_transcription.py
"""
import os
import re

from pypdf import PdfReader

# The Ning et al. (2018) Supporting Information PDF is the publisher's copyright and is
# deliberately NOT redistributed here. NING_SI_PDF points at a locally downloaded copy:
#   NING_SI_PDF=/path/to/ab8b00714_si_001.pdf python3 <this script>
SI = os.environ.get("NING_SI_PDF", "")
if not SI or not os.path.exists(SI):
    raise SystemExit(
        "Set NING_SI_PDF to your local copy of the Ning et al. 2018 Supporting "
        "Information PDF (doi 10.1021/acsbiomaterials.8b00714). It is not "
        "redistributed with this repository."
    )

# The values hardcoded in the validation script (rows = shear time s).
TYPED = {
    10:  [2.11, 3.22, 5.42, 6.18, 8.80],
    30:  [4.73, 5.13, 8.60, 8.29, 12.25],
    60:  [6.30, 8.62, 11.64, 12.08, 18.05],
    120: [9.06, 13.23, 15.44, 17.26, 25.49],
}
TAU = [100, 400, 600, 800, 1000]

reader = PdfReader(SI)
print(f"SI pages: {len(reader.pages)}\n")

pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]

# Locate the table: the page mentioning both "Table S1" and L8.
hits = [(pg, t) for pg, t in pages if re.search(r"Table\s*S1", t, re.I)]
if not hits:
    hits = [(pg, t) for pg, t in pages if "L8" in t]

for pg, t in hits:
    print("=" * 78)
    print(f"PAGE {pg} — raw text as extracted")
    print("=" * 78)
    print(t)
    print()

print("=" * 78)
print("ALL NUMBERS FOUND ON THOSE PAGES, IN ORDER")
print("=" * 78)
for pg, t in hits:
    nums = re.findall(r"\d+\.\d+|\b\d+\b", t)
    print(f"[p{pg}] {nums}")

print("\n" + "=" * 78)
print("CURRENTLY HARDCODED (for eyeball diff against the page dump above)")
print("=" * 78)
print(f"{'t(s)':>5} | " + " | ".join(f"{x:>6} Pa" for x in TAU))
for t_s, row in TYPED.items():
    print(f"{t_s:5d} | " + " | ".join(f"{v:9.2f}" for v in row))
print("\nEvery value above must appear in the page dump. Any that does not is a "
      "transcription error and invalidates the R^2 until corrected.")
