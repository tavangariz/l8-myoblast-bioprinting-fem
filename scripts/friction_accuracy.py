#!/usr/bin/env python3
"""The accuracy of the nozzle friction subtracted in Eq. 17, from the Bagley check (Additional file 1, Section S7).

The entrance pressure is the applied pressure minus the developed nozzle friction computed analytically from the
solved flow (Eq. 17). The Bagley extrapolation on the same finite element model (Additional file 1, Fig. S3C;
nozzle-length sweeps of 2.5, 5 and 10 mm at each nozzle's 200 kPa flow, linear intercept at zero length) does not
use that formula. Its difference from the residual, as a share of the friction, is the error of the computed
friction that the solved pressure field shows. Every ordering of extra_numbers.py and the interior minimum of the
100 µm nozzle are re-evaluated with the friction of each nozzle moved by its own difference. The same check is
repeated at the operating flow of 1 µL/s (results/fem/runs/2026-11-04_bagley-operating) and applied to the
matched-flow comparison of rsc96_transfer.py.

Writes results/analyses/friction_accuracy.json.

    python scripts/friction_accuracy.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from scipy.interpolate import PchipInterpolator

import l8_calibration as C
import l8_supporting_data as SD
import ning2018_data as ning
import rsc96_transfer as RT
import l8_surrogate as S
from l8_damage import A2, B2, D0, K_VIS, LTIP, N_EXP, compute_Dst
from l8_paths import DATA

P_REF = 200.0          # kPa: the applied pressure of the Bagley production solves and of the fixed-pressure series
DIAMS = (100.0, 200.0, 250.0, 330.0)


def dt_pct(d, p_kpa, q_uls, eps):
    """Dt (%) with the developed nozzle friction scaled by (1 + eps), solved flow held fixed (Eq. 17)."""
    r = d * 0.5e-6
    gdot = (3 * N_EXP + 1) / (4 * N_EXP) * 4 * q_uls * 1e-9 / (np.pi * r ** 3)
    dptip = (1 + eps) * 2 * LTIP * K_VIS * gdot ** N_EXP / r
    te = 3 * (N_EXP + 1) / 8 * max(p_kpa * 1e3 - dptip, 0.0)
    return 100 * (1 - (1 - D0) * (1 - float(C.det(te, A2, B2))) * (1 - compute_Dst(dptip, r)))


OP_DIR = DATA / "fem" / "runs" / "2026-11-04_bagley-operating"
Q_OP = 1.0             # uL/s: the operating flow of the matched-flow comparison


def op_bagley(pts1):
    """Bagley intercept at 1 uL/s against the reported entrance pressure there, for every nozzle whose three series exist;
    None if no nozzle is complete. The nozzles left out are named in the result (the 100 um series are the slowest)."""
    files = {(int(d), tag): OP_DIR / f"bagley_op_D{int(d)}_L{tag}.csv" for d, _, _ in pts1 for tag in SD._BAGLEY_L}
    done = [t for t in pts1 if all(files[(int(t[0]), tag)].exists() for tag in SD._BAGLEY_L)]
    pending = [t[0] for t in pts1 if t not in done]
    if pending:
        print(f"operating-flow Bagley: series incomplete for {pending} um ({OP_DIR})")
    if not done:
        return None
    per, raw = {}, {}
    for d, p_op, q in done:
        assert abs(q - Q_OP) < 1e-9
        xy = []
        for tag, L in sorted(SD._BAGLEY_L.items(), key=lambda t: t[1]):
            ser = [r for r in SD._read_bagley_series_at(files[(int(d), tag)]) if r["needle_um"] == d]
            assert len(ser) == 5, (d, tag, len(ser))
            qs = [r["Q_uLs"] for r in ser]
            assert min(qs) < Q_OP < max(qs), (d, tag, qs)   # interpolated, never extrapolated
            xy.append((L, SD._loglog_interp(ser, Q_OP)))
        _, p_bag = SD._linfit([x for x, _ in xy], [y for _, y in xy])
        i_short = SD._linfit([x for x, _ in xy[:2]], [y for _, y in xy[:2]])[1]
        i_long = SD._linfit([x for x, _ in xy[1:]], [y for _, y in xy[1:]])[1]
        rep = SD.derive(p_op, Q_OP, SD.TIP_RADIUS_M[int(d)])["P_en_kPa"]
        fric = p_op - rep
        eps = (rep - p_bag) / fric
        eps_any = max(abs(eps), *(abs(rep - i) / fric for i in (i_short, i_long)))
        raw[d] = {"eps": eps, "eps_any": eps_any}
        per[d] = {"P_op_kPa": round(p_op, 2), "bagley_kPa": round(p_bag, 3), "residual_kPa": round(rep, 3),
                  "friction_kPa": round(fric, 2), "two_tip_intercepts_kPa": [round(i_short, 3), round(i_long, 3)],
                  "diff_pct_of_friction": round(100 * eps, 3), "any_fit_pct_of_friction": round(100 * eps_any, 3)}
        print(f"{d:5.0f} um at 1 uL/s: P {p_op:.1f}  Bagley {p_bag:.4f}  residual {rep:.4f}  friction {fric:.2f} kPa  "
              f"difference {100 * eps:+.4f}%  any fit {100 * eps_any:.4f}% of friction")
    worst = max(raw, key=lambda d: raw[d]["eps_any"])
    return {"per_nozzle": per, "per_nozzle_raw": raw, "nozzles_pending_um": pending,
            "max_abs_diff_pct_of_friction": round(100 * raw[worst]["eps_any"], 3), "max_abs_diff_nozzle_um": worst}


def main():
    rows = S.load_grid()
    bag, eps, eps_any, raw = {}, {}, {}, {}
    for needle, p_bag, _naive, _feed, p_rep in SD.residual_rows():
        d = float(needle)
        fric = P_REF - p_rep
        eps[d] = (p_rep - p_bag) / fric
        raw[d] = (p_rep, fric)
        bag[d] = {"bagley_kPa": round(p_bag, 3), "residual_kPa": round(p_rep, 3), "friction_kPa": round(fric, 1),
                  "diff_pct_of_bagley": round(100 * (p_rep - p_bag) / p_bag, 2),
                  "diff_pct_of_friction": round(100 * eps[d], 3)}
        print(f"{d:5.0f} um  Bagley {p_bag:.4f}  residual {p_rep:.4f}  friction {fric:.2f} kPa  "
              f"difference {100 * eps[d]:+.4f}% of friction")
    # precision of the Bagley intercept itself: the intercepts of the two shorter and of the two longer tips
    tq = SD._target_q()
    for needle in SD.TIPS:
        pts_ = []
        for tag, L in sorted(SD._BAGLEY_L.items(), key=lambda t: t[1]):
            ser = [r for r in SD._read_bagley_series(f"bagley_ning_D{needle}_L{tag}.csv") if r["needle_um"] == needle]
            pts_.append((L, SD._loglog_interp(ser, tq[needle])))
        i_short = SD._linfit([x for x, _ in pts_[:2]], [y for _, y in pts_[:2]])[1]
        i_long = SD._linfit([x for x, _ in pts_[1:]], [y for _, y in pts_[1:]])[1]
        bag[float(needle)]["two_tip_intercepts_kPa"] = [round(i_short, 3), round(i_long, 3)]
        rep_, fric_ = raw[float(needle)]
        eps_any[float(needle)] = max(abs(eps[float(needle)]), *(abs(rep_ - i) / fric_ for i in (i_short, i_long)))
        print(f"{needle:5d} um  two-tip intercepts {i_short:.4f} and {i_long:.4f} kPa")
    max_eps3 = max(abs(v) for v in eps.values())
    # the bound used below covers the three-tip intercept and both two-tip intercepts
    max_eps = max(eps_any.values())
    print(f"largest difference, three-tip fit {100 * max_eps3:.3f}%, any of the three fits {100 * max_eps:.3f}% of friction")

    sets = {}
    for qt in (1.0, 17.94):
        lst = []
        for d in DIAMS:
            s_ = sorted([r for r in rows if r["needle_um"] == d], key=lambda r: r["Q_uLs"])
            lq = np.log10([r["Q_uLs"] for r in s_]); lp = np.log10([r["pressure_kPa"] for r in s_])
            if lq[0] <= np.log10(qt) <= lq[-1]:
                lst.append((d, 10 ** float(PchipInterpolator(lq, lp)(np.log10(qt))), qt))
        sets[f"matched_{qt}"] = (lst, True)
    fixed = []
    for d in DIAMS:
        s_ = sorted([r for r in rows if r["needle_um"] == d], key=lambda r: r["pressure_kPa"])
        lp = np.log10([r["pressure_kPa"] for r in s_]); lq = np.log10([r["Q_uLs"] for r in s_])
        fixed.append((d, P_REF, 10 ** float(PchipInterpolator(lp, lq)(np.log10(P_REF)))))
    sets["fixed_200kPa"] = (fixed, False)

    ext = json.loads((DATA / "analyses" / "extra_numbers.json").read_text())
    ref = {"matched_1.0": ext["friction_1pct_matched_flow_Dt_pct"]["1.0"],
           "matched_17.94": ext["friction_1pct_matched_flow_Dt_pct"]["17.94"],
           "fixed_200kPa": ext["friction_1pct_fixed_200kPa_Dt_pct"]}
    orderings, kept = {}, 0
    for name, (pts, decreasing) in sets.items():
        base = [dt_pct(d, p, q, 0.0) for d, p, q in pts]
        corr = [dt_pct(d, p, q, eps[d]) for d, p, q in pts]
        for (d, _, _), b in zip(pts, base):          # the unperturbed point reproduces extra_numbers
            assert abs(b - ref[name][str(d)]["base"]) < 0.01, (name, d, b)
        ok = all((a > b) if decreasing else (a < b) for a, b in zip(corr, corr[1:]))
        kept += ok
        orderings[name] = {"diam_um": [p[0] for p in pts], "base": [round(v, 2) for v in base],
                           "bagley": [round(v, 2) for v in corr], "ordering_holds": ok}
        print(f"{name}: base {[round(v, 2) for v in base]}  with Bagley friction {[round(v, 2) for v in corr]}  "
              f"ordering {'kept' if ok else 'LOST'}")
    print(f"orderings kept with each nozzle's Bagley friction: {kept} of {len(sets)}")
    # worst case: each nozzle's friction moved independently by the largest difference of any of the three fits,
    # in either direction
    # (Dt is monotone in the friction of its own nozzle, so the corners of the box suffice)
    worst_kept = 0
    for name, (pts, decreasing) in sets.items():
        ok = all(all((a > b) if decreasing else (a < b) for a, b in zip(v, v[1:]))
                 for signs in itertools.product((-max_eps, max_eps), repeat=len(pts))
                 for v in [[dt_pct(d, p, q, e) for (d, p, q), e in zip(pts, signs)]])
        worst_kept += ok
        orderings[name]["holds_any_signs_at_max"] = ok
    print(f"orderings kept with every nozzle moved independently by the largest difference: {worst_kept} of {len(sets)}")

    g = sorted([r for r in rows if r["needle_um"] == DIAMS[0]], key=lambda r: r["pressure_kPa"])
    curve = [dt_pct(DIAMS[0], r["pressure_kPa"], r["Q_uLs"], eps[DIAMS[0]]) for r in g]
    k = int(np.argmin(curve))
    minimum = {"P_kPa": g[k]["pressure_kPa"], "interior": 0 < k < len(g) - 1}
    print(f"100 um with its Bagley friction: minimum at {minimum['P_kPa']} kPa, interior {minimum['interior']}")
    for e in (-max_eps, max_eps):            # the same, with the friction moved by the largest difference of any fit
        c_ = [dt_pct(DIAMS[0], r["pressure_kPa"], r["Q_uLs"], e) for r in g]
        k_ = int(np.argmin(c_))
        minimum[f"at_{'minus' if e < 0 else 'plus'}_max"] = {"P_kPa": g[k_]["pressure_kPa"], "interior": 0 < k_ < len(g) - 1}
        print(f"100 um with friction {100 * e:+.3f}%: minimum at {g[k_]['pressure_kPa']} kPa, interior {0 < k_ < len(g) - 1}")

    # RSC96 (rsc96_transfer): the matched-flow decrease at Ning's pressures. The Bagley sweep was solved for the
    # L8 suspension only, so its differences are applied here as an estimate for the RSC96 rheology.
    tr = json.loads((DATA / "analyses" / "rsc96_transfer.json").read_text())
    po, fr1 = tr["variants"]["O"], tr["friction_1pct_at_ning_matched_pressures_O"]
    rsc_rows = RT.load("RSC96")
    pts_r = []
    for d, (p_n, _meas) in sorted(ning.MATCHED_FLOW_RSC96_MEASURED.items()):
        s_ = sorted([r for r in rsc_rows if r["D"] == float(d)], key=lambda r: r["P"])
        lp = np.log10([r["P"] for r in s_])
        pts_r.append((float(d), p_n, float(10 ** PchipInterpolator(lp, np.log10([r["Q"] for r in s_]))(np.log10(p_n)))))

    def dt_r(d, p, q, e):
        ch = C.chain(p, q, d * 0.5e-6, "RSC96", fric_scale=1 + e)
        return 100 * float(C.total(ch["tau_e_Pa"], ch["Dst"], po["a2"], po["b2"], po["D0"]))

    base_r = [dt_r(d, p, q, 0.0) for d, p, q in pts_r]
    for (d, _, _), b in zip(pts_r, base_r):          # the unperturbed point reproduces rsc96_transfer
        assert abs(b - fr1[str(int(d))]["base"]) < 0.01, ("rsc96", d, b)
    bag_r = [dt_r(d, p, q, eps[d]) for d, p, q in pts_r]
    swaps, worst = set(), {}
    for signs in itertools.product((-max_eps, max_eps), repeat=len(pts_r)):
        v = [dt_r(d, p, q, e) for (d, p, q), e in zip(pts_r, signs)]
        for i in range(len(v) - 1):
            if not v[i] > v[i + 1]:
                pair = f"{pts_r[i][0]:.0f}-{pts_r[i + 1][0]:.0f}"
                swaps.add(pair)
                if v[i + 1] - v[i] > worst.get(pair, [0, 0, -1])[2]:
                    worst[pair] = [round(v[i], 2), round(v[i + 1], 2), v[i + 1] - v[i]]
    rsc = {"diam_um": [p[0] for p in pts_r], "base": [round(v, 2) for v in base_r],
           "bagley": [round(v, 2) for v in bag_r],
           "decrease_holds_bagley": all(a > b for a, b in zip(bag_r, bag_r[1:])),
           "worst_case_swapped_pairs": sorted(swaps),
           "worst_case_swapped_values": {k: v[:2] for k, v in worst.items()}}
    print(f"RSC96 base {rsc['base']}  with the L8 Bagley friction {rsc['bagley']}  decrease "
          f"{'kept' if rsc['decrease_holds_bagley'] else 'LOST'}; worst case swaps {rsc['worst_case_swapped_pairs']} "
          f"{rsc['worst_case_swapped_values']}")

    # The same check at the operating flow of 1 uL/s (run 2026-11-04_bagley-operating, HTCondor 2083565): tip-length
    # sweeps at five pressures around each nozzle's 1 uL/s operating pressure, the same M16 mesh and construction as at
    # 200 kPa. The operating pressures are far below 200 kPa for the wider nozzles, where the friction is a larger share.
    op = op_bagley(sets["matched_1.0"][0])
    if op is not None:
        pts1 = sets["matched_1.0"][0]
        e_op = {d: v["eps"] for d, v in op["per_nozzle_raw"].items()}
        corr1 = [dt_pct(d, p, q, e_op.get(d, 0.0)) for d, p, q in pts1]   # a pending nozzle keeps the computed friction
        op["ordering_1uLs_with_op_bagley"] = {"bagley": [round(v, 2) for v in corr1],
                                              "ordering_holds": all(a > b for a, b in zip(corr1, corr1[1:]))}
        m = op["max_abs_diff_pct_of_friction"] / 100
        op["ordering_1uLs_any_signs_at_op_max"] = all(
            all(a > b for a, b in zip(v, v[1:]))
            for signs in itertools.product((-m, m), repeat=len(e_op))
            for v in [[dt_pct(d, p, q, dict(zip(e_op, signs)).get(d, 0.0)) for d, p, q in pts1]])
        print(f"1 uL/s with each nozzle's operating Bagley friction {op['ordering_1uLs_with_op_bagley']['bagley']} "
              f"ordering {'kept' if op['ordering_1uLs_with_op_bagley']['ordering_holds'] else 'LOST'}; "
              f"any signs at {100 * m:.3f}%: {'kept' if op['ordering_1uLs_any_signs_at_op_max'] else 'LOST'}")
        del op["per_nozzle_raw"]

    lost = ext["friction_ordering_lost_at"]
    thr = min(v["plus_pct"] for v in lost.values())
    out = {"per_nozzle": bag, "max_abs_diff_pct_of_friction_three_tip": round(100 * max_eps3, 3),
           "max_abs_diff_pct_of_friction": round(100 * max_eps, 3),
           "max_abs_diff_nozzle_um": max(eps_any, key=lambda d: eps_any[d]),
           "one_pct_over_max": round(0.01 / max_eps, 1), "smallest_loss_threshold_pct": thr,
           "smallest_threshold_over_max": round(thr / (100 * max_eps), 1),
           "orderings_with_bagley_friction": orderings, "orderings_kept": kept, "orderings_kept_worst_case": worst_kept,
           "minimum_100um_with_bagley_friction": minimum, "rsc96_matched_flow_l8_bagley_applied": rsc,
           "operating_flow_1uLs": op}
    if op is not None:
        out["operating_max_over_200kPa_max"] = round(op["max_abs_diff_pct_of_friction"] / (100 * max_eps), 1)
        out["smallest_threshold_over_operating_max"] = round(thr / op["max_abs_diff_pct_of_friction"], 1)
    print(f"largest difference, any fit, {100 * max_eps:.3f}% of friction; 1% is {out['one_pct_over_max']} times it; "
          f"the smallest loss threshold, {thr}%, is {out['smallest_threshold_over_max']} times it")
    (DATA / "analyses" / "friction_accuracy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
