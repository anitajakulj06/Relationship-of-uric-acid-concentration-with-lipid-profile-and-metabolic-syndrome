# -*- coding: utf-8 -*-
"""
Slika 1 / Figure 1: Spearmanov (parcijalni) koeficijent korelacije mokraćne
kiseline s parametrima lipidnog profila prije i nakon prilagodbe, s bootstrap
95 % intervalom pouzdanosti (5000 uzoraka, sjeme 42).

Pokretanje / Usage:
    python slika1.py --podaci tablica_unos_podataka.xlsx [--izlaz slika1.png]
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analiza import ucitaj, korelacija, STUPCI

PLAVA, SIVA = "#2b5ca8", "#9a9a9a"


def glavna(put, izlaz, sheet="Podaci"):
    df = ucitaj(put, sheet)
    U, TG, HDL, LDL = (STUPCI[k] for k in ("urat", "tg", "hdl", "ldl"))
    OSN = [STUPCI["dob"], "spol_m", STUPCI["egfr"]]
    BMI = STUPCI["bmi"]
    grupe = [
        ("Triglycerides", TG, [("Unadjusted", []),
                               ("Adjusted for age, sex, eGFR", OSN),
                               ("Adjusted for BMI", [BMI])]),
        ("HDL-cholesterol", HDL, [("Unadjusted", []),
                                  ("Adjusted for sex", ["spol_m"]),
                                  ("Adjusted for age, sex, eGFR", OSN),
                                  ("Additionally adjusted for TG", OSN + [TG]),
                                  ("Adjusted for age, sex, eGFR, BMI", OSN + [BMI])]),
        ("LDL-cholesterol", LDL, [("Unadjusted", []),
                                  ("Adjusted for age, sex, eGFR", OSN),
                                  ("Adjusted for BMI", [BMI])]),
    ]

    redovi = []   # (oznaka, naslov?, r, lo, hi, p)
    for naslov, y, modeli in grupe:
        redovi.append((naslov, True, None, None, None, None))
        for oznaka, kov in modeli:
            r, p, n, lo, hi = korelacija(df, U, y, kov)
            redovi.append((oznaka, False, r, lo, hi, p))
            print(f"{naslov:16s} {oznaka:34s} rho = {r:.3f}; 95 % CI {lo:.3f} do {hi:.3f}; p = {p:.3g}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.47, right=0.86, top=0.98, bottom=0.17)
    N = len(redovi)
    for i, (oznaka, je_naslov, r, lo, hi, p) in enumerate(redovi):
        yy = N - 1 - i
        if je_naslov:
            ax.text(-0.97, yy, oznaka, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=8.5, fontweight="bold")
            continue
        boja = PLAVA if p < 0.05 else SIVA
        ax.plot([lo, hi], [yy, yy], color=boja, lw=2, solid_capstyle="round")
        ax.plot(r, yy, "o", color=boja, ms=5.5)
        ax.text(-0.92, yy, oznaka, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.5, color="#333333")
        ax.text(1.02, yy, f"ρ = {r:.3f}".replace("-", "−"), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.5, color="#333333")
    ax.axvline(0, color="#555555", ls="--", lw=1)
    ax.set_xlim(-0.5, 0.6)
    ax.set_ylim(-0.7, N - 0.3)
    ax.set_yticks([])
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["−0.4", "−0.2", "0", "0.2", "0.4", "0.6"])
    ax.grid(axis="x", color="#e3e3e3", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Spearman correlation coefficient with uric acid (ρ, 95% CI)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8.5)
    h1, = ax.plot([], [], "o-", color=PLAVA, lw=2, ms=5.5, label="p < 0.05")
    h2, = ax.plot([], [], "o-", color=SIVA, lw=2, ms=5.5, label="not significant")
    fig.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.66, 0.01),
               ncol=2, frameon=False, fontsize=8.5)
    fig.savefig(izlaz, dpi=600)
    print("Spremljeno:", izlaz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--podaci", required=True)
    ap.add_argument("--sheet", default="Podaci")
    ap.add_argument("--izlaz", default="slika1.png")
    a = ap.parse_args()
    glavna(a.podaci, a.izlaz, a.sheet)
