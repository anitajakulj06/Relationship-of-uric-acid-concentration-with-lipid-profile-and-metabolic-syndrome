# -*- coding: utf-8 -*-
"""
Korelacija koncentracije mokraćne kiseline s lipidnim profilom i metaboličkim
sindromom: retrospektivna presječna studija.
Correlation of uric acid concentration with lipid profile and metabolic
syndrome: a retrospective cross-sectional study.

Statistička analiza / Statistical analysis — Python 3.11+
(pandas, NumPy, SciPy, statsmodels; openpyxl za čitanje .xlsx / to read .xlsx).

Instalacija / Installation:
    pip install -r requirements.txt

Pokretanje / Usage:
    python analiza.py --podaci tablica_unos_podataka.xlsx [--sheet Podaci]
    python analiza.py --podaci primjer_podaci.xlsx      # sintetički primjer / synthetic example

Skripta ispisuje sve brojke koje se pojavljuju u poglavlju Rezultati.
Autorica istraživanja / Author of the study: Anita Jakulj, dr. med.
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

SJEME = 42        # fiksirano sjeme generatora slučajnih brojeva (ponovljivost)
B_BOOT = 5000     # broj uzoraka u postupku samoučitavanja (bootstrap)
ALFA = 0.05

STUPCI = {
    "urat": "Urat\n(μmol/L)", "ldl": "LDL\n(mmol/L)", "hdl": "HDL\n(mmol/L)",
    "tg": "TG\n(mmol/L)", "glukoza": "Glukoza\n(mmol/L)",
    "egfr": "eGFR\n(mL/min/1.73m²)", "dob": "Dob", "spol": "Spol", "bmi": "BMI",
    "struk": "opseg struka", "sist": "TT sistolički\n(mmHg)",
    "dijast": "TT dijastolički\n(mmHg)", "hipolipemik": "Hipolipemik",
    "hipourikemik": "Hiopurikemik", "mets": "MetS",
    "broj_kriterija": "Broj kriterija", "datum": "Datum vađenja krvi",
    "sifra": "Šifra pacijenta",
}

# gornja granica referentnog intervala laboratorija (µmol/L)
HIPERURICEMIJA = {"M": 400, "Z": 340}


# ---------------------------------------------------------------- podaci ---
def ucitaj(put, sheet="Podaci"):
    df = pd.read_excel(put, sheet_name=sheet)
    df = df[df[STUPCI["sifra"]].notna()].copy()
    df["spol_m"] = (df[STUPCI["spol"]].astype(str).str.upper().str[0] == "M").astype(int)
    df["mets_da"] = (df[STUPCI["mets"]].astype(str).str.strip().str.lower() == "da").astype(int)
    for k in ("hipolipemik", "hipourikemik"):
        df[k + "_da"] = (df[STUPCI[k]].astype(str).str.strip().str.lower() == "da").astype(int)
    df["log_tg"] = np.log(df[STUPCI["tg"]])
    df["log_hdl"] = np.log(df[STUPCI["hdl"]])
    return df


def opis(x):
    x = pd.Series(x).dropna()
    return (f"n={len(x)}; M±SD {x.mean():.1f}±{x.std(ddof=1):.1f}; "
            f"medijan {x.median():.2f} (IQR {x.quantile(.25):.2f}–{x.quantile(.75):.2f}); "
            f"raspon {x.min():.2f}–{x.max():.2f}")


# ------------------------------------------------------------ korelacije ---
def spearman(x, y):
    """Spearmanov koeficijent korelacije ranga."""
    A = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    A = A[~np.isnan(A).any(axis=1)]
    rho, p = stats.spearmanr(A[:, 0], A[:, 1])
    return rho, p, len(A)


def parcijalna_spearman(x, y, *kovarijate):
    """
    Parcijalna korelacija na rangovima: sve se varijable pretvore u rangove,
    iz x i y se linearnom regresijom ukloni utjecaj kovarijata, a Pearsonova
    korelacija reziduala jednaka je parcijalnom Spearmanovu rho.
    """
    A = np.column_stack([np.asarray(x, float), np.asarray(y, float)] +
                        [np.asarray(k, float) for k in kovarijate])
    A = A[~np.isnan(A).any(axis=1)]
    R = np.apply_along_axis(stats.rankdata, 0, A)
    K = np.column_stack([np.ones(len(R)), R[:, 2:]])
    coef, *_ = np.linalg.lstsq(K, R[:, :2], rcond=None)
    E = R[:, :2] - K @ coef
    r = float(np.corrcoef(E[:, 0], E[:, 1])[0, 1])
    n, k = len(A), len(kovarijate)
    dfree = n - k - 2
    t = r * np.sqrt(dfree / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), dfree)), n


def korelacija(df, x, y, kovarijate=(), b=B_BOOT, sjeme=SJEME):
    """
    (Parcijalni) Spearmanov koeficijent s 95 % intervalom pouzdanosti
    dobivenim postupkom samoučitavanja (percentilna metoda).
    """
    A = np.column_stack([df[x].values.astype(float), df[y].values.astype(float)] +
                        [df[k].values.astype(float) for k in kovarijate])
    A = A[~np.isnan(A).any(axis=1)]

    def ocjena(M):
        if kovarijate:
            return parcijalna_spearman(M[:, 0], M[:, 1],
                                       *[M[:, j] for j in range(2, M.shape[1])])[0]
        return stats.spearmanr(M[:, 0], M[:, 1])[0]

    if kovarijate:
        r, p, n = parcijalna_spearman(A[:, 0], A[:, 1],
                                      *[A[:, j] for j in range(2, A.shape[1])])
    else:
        r, p, n = spearman(A[:, 0], A[:, 1])

    rng = np.random.default_rng(sjeme)
    boot = np.empty(b)
    for i in range(b):
        boot[i] = ocjena(A[rng.integers(0, n, n)])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [100 * ALFA / 2, 100 * (1 - ALFA / 2)])
    return r, p, n, lo, hi


def steiger(r_jk, r_jh, r_kh, n):
    """
    Steigerov z-test za usporedbu dviju zavisnih korelacija koje dijele
    zajedničku varijablu. Uspoređuju se apsolutne vrijednosti koeficijenata,
    jer je predmet usporedbe jačina, a ne smjer povezanosti.
    """
    r_jk, r_jh, r_kh = abs(r_jk), abs(r_jh), abs(r_kh)
    rm2 = (r_jk ** 2 + r_jh ** 2) / 2
    f = min((1 - r_kh) / (2 * (1 - rm2)), 1.0)
    h = (1 - f * rm2) / (1 - rm2)
    z = (np.arctanh(r_jk) - np.arctanh(r_jh)) * np.sqrt((n - 3) / (2 * (1 - r_kh) * h))
    return float(z), float(2 * stats.norm.sf(abs(z)))


# ------------------------------------------------------------- regresija ---
def hijerarhijska(df, ishod, blok1, blok2, blok3=None):
    """
    Hijerarhijska multipla linearna regresija.
    Blok 1: dob, spol, eGFR.  Blok 2: koncentracija mokraćne kiseline.
    Blok 3 (neobavezno): dodatni prediktor, npr. log(trigliceridi).
    """
    dat = df[[ishod] + blok1 + blok2 + (blok3 or [])].dropna()
    y = dat[ishod].values
    sd = dat.std(ddof=1)

    def fit(pred):
        return sm.OLS(y, sm.add_constant(dat[pred].astype(float).values)).fit()

    m1, m2 = fit(blok1), fit(blok1 + blok2)
    dR2 = m2.rsquared - m1.rsquared
    F = (dR2 / len(blok2)) / ((1 - m2.rsquared) / m2.df_resid)
    out = {"R2_1": m1.rsquared, "R2_2": m2.rsquared, "dR2": dR2, "F": F,
           "df": (len(blok2), int(m2.df_resid)),
           "p": float(stats.f.sf(F, len(blok2), m2.df_resid)),
           "beta": m2.params[1 + len(blok1)] * sd[blok2[0]] / sd[ishod],
           "VIF_max": max(_vif(dat[blok1 + blok2])), "n": len(dat)}
    if blok3:
        m3 = fit(blok1 + blok2 + blok3)
        out.update({"R2_3": m3.rsquared,
                    "beta_urat_3": m3.params[1 + len(blok1)] * sd[blok2[0]] / sd[ishod],
                    "p_urat_3": m3.pvalues[1 + len(blok1)],
                    "beta_3": m3.params[1 + len(blok1) + len(blok2)] * sd[blok3[0]] / sd[ishod],
                    "p_3": m3.pvalues[1 + len(blok1) + len(blok2)]})
    return out


def _vif(X):
    """Faktor povećanja varijance (VIF) za svaki prediktor."""
    X = X.astype(float)
    vifs = []
    for i, c in enumerate(X.columns):
        ostali = sm.add_constant(X.drop(columns=[c]).values)
        r2 = sm.OLS(X[c].values, ostali).fit().rsquared
        vifs.append(1 / max(1e-12, 1 - r2))
    return vifs


# ----------------------------------------------------------------- ispis ---
def red(naziv, r, p, n, lo, hi):
    print(f"  ρ ({naziv}) = {r:.3f}; 95 % CI {lo:.3f} do {hi:.3f}; p = {p:.3g}; n = {n}")


def glavna(put, sheet):
    df = ucitaj(put, sheet)
    U, TG, HDL, LDL = (STUPCI[k] for k in ("urat", "tg", "hdl", "ldl"))
    DOB, EGFR, BMI, BK = STUPCI["dob"], STUPCI["egfr"], STUPCI["bmi"], STUPCI["broj_kriterija"]
    OSNOVNE = [DOB, "spol_m", EGFR]
    P = print

    P("=" * 78); P("4.1  OPIS UZORKA"); P("=" * 78)
    P(f"n = {len(df)}; muškaraca {df.spol_m.sum()} ({100*df.spol_m.mean():.1f} %), "
      f"žena {(1-df.spol_m).sum()} ({100*(1-df.spol_m.mean()):.1f} %)")
    P(f"Dob: medijan {df[DOB].median():.0f} (IQR {df[DOB].quantile(.25):.0f}–"
      f"{df[DOB].quantile(.75):.0f}); raspon {df[DOB].min():.0f}–{df[DOB].max():.0f}")
    P(f"Metabolički sindrom: {df.mets_da.sum()} ({100*df.mets_da.mean():.1f} %)")
    dobne = pd.cut(df[DOB], [0, 39.99, 49.99, 59.99, 69.99, 200],
                   labels=["<40", "40–49", "50–59", "60–69", "≥70"])
    P("Dobne skupine (ukupno / muškarci / MetS+):")
    for g in dobne.cat.categories:
        pod = df[dobne == g]
        P(f"  {g:7s} n = {len(pod):3d} ({100*len(pod)/len(df):4.1f} %); "
          f"M {pod.spol_m.sum():2d}; MetS+ {pod.mets_da.sum():2d}")
    P(f"  χ² spol × dobna skupina: p = {stats.chi2_contingency(pd.crosstab(dobne, df.spol_m))[1]:.3g}")
    P(f"  χ² MetS × dobna skupina: p = {stats.chi2_contingency(pd.crosstab(dobne, df.mets_da))[1]:.3g}")
    P(f"  χ² MetS × spol:          p = {stats.chi2_contingency(pd.crosstab(df.spol_m, df.mets_da))[1]:.3g}")
    for ime, stupac in (("Urat", U), ("TG", TG), ("HDL", HDL), ("LDL", LDL), ("BMI", BMI),
                        ("eGFR", EGFR), ("Opseg struka", STUPCI["struk"]),
                        ("Glukoza", STUPCI["glukoza"]), ("RR sist.", STUPCI["sist"]),
                        ("RR dijast.", STUPCI["dijast"])):
        P(f"  {ime:14s} {opis(df[stupac])}")

    P(); P("=" * 78); P("4.2  NORMALNOST RASPODJELE (Shapiro-Wilk + koeficijent asimetrije)"); P("=" * 78)
    for ime, stupac in (("Urat", U), ("TG", TG), ("HDL", HDL), ("LDL", LDL)):
        x = df[stupac].dropna()
        W, p = stats.shapiro(x)
        P(f"  {ime:6s} W = {W:.3f}; p = {p:.3g}; asimetrija = {stats.skew(x, bias=False):.2f}")

    P(); P("=" * 78); P("4.3  MOKRAĆNA KISELINA PREMA SPOLU"); P("=" * 78)
    m, z = df[df.spol_m == 1][U], df[df.spol_m == 0][U]
    P(f"  Muškarci medijan {m.median():.1f} (IQR {m.quantile(.25):.1f}–{m.quantile(.75):.1f}) vs "
      f"žene {z.median():.1f} (IQR {z.quantile(.25):.1f}–{z.quantile(.75):.1f}) µmol/L; "
      f"Mann-Whitney p = {stats.mannwhitneyu(m, z)[1]:.3g}")
    hiper = ((df.spol_m == 1) & (df[U] > HIPERURICEMIJA["M"])) | \
            ((df.spol_m == 0) & (df[U] > HIPERURICEMIJA["Z"]))
    P(f"  Hiperuricemija: {hiper.sum()} ({100*hiper.mean():.1f} %); "
      f"muškarci {hiper[df.spol_m==1].sum()} ({100*hiper[df.spol_m==1].mean():.1f} %), "
      f"žene {hiper[df.spol_m==0].sum()} ({100*hiper[df.spol_m==0].mean():.1f} %)")

    P(); P("=" * 78); P("4.4  URAT ~ TRIGLICERIDI  (H1)"); P("=" * 78)
    red("bez prilagodbe", *korelacija(df, U, TG))
    red("dob, spol, eGFR", *korelacija(df, U, TG, OSNOVNE))
    red("ITM", *korelacija(df, U, TG, [BMI]))
    red("dob, spol, eGFR, ITM", *korelacija(df, U, TG, OSNOVNE + [BMI]))
    for ime, pod in (("bez hipourikemika", df[df.hipourikemik_da == 0]),
                     ("bez hipolipemika", df[df.hipolipemik_da == 0])):
        r, p, n, lo, hi = korelacija(pod, U, TG)
        P(f"  Osjetljivost — {ime}: ρ = {r:.3f}; 95 % CI {lo:.3f} do {hi:.3f}; p = {p:.3g}; n = {n}")
    A = df[[U, TG]].dropna().values
    r, p = stats.pearsonr(A[:, 0], A[:, 1])
    P(f"  Pearsonov r (usporedno) = {r:.3f}; p = {p:.3g}")

    P(); P("=" * 78); P("4.5  URAT ~ HDL-KOLESTEROL  (H2)"); P("=" * 78)
    red("bez prilagodbe", *korelacija(df, U, HDL))
    red("spol", *korelacija(df, U, HDL, ["spol_m"]))
    red("dob, spol, eGFR", *korelacija(df, U, HDL, OSNOVNE))
    red("dob, spol, eGFR, TG", *korelacija(df, U, HDL, OSNOVNE + [TG]))
    red("dob, spol, eGFR, ITM", *korelacija(df, U, HDL, OSNOVNE + [BMI]))
    for s, ime in ((1, "muškarci"), (0, "žene")):
        pod = df[df.spol_m == s]
        r, p, n = spearman(pod[U], pod[HDL])
        P(f"  ρ ({ime}) = {r:.3f}; p = {p:.3g}; n = {n}")
    P(f"  HDL po spolu: muškarci medijan {df[df.spol_m==1][HDL].median():.2f} vs "
      f"žene {df[df.spol_m==0][HDL].median():.2f} mmol/L; "
      f"p = {stats.mannwhitneyu(df[df.spol_m==1][HDL], df[df.spol_m==0][HDL])[1]:.3g}")
    kv = pd.qcut(df[U], 4, labels=[1, 2, 3, 4])
    sk = [df[kv == q][HDL].dropna() for q in (1, 2, 3, 4)]
    P(f"  Kvartili urata — medijan HDL-a: {[round(float(s.median()), 2) for s in sk]}; "
      f"Kruskal-Wallis p = {stats.kruskal(*sk)[1]:.3g}")
    P(f"  Udio muškaraca po kvartilima: {[round(100*df[kv==q].spol_m.mean()) for q in (1,2,3,4)]} %")
    for s, ime in ((1, "muškarci"), (0, "žene")):
        pod = df[df.spol_m == s].copy()
        k2 = pd.qcut(pod[U], 4, labels=[1, 2, 3, 4])
        sk = [pod[k2 == q][HDL].dropna() for q in (1, 2, 3, 4)]
        P(f"    kvartili unutar skupine {ime}: Kruskal-Wallis p = {stats.kruskal(*sk)[1]:.3g}")
    pod = df[df.hipolipemik_da == 0]
    r, p, n = spearman(pod[U], pod[HDL])
    P(f"  Osjetljivost — bez hipolipemika: ρ = {r:.3f}; p = {p:.3g}; n = {n}")

    P(); P("=" * 78); P("4.6  URAT ~ LDL-KOLESTEROL  (H3)"); P("=" * 78)
    red("bez prilagodbe", *korelacija(df, U, LDL))
    red("dob, spol, eGFR", *korelacija(df, U, LDL, OSNOVNE))
    red("ITM", *korelacija(df, U, LDL, [BMI]))
    red("dob, spol, eGFR, ITM", *korelacija(df, U, LDL, OSNOVNE + [BMI]))
    r_tg, r_hdl, r_ldl = (spearman(df[U], c)[0] for c in (df[TG], df[HDL], df[LDL]))
    for ime, r_drugi, r_par in (("TG", r_tg, spearman(df[LDL], df[TG])[0]),
                                ("HDL", r_hdl, spearman(df[LDL], df[HDL])[0])):
        z, p = steiger(r_drugi, r_ldl, r_par, len(df))
        P(f"  Steigerov z-test (LDL vs {ime}): z = {z:.2f}; p = {p:.3g}")
    for ime, pod in (("bez hipolipemika", df[df.hipolipemik_da == 0]),
                     ("na hipolipemiku", df[df.hipolipemik_da == 1])):
        P(f"  Osjetljivost {ime} (n = {len(pod)}): " + "; ".join(
            f"{nm} ρ = {spearman(pod[U], pod[c])[0]:.3f} (p = {spearman(pod[U], pod[c])[1]:.3g})"
            for nm, c in (("TG", TG), ("HDL", HDL), ("LDL", LDL))))
    a, b = df[df.hipolipemik_da == 1][LDL], df[df.hipolipemik_da == 0][LDL]
    P(f"  LDL prema terapiji: na hipolipemiku medijan {a.median():.2f} vs "
      f"bez terapije {b.median():.2f} mmol/L; p = {stats.mannwhitneyu(a, b)[1]:.3g}")

    P(); P("=" * 78); P("4.7  METABOLIČKI SINDROM  (H4)"); P("=" * 78)
    a, b = df[df.mets_da == 1][U], df[df.mets_da == 0][U]
    P(f"  MetS+ (n = {len(a)}) medijan {a.median():.1f} (IQR {a.quantile(.25):.1f}–{a.quantile(.75):.1f}) vs "
      f"MetS− (n = {len(b)}) {b.median():.1f} (IQR {b.quantile(.25):.1f}–{b.quantile(.75):.1f}); "
      f"p = {stats.mannwhitneyu(a, b)[1]:.3g}")
    for s, ime in ((1, "muškarci"), (0, "žene")):
        pod = df[df.spol_m == s]
        aa, bb = pod[pod.mets_da == 1][U], pod[pod.mets_da == 0][U]
        P(f"    {ime}: {aa.median():.1f} vs {bb.median():.1f}; p = {stats.mannwhitneyu(aa, bb)[1]:.3g}")
    red("urat ~ MetS, spol", *korelacija(df, U, "mets_da", ["spol_m"]))
    red("urat ~ MetS, dob, spol, eGFR", *korelacija(df, U, "mets_da", OSNOVNE))
    sk = [df[df[BK] == k][U].dropna() for k in range(6)]
    P("  Urat po broju zadovoljenih kriterija: " + ", ".join(
        f"{k}: n={len(s)}, medijan={s.median():.0f}" for k, s in enumerate(sk) if len(s)))
    P(f"  Kruskal-Wallis p = {stats.kruskal(*[s for s in sk if len(s)])[1]:.3g}")
    r, p, n = spearman(df[U], df[BK])
    P(f"  Spearman urat ~ broj kriterija: ρ = {r:.3f}; p = {p:.3g}; n = {n}")

    P(); P("=" * 78); P("4.8  HIJERARHIJSKA MULTIPLA LINEARNA REGRESIJA  (H5)"); P("=" * 78)
    P("  Blok 1: dob, spol, eGFR | Blok 2: + koncentracija mokraćne kiseline")
    for ime, ishod, blok3 in (("log(TG) ", "log_tg", None),
                              ("log(HDL)", "log_hdl", ["log_tg"]),
                              ("LDL     ", STUPCI["ldl"], None)):
        r = hijerarhijska(df, ishod, OSNOVNE, [U], blok3)
        P(f"  {ime}: R² {r['R2_1']:.3f} → {r['R2_2']:.3f}; ΔR² = {r['dR2']:.3f}; "
          f"F({r['df'][0]},{r['df'][1]}) = {r['F']:.2f}; p = {r['p']:.3g}; "
          f"β = {r['beta']:.3f}; VIF max = {r['VIF_max']:.2f}; n = {r['n']}")
        if blok3:
            P(f"            + blok 3 (log TG): R² = {r['R2_3']:.3f}; "
              f"β(urat) = {r['beta_urat_3']:.3f} (p = {r['p_urat_3']:.3g}); "
              f"β(log TG) = {r['beta_3']:.3f} (p = {r['p_3']:.3g})")

    P(); P("=" * 78); P("DODATNE ANALIZE OSJETLJIVOSTI"); P("=" * 78)
    d = pd.to_datetime(df[STUPCI["datum"]], errors="coerce")
    zadnja24 = df[d >= (d.max() - pd.DateOffset(months=24))]
    r, p, n, lo, hi = korelacija(zadnja24, U, TG)
    P(f"  Nalazi iz posljednja 24 mjeseca (n = {n}): ρ = {r:.3f}; "
      f"95 % CI {lo:.3f} do {hi:.3f}; p = {p:.3g}")
    P(f"  Razdoblje vađenja krvi: {d.min():%Y-%m-%d} – {d.max():%Y-%m-%d}")
    P(); P(f"[sjeme = {SJEME}; broj bootstrap uzoraka = {B_BOOT}; α = {ALFA}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--podaci", required=True, help="put do Excel datoteke s podacima")
    ap.add_argument("--sheet", default="Podaci", help="naziv lista (default: Podaci)")
    a = ap.parse_args()
    glavna(a.podaci, a.sheet)
