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

Skripta ispisuje sve brojke iz poglavlja Rezultati, redoslijedom poglavlja 4
završnog specijalističkog rada (u zagradama odgovarajući odjeljak članka).
Autorica istraživanja / Author of the study: Anita Jakulj, dr. med.
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

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
    "sifra": "Šifra pacijenta", "kreatinin": "Kreatinin\n(μmol/L)",
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
    X2 = sm.add_constant(dat[blok1 + blok2].astype(float).values)
    vifs = _vif(dat[blok1 + blok2])
    dR2 = m2.rsquared - m1.rsquared
    F = (dR2 / len(blok2)) / ((1 - m2.rsquared) / m2.df_resid)
    out = {"R2_1": m1.rsquared, "R2_2": m2.rsquared, "dR2": dR2, "F": F,
           "df": (len(blok2), int(m2.df_resid)),
           "p": float(stats.f.sf(F, len(blok2), m2.df_resid)),
           "beta": m2.params[1 + len(blok1)] * sd[blok2[0]] / sd[ishod],
           "VIF_min": min(vifs), "VIF_max": max(vifs), "n": len(dat),
           "BP_p": float(het_breuschpagan(m2.resid, X2)[1]),
           "SW_p": float(stats.shapiro(m2.resid)[1])}
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
# Ispis slijedi poglavlje 4 (Rezultati) završnog specijalističkog rada; u
# zagradama je odgovarajući odjeljak članka (Results, 3.x).
P = print


def fp(p):
    """p-vrijednost: '= 0.xxxx' (četiri decimale) ili '< 0.001 (točna vrijednost)'."""
    return f"= {p:.4f}" if p >= 0.001 else f"< 0.001 ({p:.2g})"


def naslov(t):
    P(); P("=" * 78); P(t); P("=" * 78)


def red(naziv, r, p, n, lo, hi):
    P(f"  ρ ({naziv}) = {r:.3f}; 95 % CI {lo:.3f} do {hi:.3f}; p {fp(p)}; n = {n}")


def med_iqr(x, d):
    x = pd.Series(x).dropna()
    return f"{x.median():.{d}f} ({x.quantile(.25):.{d}f}–{x.quantile(.75):.{d}f})"


def glavna(put, sheet):
    df = ucitaj(put, sheet)
    U, TG, HDL, LDL = (STUPCI[k] for k in ("urat", "tg", "hdl", "ldl"))
    DOB, EGFR, BMI, BK = STUPCI["dob"], STUPCI["egfr"], STUPCI["bmi"], STUPCI["broj_kriterija"]
    OSNOVNE = [DOB, "spol_m", EGFR]
    M, Z = df.spol_m == 1, df.spol_m == 0

    # ---------------------------------------------------------------- 4.1
    naslov("4.1  OPIS UZORKA  (članak 3.1; Tablice 2 i 3)")
    P(f"  n = {len(df)}; muškaraca {M.sum()} ({100*M.mean():.1f} %), žena {Z.sum()} ({100*Z.mean():.1f} %)")
    P(f"  Dob: medijan {df[DOB].median():.0f} (IQR {df[DOB].quantile(.25):.0f}–{df[DOB].quantile(.75):.0f}); "
      f"raspon {df[DOB].min():.0f}–{df[DOB].max():.0f}")
    dobne = pd.cut(df[DOB], [0, 39.99, 49.99, 59.99, 69.99, 200],
                   labels=["< 40 godina", "40 – 49", "50 – 59", "60 – 69", "≥ 70 godina"])
    P(f"  χ² spol × dobna skupina: p {fp(stats.chi2_contingency(pd.crosstab(dobne, df.spol_m))[1])}")
    P(f"  Metabolički sindrom: {df.mets_da.sum()} ({100*df.mets_da.mean():.1f} %); muškarci "
      f"{100*df[M].mets_da.mean():.1f} % vs žene {100*df[Z].mets_da.mean():.1f} %; "
      f"χ² p {fp(stats.chi2_contingency(pd.crosstab(df.spol_m, df.mets_da))[1])}")
    P(f"  χ² MetS × dobna skupina: p {fp(stats.chi2_contingency(pd.crosstab(dobne, df.mets_da))[1])}")
    P()
    P("  Tablica 2 - Raspodjela ispitanika prema dobi, spolu i prisutnosti MetS-a")
    P(f"  {'Dobna skupina':14s}{'Ukupno':>14s}{'Muškarci':>14s}{'Žene':>14s}{'MetS +':>14s}{'MetS −':>14s}")
    def pc(a, b):
        return f"{a} ({100*a/b:.1f} %)"
    for g in dobne.cat.categories:
        pod = df[dobne == g]; n = len(pod)
        P(f"  {g:14s}{pc(n, len(df)):>14s}{pc(pod.spol_m.sum(), n):>14s}{pc(n-pod.spol_m.sum(), n):>14s}"
          f"{pc(pod.mets_da.sum(), n):>14s}{pc(n-pod.mets_da.sum(), n):>14s}")
    n = len(df)
    P(f"  {'Ukupno':14s}{f'{n} (100 %)':>14s}{pc(M.sum(), n):>14s}{pc(Z.sum(), n):>14s}"
      f"{pc(df.mets_da.sum(), n):>14s}{pc(n-df.mets_da.sum(), n):>14s}")
    P()
    P("  Tablica 3 - Osnovna obilježja uzorka: medijan (IQR)")
    for ime, stupac, d in (("Dob (godine)", DOB, 0), ("Indeks tjelesne mase (kg/m²)", BMI, 1),
                           ("Opseg struka (cm)", STUPCI["struk"], 0),
                           ("Sistolički tlak (mmHg)", STUPCI["sist"], 0),
                           ("Dijastolički tlak (mmHg)", STUPCI["dijast"], 0),
                           ("Mokraćna kiselina (µmol/L)", U, 1), ("Trigliceridi (mmol/L)", TG, 2),
                           ("HDL-kolesterol (mmol/L)", HDL, 2), ("LDL-kolesterol (mmol/L)", LDL, 2),
                           ("Glukoza natašte (mmol/L)", STUPCI["glukoza"], 1),
                           ("Kreatinin (µmol/L)", STUPCI["kreatinin"], 0),
                           ("eGFR (mL/min/1,73 m²)", EGFR, 0)):
        P(f"  {ime:32s} {med_iqr(df[stupac], d):>22s}   (nedostaje: {df[stupac].isna().sum()})")

    # ---------------------------------------------------------------- 4.2
    naslov("4.2  PROVJERA NORMALNOSTI RASPODJELE  (članak 3.1)")
    for ime, stupac in (("Mokraćna kiselina", U), ("Trigliceridi", TG), ("HDL-kolesterol", HDL),
                        ("LDL-kolesterol", LDL)):
        x = df[stupac].dropna(); W, p = stats.shapiro(x)
        P(f"  {ime:18s} Shapiro-Wilk W = {W:.3f}; p {fp(p)}; asimetrija = {stats.skew(x, bias=False):.2f}")

    # ---------------------------------------------------------------- 4.3
    naslov("4.3  MOKRAĆNA KISELINA PREMA SPOLU  (članak 3.2)")
    P(f"  Cijeli uzorak: medijan {med_iqr(df[U], 1)} µmol/L")
    P(f"  Muškarci {med_iqr(df[M][U], 1)} vs žene {med_iqr(df[Z][U], 1)} µmol/L; "
      f"Mann-Whitney p {fp(stats.mannwhitneyu(df[M][U], df[Z][U])[1])}")
    hiper = (M & (df[U] > HIPERURICEMIJA["M"])) | (Z & (df[U] > HIPERURICEMIJA["Z"]))
    P(f"  Hiperuricemija (> 400 µmol/L M, > 340 µmol/L Ž): {hiper.sum()} ({100*hiper.mean():.1f} %); "
      f"muškarci {hiper[M].sum()} ({100*hiper[M].mean():.1f} %), žene {hiper[Z].sum()} ({100*hiper[Z].mean():.1f} %)")

    # ---------------------------------------------------------------- 4.4
    naslov("4.4  MOKRAĆNA KISELINA I TRIGLICERIDI, H1  (članak 3.3; Slika 1)")
    red("bez prilagodbe", *korelacija(df, U, TG))
    red("dob, spol, eGFR", *korelacija(df, U, TG, OSNOVNE))
    red("ITM, zasebni model", *korelacija(df, U, TG, [BMI]))
    red("dob, spol, eGFR i ITM zajedno — dodatno, nije navedeno u radu", *korelacija(df, U, TG, OSNOVNE + [BMI]))
    P(f"  Na hipourikemiku: {df.hipourikemik_da.sum()} ({100*df.hipourikemik_da.mean():.1f} %)")
    red("bez ispitanika na hipourikemiku", *korelacija(df[df.hipourikemik_da == 0], U, TG))
    d = pd.to_datetime(df[STUPCI["datum"]], errors="coerce")
    red("nalazi iz posljednja 24 mjeseca", *korelacija(df[d >= (d.max() - pd.DateOffset(months=24))], U, TG))
    A = df[[U, TG]].dropna().values
    r, p = stats.pearsonr(A[:, 0], A[:, 1])
    P(f"  Pearsonov r = {r:.3f}; p {fp(p)}")
    rez = []
    for ime, maska in (("bez hiperuricemije", ~hiper), ("s hiperuricemijom", hiper)):
        rr = korelacija(df[maska], U, TG); rez.append(rr); red(ime, *rr)
    (ra, _, na, _, _), (rb, _, nb, _, _) = rez
    zf = (np.arctanh(ra) - np.arctanh(rb)) / np.sqrt(1 / (na - 3) + 1 / (nb - 3))
    P(f"  Fisherov z = {zf:.2f}; p {fp(2*stats.norm.sf(abs(zf)))}")
    P(f"  Razdoblje vađenja krvi: {d.min():%Y-%m-%d} – {d.max():%Y-%m-%d}; "
      f"nalazi od 2024. nadalje: {(d.dt.year >= 2024).sum()} ({100*(d.dt.year >= 2024).mean():.1f} %)")

    # ---------------------------------------------------------------- 4.5
    naslov("4.5  MOKRAĆNA KISELINA I HDL-KOLESTEROL, H2  (članak 3.4; Tablica 4; Slika 1)")
    P("  Tablica 4 - Korelacija prije i nakon prilagodbi")
    red("cijeli uzorak", *korelacija(df, U, HDL))
    red("spol", *korelacija(df, U, HDL, ["spol_m"]))
    red("dob, spol, eGFR", *korelacija(df, U, HDL, OSNOVNE))
    red("dob, spol, eGFR, TG", *korelacija(df, U, HDL, OSNOVNE + [TG]))
    red("dob, spol, eGFR, ITM", *korelacija(df, U, HDL, OSNOVNE + [BMI]))
    red("muškarci", *korelacija(df[M], U, HDL))
    red("žene", *korelacija(df[Z], U, HDL))
    P(f"  Bez hipolipemika: ρ = {spearman(df[df.hipolipemik_da == 0][U], df[df.hipolipemik_da == 0][HDL])[0]:.3f}; "
      f"p {fp(spearman(df[df.hipolipemik_da == 0][U], df[df.hipolipemik_da == 0][HDL])[1])}; "
      f"n = {(df.hipolipemik_da == 0).sum()}")
    P(f"  HDL: muškarci {med_iqr(df[M][HDL], 2)} vs žene {med_iqr(df[Z][HDL], 2)} mmol/L; "
      f"Mann-Whitney p {fp(stats.mannwhitneyu(df[M][HDL], df[Z][HDL])[1])}")
    kv = pd.qcut(df[U], 4, labels=[1, 2, 3, 4])
    P(f"  Granice kvartila mokraćne kiseline (µmol/L): "
      f"{', '.join(f'{g:.1f}' for g in df[U].quantile([.25, .5, .75]))}")
    sk = [df[kv == q][HDL] for q in (1, 2, 3, 4)]
    P(f"  Kvartili, cijeli uzorak — medijan HDL-a: {', '.join(f'{s.median():.2f}' for s in sk)}; "
      f"Kruskal-Wallis p {fp(stats.kruskal(*sk)[1])}")
    P(f"  Udio muškaraca po kvartilima: {', '.join(f'{100*df[kv == q].spol_m.mean():.0f} %' for q in (1, 2, 3, 4))}")
    for maska, ime in ((M, "muškarci"), (Z, "žene")):
        pod = df[maska]; k2 = pd.qcut(pod[U], 4, labels=[1, 2, 3, 4])
        P(f"  Kvartili unutar skupine {ime}: Kruskal-Wallis p = "
          f"{stats.kruskal(*[pod[k2 == q][HDL] for q in (1, 2, 3, 4)])[1]:.3f}")

    # ---------------------------------------------------------------- 4.6
    naslov("4.6  MOKRAĆNA KISELINA I LDL-KOLESTEROL, H3  (članak 3.5; Slika 1)")
    red("bez prilagodbe", *korelacija(df, U, LDL))
    red("dob, spol, eGFR", *korelacija(df, U, LDL, OSNOVNE))
    red("ITM, zasebni model", *korelacija(df, U, LDL, [BMI]))
    red("dob, spol, eGFR i ITM zajedno — dodatno, nije navedeno u radu", *korelacija(df, U, LDL, OSNOVNE + [BMI]))
    r_tg, r_hdl, r_ldl = (spearman(df[U], df[c])[0] for c in (TG, HDL, LDL))
    for ime, r_drugi, r_par in (("TG", r_tg, spearman(df[LDL], df[TG])[0]),
                                ("HDL", r_hdl, spearman(df[LDL], df[HDL])[0])):
        z, p = steiger(r_drugi, r_ldl, r_par, len(df))
        P(f"  Steigerov z (LDL naspram {ime}) = {z:.2f}; p {fp(p)}")
    P(f"  Na hipolipemiku: {df.hipolipemik_da.sum()} ({100*df.hipolipemik_da.mean():.1f} %)")
    for ime, pod in (("bez hipolipemika", df[df.hipolipemik_da == 0]), ("na hipolipemiku", df[df.hipolipemik_da == 1])):
        P(f"  {ime} (n = {len(pod)}): " + "; ".join(
            f"{nm} ρ = {spearman(pod[U], pod[c])[0]:.3f} (p {fp(spearman(pod[U], pod[c])[1])})"
            for nm, c in (("TG", TG), ("HDL", HDL), ("LDL", LDL))))

    # ---------------------------------------------------------------- 4.7
    naslov("4.7  MOKRAĆNA KISELINA I METABOLIČKI SINDROM, H4  (članak 3.6)")
    a, b = df[df.mets_da == 1][U], df[df.mets_da == 0][U]
    P(f"  MetS+ (n = {len(a)}) {med_iqr(a, 1)} vs MetS− (n = {len(b)}) {med_iqr(b, 1)} µmol/L; "
      f"Mann-Whitney p {fp(stats.mannwhitneyu(a, b)[1])}")
    for maska, ime in ((M, "muškarci"), (Z, "žene")):
        pod = df[maska]; aa, bb = pod[pod.mets_da == 1][U], pod[pod.mets_da == 0][U]
        P(f"    {ime}: {aa.median():.1f} vs {bb.median():.1f} µmol/L; p {fp(stats.mannwhitneyu(aa, bb)[1])}")
    red("urat ~ MetS, prilagodba za spol", *korelacija(df, U, "mets_da", ["spol_m"]))
    red("urat ~ MetS, prilagodba za dob, spol, eGFR", *korelacija(df, U, "mets_da", OSNOVNE))
    sk = [df[df[BK] == k][U].dropna() for k in range(6)]
    P("  Prema broju zadovoljenih kriterija: " + "; ".join(
        f"{k}: n = {len(s)}, medijan {s.median():.1f}" for k, s in enumerate(sk) if len(s)))
    P(f"  Kruskal-Wallis p {fp(stats.kruskal(*[s for s in sk if len(s)])[1])}")
    r, p, n = spearman(df[U], df[BK])
    P(f"  Spearman, urat ~ broj kriterija: ρ = {r:.3f}; p {fp(p)}; n = {n}")

    # ---------------------------------------------------------------- 4.8
    naslov("4.8  HIJERARHIJSKA MULTIPLA LINEARNA REGRESIJA, H5  (članak 3.7; Tablica 5)")
    P("  Blok 1: dob, spol, eGFR | Blok 2: + mokraćna kiselina")
    for ime, ishod, blok3 in (("log(TG) ", "log_tg", None), ("log(HDL)", "log_hdl", ["log_tg"]),
                              ("LDL     ", STUPCI["ldl"], None)):
        r = hijerarhijska(df, ishod, OSNOVNE, [U], blok3)
        P(f"  {ime}: R² {r['R2_1']:.3f} ({100*r['R2_1']:.1f} %) → {r['R2_2']:.3f} ({100*r['R2_2']:.1f} %); "
          f"ΔR² = {r['dR2']:.3f}; F({r['df'][0]},{r['df'][1]}) = {r['F']:.2f}; p {fp(r['p'])}; β = {r['beta']:.3f}")
        P(f"            VIF {r['VIF_min']:.2f}–{r['VIF_max']:.2f}; Breusch-Pagan p {fp(r['BP_p'])}; "
          f"Shapiro-Wilk reziduala p {fp(r['SW_p'])}; n = {r['n']}")
        if blok3:
            P(f"            + blok 3 (log TG): R² = {r['R2_3']:.3f} ({100*r['R2_3']:.1f} %); "
              f"β(urat) = {r['beta_urat_3']:.3f} (p {fp(r['p_urat_3'])}); "
              f"β(log TG) = {r['beta_3']:.3f} (p {fp(r['p_3'])})")
    P(); P(f"[sjeme = {SJEME}; broj bootstrap uzoraka = {B_BOOT}; α = {ALFA}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--podaci", required=True, help="put do Excel datoteke s podacima")
    ap.add_argument("--sheet", default="Podaci", help="naziv lista (default: Podaci)")
    a = ap.parse_args()
    glavna(a.podaci, a.sheet)
