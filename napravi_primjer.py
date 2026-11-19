# -*- coding: utf-8 -*-
"""
Generates a fully SYNTHETIC example dataset (primjer_podaci.xlsx) with the same
column layout as the real data entry table, so that analiza.py can be run
end-to-end without access to patient data. No value is taken from real
patients: all values are drawn from simple parametric models.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n = 180

muski = rng.random(n) < 0.5
dob = np.clip(rng.normal(58, 15, n), 22, 87).round()
z = rng.normal(0, 1, n)                       # latent "metabolic" factor

bmi = np.clip(29 + 4.5 * z + rng.normal(0, 2.5, n), 18, 50).round(2)
struk = np.clip(88 + 12 * z + 9 * muski + rng.normal(0, 7, n), 60, 140).round()
tg = np.clip(np.exp(np.log(1.5) + 0.40 * z + rng.normal(0, 0.35, n)), 0.4, 10).round(1)
hdl = np.clip(1.55 - 0.15 * z - 0.25 * muski + rng.normal(0, 0.25, n), 0.5, 2.6).round(1)
ldl = np.clip(rng.normal(3.4, 1.1, n), 0.9, 7.2).round(1)
glu = np.clip(5.2 + 0.7 * z + np.abs(rng.normal(0, 0.6, n)), 3.5, 13).round(1)
sist = np.clip(122 + 0.25 * dob + 5 * z + rng.normal(0, 11, n), 95, 200).round()
dijast = np.clip(80 + 3 * z + rng.normal(0, 7, n), 60, 105).round()
egfr = np.clip(125 - 0.65 * dob + rng.normal(0, 10, n), 35, 125).round()
urat = np.clip(285 + 60 * muski + 35 * z - 0.3 * (egfr - 85) + rng.normal(0, 65, n),
               150, 750).round()
hipolipemik = np.where(rng.random(n) < 0.25, "Da", "Ne")
hipourikemik = np.where(rng.random(n) < 0.03, "Da", "Ne")

# MetS criteria (harmonised definition)
k1 = np.where(muski, struk >= 94, struk >= 80)
k2 = tg >= 1.7
k3 = np.where(muski, hdl < 1.0, hdl < 1.3)
k4 = (sist >= 130) | (dijast >= 85)
k5 = glu >= 5.6
broj = (k1.astype(int) + k2 + k3 + k4 + k5)

pocetak = pd.Timestamp("2022-06-01").value // 10**9
kraj = pd.Timestamp("2026-08-31").value // 10**9
datum = pd.to_datetime(rng.integers(pocetak, kraj, n), unit="s").normalize()

df = pd.DataFrame({
    "Redni br.": np.arange(1, n + 1),
    "Šifra pacijenta": [f"SYN{i:03d}" for i in range(1, n + 1)],
    "Datum vađenja krvi": datum,
    "Dob": dob,
    "Spol": np.where(muski, "M", "Ž"),
    "BMI": bmi,
    "opseg struka": struk,
    "TT sistolički\n(mmHg)": sist,
    "TT dijastolički\n(mmHg)": dijast,
    "Urat\n(μmol/L)": urat,
    "LDL\n(mmol/L)": ldl,
    "HDL\n(mmol/L)": hdl,
    "TG\n(mmol/L)": tg,
    "Glukoza\n(mmol/L)": glu,
    "eGFR\n(mL/min/1.73m²)": egfr,
    "Hipolipemik": hipolipemik,
    "Hiopurikemik": hipourikemik,
    "Broj kriterija": broj,
    "MetS": np.where(broj >= 3, "Da", "Ne"),
})

with pd.ExcelWriter("primjer_podaci.xlsx") as w:
    df.to_excel(w, sheet_name="Podaci", index=False)
print(df.shape, "MetS:", (df.MetS == "Da").sum())
