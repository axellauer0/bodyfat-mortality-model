import pandas as pd
import numpy as np
import pyreadstat
from lifelines import CoxPHFitter

#load data files
demo, _ = pyreadstat.read_xport("DEMO_D.XPT")   #Demographics
bmx,  _ = pyreadstat.read_xport("BMX_D.XPT")    #Body Measures
dxx,  _ = pyreadstat.read_xport("DXX_D.XPT")    #DEXA scans

#gets rid of unnecessary columns
demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]   # id, age, sex
bmx  = bmx[["SEQN", "BMXBMI", "BMXWAIST"]]      # id, BMI, waist
dxx = dxx[["SEQN", "DXDTOPF"]]                  # id, DEXA BF%

#parse mortality file
MORT_FILE = "NHANES_2005_2006_MORT_2019_PUBLIC.dat"
colspecs = [
    (0, 6),      # SEQN
    (14, 15),    # ELIGSTAT   1=eligible, 2=ineligible
    (15, 16),    # MORTSTAT   0=alive, 1=dead
    (42, 45),    # PERMTH_INT  months between interview and death or interview and follow up
]
names = ["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_INT"]

mort = pd.read_fwf(MORT_FILE, colspecs=colspecs, names=names)
for c in names:
    mort[c] = pd.to_numeric(mort[c], errors="coerce")
mort = mort.dropna(subset=["SEQN"])
mort["SEQN"] = mort["SEQN"].astype(int)

#print("MORTSTAT values:", sorted(mort["MORTSTAT"].dropna().unique()))
#print("PERMTH_INT range:", mort["PERMTH_INT"].min(), "-", mort["PERMTH_INT"].max())
#print("total deaths in file:", int((mort["MORTSTAT"] == 1).sum()))
#print("mortality file loaded:", mort.shape)
#print(mort[["SEQN", "MORTSTAT", "PERMTH_INT"]].head())

#averages the 5 DEXA scans done for each person
dxx = dxx.groupby("SEQN", as_index=False)["DXDTOPF"].mean()
#print("\nDEXA collapsed to one row/person:", dxx.shape)

#merges DEMO, BMX, DXX, and MORT into one big data frame
df = demo.merge(bmx, on="SEQN", how="inner")
df = df.merge(dxx, on="SEQN", how="inner")
df = df.merge(mort[["SEQN", "MORTSTAT", "PERMTH_INT", "ELIGSTAT"]], on="SEQN", how="inner")
#print("\nafter all merges:", df.shape)

df = df[df["ELIGSTAT"] == 1]              # drops rows with ineligible people
df = df[df["RIDAGEYR"] >= 18]             # gets rid of the kids
df = df.rename(columns={
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "BMXBMI":   "bmi",
    "BMXWAIST": "waist",
    "DXDTOPF":  "bodyfat",
    "MORTSTAT": "event",
    "PERMTH_INT": "duration",
})
df = df.dropna(subset=["age", "sex", "bmi", "waist", "bodyfat", "event", "duration"])
df["event"] = df["event"].astype(int)
df["duration"] = df["duration"].astype(float)

#since Cox will break if duration is 0
df.loc[df["duration"] <= 0, "duration"] = 0.5

print("\nfinal modeling dataframe:", df.shape)
print("deaths in sample:", int(df["event"].sum()))
print("follow-up months range:", df["duration"].min(), "-", df["duration"].max())

#cox model
def fit(cols, label):
    m = CoxPHFitter()
    m.fit(df[cols + ["duration", "event"]],
          duration_col="duration", event_col="event")
    print(f"\n{label}:  C-index = {m.concordance_index_:.4f}")
    return m

#simulate AI bodyfat readings adjustable amount of times & takes avg. std. & 95% range
#from Qiao et al. 2024 (npj Digit Med), smartphone RGB validation:
#percentage body fat mean bias +1.62%, 95% LoA (-9.2, +12.5)
#LoA half-width / 1.96 = SD  ->  (12.5-(-9.2))/2 / 1.96 = 5.5
#AI image estimate = DEXA bf% + fixed bias + Gaussian noise(SD 5.5)
AI_BIAS = 1.62
AI_SD   = 5.5
N_SIMS  = 1000

ai_c_indexes = []
for i in range(N_SIMS):
    df["bodyfat_ai"] = df["bodyfat"] + AI_BIAS + np.random.normal(0, AI_SD, size=len(df))

    m = CoxPHFitter()
    m.fit(df[["bodyfat_ai", "age", "sex", "duration", "event"]],
          duration_col="duration", event_col="event")
    ai_c_indexes.append(m.concordance_index_)

ai_c_indexes = np.array(ai_c_indexes)
c2_mean = ai_c_indexes.mean()
c2_std  = ai_c_indexes.std()
c2_lo, c2_hi = np.percentile(ai_c_indexes, [2.5, 97.5])

print(f"\nSimulated AI body fat ({N_SIMS} sims):")
print(f"    mean C-index = {c2_mean:.4f}")
print(f"    std          = {c2_std:.4f}")
print(f"    95% range    = {c2_lo:.4f} - {c2_hi:.4f}")

#runs/prints results of other 2 models
m2 = fit(["bodyfat", "age", "sex"],      "DEXA Body Fat ")
m3 = fit(["bmi", "waist", "age", "sex"],      "BMI + waist")
