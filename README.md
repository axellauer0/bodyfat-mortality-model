# BMI vs. AI Body Fat Readings vs. DEXA as Mortality Rating Factors

Cox survival analysis testing whether body-fat percentage from DEXA scans is a better predictor for mortality than BMI + waist circumference, and if a simulated AI body-fat estimate from a smartphone could be a cheap and reliable substitute for a DEXA scan.

## Data
NHANES 2005–2006 (demographics, body measures, DEXA) with its Linked Mortality File follow up in 2019. 
Final sample: **3,971 adults, 322 deaths**.

## Method
Three Cox proportional-hazards models, each controlling for age and sex, differing only in the body variable:
- **Simulated AI body fat** — DEXA + bias/noise from a smartphone AI body-fat study (Qiao et al. 2024), run as a 1,000-iteration Monte Carlo
- **DEXA**
- **BMI + waist**

Compares the models C-indexes

## Result
All three land within ~0.005 (C-index ≈ 0.80). Once age and sex are controlled, DEXA, BMI + waist, and the noisy AI estimate perform almost all equivalently. Takeaway: for mortality ranking, the expensive scan isn't worth the high costs and a phone estimate wouldn't lose meaningful signal, but a BMI + waist circumference reading performs just as well if not better.

Model                      C-index
BMI + waist                0.8043
AI body fat (1000 sim)     0.7993
DEXA                       0.7993

## Notes
- DEXA data had 5 scans per participant, collapsed by averaging.
- Model tracks all-cause mortality. Cardiovascular-specific & diabetic related deaths are a next step.
- The AI model simulates error on the DEXA scan data; it measures signal loss vs. DEXA, since there is no data tracking mortality on AI body-fat readings.
