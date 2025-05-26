# Rookie Directors & Market Reaction
This project explores the market reaction to the appointment of rookie directors using event studies and statistical analysis in Python. The project encapsulates multiple levels of data wrangling and transformation before conducting analysis, to get a better view of certain necessary characteristics of existing or generated data points.

---

## Objectives
- Transform director level data to output important variables like skills, past directorships, closing balance of directorships, etc.
- Use PCA (PRinciple Component Analysis) to generate General Ability Index of Directors based on skills, experience, etc.
- Tranform the above outputs to extract firm level board chareacteristics.
- Conduct firm level PSM and director level PSM analysis on various dependent variables and extract their statistical significance for study.

---

## Methodology Overview
Each step in this project is clearly separated as sub folders within the root. This is to maintain efficiency and exit points for data extraction and inference whenever needed.
1. **Director level data wrangling and transformation**
   - Merge and clean director and firm-level data.
   - Use PCA to generate control-treatment pairs on IsRookieIndep
   - Match pairs using nearest neighbours approach, with replacement and generate the principle component n=1.
2. **Firm level data wrangling and transformation**
   - Conduct multiple levels of transformations to get firm level characteristics, and merge firm level financial data
   - Conduct further transformation to output City level characteristics.
3. **Event Study using PSM on a number of exogenous variables for Director level data**
   - Calculate CAR using CAPM and FF3.
   - Perform T-tests for statistical significance on Rookie Director Appointments on Director level data.
4. **Event Study using PSM on a number of exogenous variables for Firm level data**
   - Calculate CAR using CAPM and FF3.
   - Perform T-tests for statistical significance on % Rookie Director Appointments on firm level data.
   - Study statistical significance on multiple dependent variables

---

## Project Structure

```bash
rookie-directors
├── .gitignore
├── Data Files
│   └── [CMIE] Data Files  # Collection of financial datafiles from CMIE Prowess 
│       └── Prowess IQ Custom
│           └── Adjusted Closing Price
│               └── Adjusted Closing Price Collation.ipynb
├── [0] Archive
└── [IN USE] Rookie Directors
    ├── [1.0] Director Level
    │   ├── Individual years data  # Contains Prime Infobase Indian Directors Data of NSE listed companies
    │   │   └── [190525] Main_Individual Years_Merge_incl Cess.ipynb # Prepares long data from the aforementioned files
    │   ├── [190525] Main_Director_Wrangle.ipynb # Director level transformation
    │   └── director_wrangle_output
    │       └── PCA.ipynb # Principal Component Analysis on output pickle file
    ├── [2] Firm Level Wrangling
    │   ├── Firm level Iterative 
    │   │   ├── Firm Lev + Fin
    │   │   │   ├── City level 
    │   │   │   │   └── [190525] City level Calc_FirDirCity.ipynb  # Firm Level transformation step 3
    │   │   │   └── [190525] Firm_Finance_Calc.ipynb  # Firm Level transformation step 2
    │   │   └── [190525] Firm level Calc.ipynb  # Firm Level transformation step 1 after Director level transformation
    ├── [3] PSM  # Propensity Score Matching analysis on Firm level output
    ├── [4] CAPM CAR precode   # CAPM CAR Eventstudy analysis on Director level output
    └── [4] FF3M CAR precode   # FF3 CAR Eventstudy analysis on Director level output


```

## 🔒 Important Notes

  - [0] OBS folders contain older version of files from their current directory, and can be safely ignored.
  - [0] Archive folder contains older version of high level project tasks mentioned in the root directory, and can be safely ignored.
  - `.pkl`, `.txt`, `.csv`, `.pdf`, have been ignored via `.gitignore` to preserve the confidentiality of the data files used.
  - Please contact me for the raw data files if you wish to run this project. Requests from only authorised people will be processed over private and secure channels.
  - This project is a work in progress.

---

## Getting Started
Ensure you have `pandas`, `scipy`, `numpy`, `sklearn`, `statsmodels`, and `matplotlib` installed.

```bash
git clone https://github.com/Leshleon/rookie-directors.git
cd rookie-directors
```
Note: Actual data files will be required to run notebooks.

---

## Contact
Please find below my contact information. While I am wary of sharing raw files, I would be glad to have discussions where I can.

**Shivam S**
- Email: shivam.s@learner.manipal.edu
- GitHub: github.com/Leshleon

---

## Keywords
event study, finance, CAPM, Fama-French, rookie directors, PSM, Python, financial markets, data wrangling, corporate governance, statistical analysis
