# OnlineCovCPD: A spectral approach for online covariance change point detection

---

This repository contains the code for the paper "A spectral approach for online covariance change point detection" by Zhigang Bao, Kha Man Cheong, Yuji Li, Jiaxin Qiu (2026).

## Project structure

```
OnlineCovCPD/
├── README.md
├── OnlineCovCPD_utils.jl     # Tool functions
├── Project.toml              # Julia deps (julia --project=. -e 'using Pkg; Pkg.instantiate()')
├── requirements.txt          # Python deps (for notebooks/plots)
├── simulation/               # Size (H0) and power (H1) experiments
│   ├── H0.jl                 # Size under no change
│   ├── H1.jl                 # Power and EDD
│   ├── H1_plot.ipynb         # Figures 2 - 5 in the paper
│   ├── comparison.jl         # Comparison with existing methods 
│   ├── LSS_diff_BM_plot.jl   # Figure 1 (cumulative LSS difference)
│   ├── data/                 # Simulation outputs (CSV)
│   └── tex/                  # LaTeX tables from results
└── real_data/                # Real-data example (S&P 500)
    ├── get_data.jl           # Fetch and clean S&P 500 data, get 'sp500_*.csv'
    ├── real_data_example.jl  # S&P 500 data analysis 
    └── sp500_*.csv           # Preprocessed data
```

--- 
## Quick start

The results of the empirical studies presented in our paper can be replicated by running the Julia scripts or Jupyter notebook found in the directory.

```bash
# ======================
# 1 Dependencies
# ======================
julia --project=. -e 'using Pkg; Pkg.instantiate()'
pip install -r requirements.txt

# ======================
# 2 Run Simulation
# ======================
julia simulation/H0.jl # Table 2
julia simulation/H1.jl # Tables 3 - 5
julia simulation/comparison.jl # Table 6
# Results are written to 'simulation/data/' 
# and can be turned into LaTeX tables using 'csv2tex*.py' under 'simulation/tex/'.
# Run 'H1_plot.ipynb' to get Figures 2 - 5

# ======================
# 3 Real Data Example 
# ======================
julia real_data/get_data.jl # get 'sp500_*.csv' under 'real_data/'
julia real_data/real_data_example.jl # Figure 6
``` 
---
