
# Proof-of-Concept SEAS5 Forecast Evaluation

This repository contains a **proof-of-concept workflow** for evaluating SEAS5 seasonal forecasts against observational datasets over the U.S. Midwest. It includes data downloading, preprocessing, anomaly correlation analysis, and visualization of ensemble forecast skill.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Repository Structure](#repository-structure)  
- [Workflow](#workflow)  
- [Dependencies](#dependencies)  
- [Usage](#usage)  
- [Output](#output)  
- [License](#license)  

---

## Project Overview

This project demonstrates a proof-of-concept evaluation of SEAS5 forecasts for temperature variables (`T2MAX`, `T2MIN`). It focuses on:

- Downloading SEAS5 and observational (OBS) datasets.
- Extracting region-specific averages for the Midwest.
- Computing **interannual anomaly correlations** (IAC) to assess forecast skill.
- Classifying ensemble members based on percentile-based skill categories.
- Visualizing correlations and skill metrics for each forecast start month and lead time.

> **Note:** This is a proof-of-concept; certain steps use a limited number of years or simplified processing.

---

## Repository Structure

```
├── 01_download_data.py       # Script to download SEAS5 and OBS NetCDF files
├── 02_extract_data.ipynb     # Notebook to extract and aggregate regional averages
├── 03_AIC_analysis.ipynb     # Notebook to compute anomaly correlations and classify ensemble skill
├── 04_visualize.ipynb        # Notebook to visualize percentile correlations and skill
├── data/                     # Folder to store downloaded NetCDF files
├── output/                   # Folder to store processed CSVs and results
└── README.md                 # This documentation
```

---

## Workflow

The workflow follows four main steps:

1. **Download Data**  
   Use `01_download_data.py` to download:
   - Observational (OBS) datasets.
   - SEAS5 seasonal forecast datasets (from 1993 onward).  

2. **Extract Regional Data**  
   `02_extract_data.ipynb`:
   - Reads downloaded NetCDF files.
   - Applies a spatial mask for the U.S. Midwest (latitude 36–49°, longitude -105 to -84°).
   - Computes regional averages for `T2MAX` and `T2MIN`.
   - Saves processed data to CSV for further analysis.

3. **Compute Anomaly Correlations (IAC)**  
   `03_AIC_analysis.ipynb`:
   - Adds year and month columns.
   - Computes anomalies by removing monthly climatology.
   - Calculates correlation between forecast anomalies and observations.
   - Classifies ensemble members into **problematic**, **acceptable**, or **good** categories based on percentiles (33rd, 50th, 66th).

4. **Visualization**  
   `04_visualize.ipynb`:
   - Plots anomaly correlation percentiles vs. lead time.
   - Generates grid plots for January–September start months.
   - Creates grouped bar charts for the 66th percentile (`p66`) correlations.

**Workflow Diagram:**

```
+-------------------+
| 01_download_data  |
+--------+----------+
         |
         v
+-------------------+
| 02_extract_data   |
|  - Mask Midwest   |
|  - Compute means  |
+--------+----------+
         |
         v
+-------------------+
| 03_AIC_analysis   |
|  - Compute IAC    |
|  - Percentile/classification |
+--------+----------+
         |
         v
+-------------------+
| 04_visualize      |
|  - Grid plots     |
|  - Bar charts     |
+-------------------+
```

---

## Dependencies

The following Python packages are required:

- `numpy`
- `pandas`
- `xarray`
- `requests`
- `matplotlib`
- `glob` (standard library)
- `pathlib` (standard library)

Install via pip:

```bash
pip install numpy pandas xarray requests matplotlib
```

> For reproducibility, using a virtual environment is recommended.

---

## Usage

1. **Download Data**  
   ```bash
   python 01_download_data.py
   ```
   > By default, only SEAS5 datasets are downloaded. Uncomment OBS download in the script if needed.

2. **Extract Regional Data**  
   Open `02_extract_data.ipynb` and run all cells.  
   Adjust `vars_to_process`, spatial bounds (`lat_min`, `lat_max`, `lon_min`, `lon_max`), and lead times as needed.

3. **Compute Anomaly Correlations**  
   Open `03_AIC_analysis.ipynb` and run all cells.  
   - Ensure `SEASS_midwest_T2.csv` and `OBS_midwest_T2.csv` exist in `output/`.

4. **Visualize Results**  
   Open `04_visualize.ipynb` and run all cells.  
   - Grid plots show correlation percentiles per start month.
   - Bar charts highlight the 66th percentile correlation for ensemble members.

---

## Output

- `output/SEASS_midwest_T2.csv` – Regional means for SEAS5 forecasts.  
- `output/OBS_midwest_T2.csv` – Regional means for OBS data.  
- `output/anomaly_corr.csv` – Anomaly correlation results per ensemble member.  
- `output/classified_df.csv` – Ensemble members classified into skill categories.  
- `output/summary_df.csv` – Percentile summary for plotting and analysis.

---

## License

This repository is released under the **MIT License**.
