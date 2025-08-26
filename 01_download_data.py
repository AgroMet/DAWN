import os
import requests
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------
BASE_URLS = {
    "OBS": "https://dawn2.umd.edu/OBS/regrid_monthly/",
    "SEAS5": "https://dawn2.umd.edu/GCM/SEAS5/monthly/",
}

YEARS = range(1980, 2026)           # Years to download
MONTHS = range(1, 13)               # Months (for SEAS5)
VARIABLES = ['ASWDNS', 'PRAVG', 'T2MAX', 'T2MIN', 'VPD', 'WD', 'WS']

OUTPUT_DIR = Path("data")           # Base output directory
CHUNK_SIZE = 8192                   # Download chunk size in bytes

# -------------------------------
# Utility functions
# -------------------------------

def build_obs_urls(base_url, prefix, variables, years):
    """
    Construct URLs for OBS datasets.
    Each URL corresponds to a specific variable and year.
    """
    return [
        f"{base_url}{prefix}_{var}_{year}-01-01-00_{year}-12-31-18.nc"
        for var in variables
        for year in years
    ]

def build_seas5_urls(base_url, years, months, start_year=1993):
    """
    Construct URLs for SEAS5 datasets.
    Each URL corresponds to a specific year and start month.
    """
    return [
        f"{base_url}ecmwf_year-{year}_month-{month:02d}_sfc.nc"
        for year in years if year >= start_year
        for month in months
    ]

def download_file(url, output_folder):
    """
    Download a single file from a URL and save it locally.
    Skips download if file already exists.
    """
    filename = url.split("/")[-1]
    filepath = output_folder / filename

    if filepath.exists():
        print(f"Skipped (already exists): {filename}")
        return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write file in chunks
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)
        print(f"Downloaded: {filename}")
    except requests.RequestException as e:
        print(f"Failed: {filename} -> {e}")

def download_dataset(name, base_url, variables=None, years=None, months=None):
    """
    Download all files for a given dataset type (OBS or SEAS5).
    Saves a list of URLs in a text file.
    """
    print(f"\nStarting downloads for {name}...")
    output_folder = OUTPUT_DIR / name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Build dataset-specific URLs
    if name == "OBS":
        urls = build_obs_urls(base_url, "OBS_monthly", variables, years)
    elif name == "SEAS5":
        urls = build_seas5_urls(base_url, years, months)
    else:
        raise ValueError(f"Unknown dataset type: {name}")

    # Save URLs to text file for reference
    with open(output_folder / "urls.txt", "w") as f:
        f.write("\n".join(urls))

    # Download all files
    for url in urls:
        download_file(url, output_folder)

# -------------------------------
# Main script
# -------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Uncomment to download OBS data
    # download_dataset("OBS", BASE_URLS["OBS"], VARIABLES, YEARS)

    # Download SEAS5 dataset
    download_dataset("SEAS5", BASE_URLS["SEAS5"], years=YEARS, months=MONTHS)

# Run the script
if __name__ == "__main__":
    main()
