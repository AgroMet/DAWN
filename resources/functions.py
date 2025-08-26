import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches


# --------------------------
# User-defined parameters
# --------------------------
vars_to_process = ["T2MAX", "T2MIN"]       # Variables of interest
lat_min, lat_max = 36, 49                  # Latitude bounds for Midwest
lon_min, lon_max = -105, -84               # Longitude bounds for Midwest
lead_times = range(0, 6)                   # Lead times: 0 to 5 months
weighted = False                            # Whether to apply cosine-latitude weighting

def add_year_month(df, time_col='time'):
    dt = pd.to_datetime(df[time_col])
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    return df

def compute_anomaly_corr(seass, obs, vars_to_process=["T2MAX", "T2MIN"]):
    """
    Compute anomaly correlations between forecast and observed data for specified variables.

    Parameters
    ----------
    seass : pd.DataFrame
        Forecast dataset with columns ['start_month', 'lead_time', 'number', 'year', 'month', vars_to_process...]
    obs : pd.DataFrame
        Observational dataset with columns ['year', 'month', vars_to_process...]
    vars_to_process : list of str
        Variables for which to compute anomaly correlations (default: ['T2MAX','T2MIN'])

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['start_month','lead_time','ensemble','variable','correlation']
    """
    results = []

    # Loop over all forecast start months
    for sm in sorted(seass['start_month'].unique()):
        df_sm = seass[seass['start_month'] == sm]

        # Loop over all forecast lead times
        for lt in sorted(df_sm['lead_time'].unique()):
            df_lt = df_sm[df_sm['lead_time'] == lt]

            # Loop over all ensemble members
            for ens in sorted(df_lt['number'].unique()):
                df_ens = df_lt[df_lt['number'] == ens]

                # Loop over each variable of interest
                for var in vars_to_process:

                    # Identify overlapping years between forecast and observations
                    common_years = np.intersect1d(df_ens['year'].unique(), obs['year'].unique())

                    # TODO For this proof-of-concept, force the analysis to last 5 years
                    common_years = [2020, 2021, 2022, 2023, 2024]
                    if len(common_years) == 0:
                        continue

                    # Subset forecast and observation data to the selected years
                    fcst = df_ens[df_ens['year'].isin(common_years)][['year','month',var]].copy()
                    obsv = obs[obs['year'].isin(common_years)][['year','month',var]].copy()

                    # Compute monthly climatology for each variable
                    clim_fcst = fcst.groupby('month')[var].transform('mean')
                    clim_obs  = obsv.groupby('month')[var].transform('mean')

                    # Calculate anomalies by subtracting monthly climatology
                    fcst['anom'] = fcst[var] - clim_fcst
                    obsv['anom'] = obsv[var] - clim_obs

                    # Align forecast and observation anomalies by year and month
                    merged = pd.merge(fcst, obsv, on=['year','month'], suffixes=('_fcst','_obs'))

                    # Compute correlation of anomalies across years
                    if len(merged) > 1:
                        corr = merged['anom_fcst'].corr(merged['anom_obs'])
                    else:
                        corr = np.nan

                    # Store result for this combination of start_month, lead_time, ensemble, and variable
                    results.append({
                        'start_month': sm,
                        'lead_time': lt,
                        'ensemble': ens,
                        'variable': var,
                        'correlation': corr
                    })

    # Return all results as a DataFrame
    return pd.DataFrame(results)

def summarize_with_brackets(df, percentiles=[33, 50, 66]):
    """
    Summarize ensemble correlations by percentiles and classify ensemble member skill.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns:
        ['start_month', 'lead_time', 'ensemble', 'variable', 'correlation']
    percentiles : list of int
        Percentiles to compute across ensemble members.
        Must include 33 and 66 for skill classification.

    Returns
    -------
    summary_df : pd.DataFrame
        Percentile summary for each (start_month, lead_time, variable) group.
        Columns: ['start_month','lead_time','variable','p33','p50','p66', ...]
    classified_df : pd.DataFrame
        Original dataframe with added percentile columns and a 'category' column
        labeling member skill as 'problematic', 'acceptable', or 'good'.
    """

    # ---- Step 1: Compute percentiles across ensemble members ----
    # Group by start month, lead time, and variable
    grouped = df.groupby(['start_month','lead_time','variable'])

    # Compute requested percentiles (e.g., 33rd, 50th, 66th) for each group
    pct = grouped['correlation'].quantile([p/100 for p in percentiles]).unstack()

    # Rename columns to indicate percentile (e.g., p33, p50, p66)
    pct.columns = [f"p{p}" for p in percentiles]

    # Convert MultiIndex to columns for easy merging later
    summary_df = pct.reset_index()

    # ---- Step 2: Merge percentiles back into the original dataframe ----
    # This ensures every ensemble member has its group's percentile values
    classified_df = df.merge(summary_df, on=['start_month','lead_time','variable'], how='left')

    # ---- Step 3: Classify each ensemble member based on percentiles ----
    def classify(row):
        """
        Assign skill category based on correlation relative to 33rd and 66th percentiles.
        """
        if row['correlation'] < row['p33']:
            return "problematic"
        elif row['correlation'] < row['p66']:
            return "acceptable"
        else:
            return "good"

    # Apply classification row-wise
    classified_df['category'] = classified_df.apply(classify, axis=1)

    # ---- Step 4: Return results ----
    return summary_df, classified_df

def compute_anomaly_corr(seass, obs, vars_to_process=["T2MAX", "T2MIN"]):
    results = []

    # Loop over forecast start months
    for sm in sorted(seass['start_month'].unique()):
        df_sm = seass[seass['start_month'] == sm]

        # Loop over lead times
        for lt in sorted(df_sm['lead_time'].unique()):
            df_lt = df_sm[df_sm['lead_time'] == lt]

            # Loop over ensemble members
            for ens in sorted(df_lt['number'].unique()):
                df_ens = df_lt[df_lt['number'] == ens]

                for var in vars_to_process:
                    # Get overlapping years
                    common_years = np.intersect1d(df_ens['year'].unique(), obs['year'].unique())

                    # (You forced to last 5 years, keep this for now)
                    common_years = [2020, 2021, 2022, 2023, 2024]
                    if len(common_years) == 0:
                        continue

                    fcst = df_ens[df_ens['year'].isin(common_years)].copy()
                    obsv = obs[obs['year'].isin(common_years)].copy()

                    # Keep only year, month, var
                    fcst = fcst[['year','month',var]].copy()
                    obsv = obsv[['year','month',var]].copy()

                    # Compute climatology per month
                    clim_fcst = fcst.groupby('month')[var].transform('mean')
                    clim_obs  = obsv.groupby('month')[var].transform('mean')

                    # Subtract climatology → anomalies
                    fcst['anom'] = fcst[var] - clim_fcst
                    obsv['anom'] = obsv[var] - clim_obs

                    # Merge on year,month to align anomalies
                    merged = pd.merge(fcst, obsv, on=['year','month'], suffixes=('_fcst','_obs'))

                    # Compute correlation across years
                    if len(merged) > 1:
                        corr = merged['anom_fcst'].corr(merged['anom_obs'])
                    else:
                        corr = np.nan

                    results.append({
                        'start_month': sm,
                        'lead_time': lt,
                        'ensemble': ens,
                        'variable': var,
                        'correlation': corr
                    })

    return pd.DataFrame(results)

def plot_percentile_summary(summary_df, variables=["T2MAX", "T2MIN"]):
    """
    Make a 3×3 grid of subplots (Jan–Sep), plotting percentiles vs. lead time.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ['start_month','lead_time','variable','p33','p50','p66']
    variables : list
        Which variables to plot separately.
    """
    # Map start_month numbers to short names
    month_map = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May",
                 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep"}
    
    for var in variables:
        df_var = summary_df[summary_df['variable'] == var]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for i, sm in enumerate(sorted(df_var['start_month'].unique())):
            ax = axes[i]
            df_sm = df_var[df_var['start_month'] == sm]
            
            # Shift x-axis labels from 0–5 to 1–6
            lead_times = df_sm['lead_time'] + 1
            
            ax.plot(lead_times, df_sm['p33'], label="33rd", color="red", linestyle="--")
            ax.plot(lead_times, df_sm['p50'], label="50th", color="black", linestyle="-")
            ax.plot(lead_times, df_sm['p66'], label="66th", color="blue", linestyle="--")
            
            ax.set_title(f"{month_map.get(sm, sm)} start", fontsize=18)  # increased title fontsize
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=14)  # increase tick labels font
        
        # Common labels
        fig.suptitle(f"Interannual Anomaly Correlations ({var})", fontsize=24)
        fig.supxlabel("Lead time (months)", fontsize=18)
        fig.supylabel("IAC (correlation)", fontsize=18)
        
        # Put legend only once (top center)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, 
            loc="upper center", 
            ncol=3, 
            frameon=False, 
            bbox_to_anchor=(0.5, 1.02),
            fontsize=14
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

def plot_percentile_bars(summary_df, variables=["T2MAX", "T2MIN"]):
    """
    Plot the 66th percentile (p66) as grouped bar charts vs lead time.
    Colors indicate start month.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ['start_month','lead_time','variable','p66']
    variables : list
        Which variables to plot separately.
    """
    month_map = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May",
                 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep"}
    
    cmap = plt.cm.get_cmap("tab20", 9)
    
    for var in variables:
        df_var = summary_df[summary_df['variable'] == var]
        
        months = sorted(df_var['start_month'].unique())
        n_months = len(months)
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        lead_times = sorted(df_var['lead_time'].unique() + 1)  # shift 0–5 → 1–6
        x = np.arange(len(lead_times))
        width = 0.8 / n_months
        
        for i, sm in enumerate(months):
            df_sm = df_var[df_var['start_month'] == sm].sort_values("lead_time")
            values = df_sm['p66'].values
            
            ax.bar(
                x + i*width - (n_months-1)*width/2,
                values,
                width=width,
                label=month_map.get(sm, sm),
                color=cmap(i)
            )
        
        ax.set_title(f"66th Percentile Anomaly Correlations ({var})", fontsize=22)
        ax.set_xlabel("Lead time (months)", fontsize=18)
        ax.set_ylabel("IAC (correlation)", fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(lead_times, fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(True, axis="y", alpha=0.3)
        
        # Fixed y-axis range
        ax.set_ylim(-0.3, 0.65)
        
        ax.legend(
            loc="center left", 
            bbox_to_anchor=(1, 0.5), 
            fontsize=13,
            title="Start month"
        )
        
        plt.tight_layout()
        plt.show()

def plot_region_mask(subset, ds, var_name):
    """
    Plot the spatial mask of a selected region with state boundaries 
    based on the subsetted data itself. Lat/Lon are taken from the original dataset.
    
    Parameters
    ----------
    subset : xarray.DataArray
        Subset of the variable restricted to the region.
    ds : xarray.Dataset
        Original dataset containing 'lat' and 'lon'.
    var_name : str
        Variable name (for title/legend).
    """

    # Access lat/lon from original dataset
    lat = ds['lat'].isel(time=0)
    lon = ds['lon'].isel(time=0)

    # Mask where subset has valid (non-NaN) values
    mask2d = ~np.isnan(subset.isel(time=0))

    # --- Plot ---
    plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    plt.scatter(
        lon.values[mask2d], lat.values[mask2d],
        s=5, c="red", transform=ccrs.PlateCarree(),
        label=f"Selected Region ({var_name})"
    )

    # Add state/country boundaries
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="black")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"))
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))

    # Auto bounding box from subset coordinates
    lat_min, lat_max = float(lat.values[mask2d].min()), float(lat.values[mask2d].max())
    lon_min, lon_max = float(lon.values[mask2d].min()), float(lon.values[mask2d].max())
    rect = mpatches.Rectangle(
        (lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
        linewidth=2, edgecolor='blue', facecolor='none', transform=ccrs.PlateCarree(),
        label="Bounding Box"
    )
    ax.add_patch(rect)

    # Zoom around region with small padding
    ax.set_extent([lon_min-2, lon_max+2, lat_min-2, lat_max+2])

    plt.legend()
    plt.title(f"Region Mask Check for {var_name}")
    plt.show()
