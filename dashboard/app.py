import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from faicons import icon_svg

from meteostat import Stations, Hourly
from datetime import datetime
from geopy.distance import geodesic

import pandas as pd

# Import data from shared.py
from shared import app_dir, df
from shiny import App, reactive, render, ui

import rasterio
from scipy.ndimage import rotate
import matplotlib.colors as mcolors
import os
import glob

## ----- Temperature information ------

def C_to_F(c):
    return c * 9/5 + 32

# create a time frame, we are looking at april 2025 - june 2025
start = datetime(2025, 2, 1)
end = datetime(2025, 5, 31)

# station IDs 
tvc_id = 'KTVC0'
acb_id = 'KACB0'

# Fetch data from NOAA using meteostat
df_tvc = Hourly(tvc_id, start, end).fetch()
df_acb = Hourly(acb_id, start, end).fetch()

def interpolateTemp(df1, df2, coord1, coord2, lat, lon):
    """
    Interpolate data of two locations. 
    
    Inputs
    ------
    df1, df2: DataFrames with datetime index and 'temp' columns (Celsius)
    coord1, coord2: (lat, lon) tuples of each station
    lat, lon: coordinates of interpolation point
    
    Returns
    -------
    DataFrame for interpolation point.
    """
    # Calculate distances (in km) from each station to the target location
    d1 = geodesic(coord1, (lat, lon)).km
    d2 = geodesic(coord2, (lat, lon)).km
    w1 = 1/d1
    w2 = 1/d2
    w_sum = w1 + w2

    # Align on datetimes present in both series (inner join)
    merged = pd.merge(df1[['temp']], df2[['temp']], left_index=True, right_index=True, suffixes=('_1', '_2'))

    # Weighted temperature at each hour
    merged['temp_interp'] = (merged['temp_1'] * w1 + merged['temp_2'] * w2) / w_sum

    # Index: datetime, columns: temp_interp (°C)
    return merged[['temp_interp']]

# Station coordinates:
coord_tvc = (44.7416, -85.5824)   # TVC
coord_acb = (44.9886, -85.1984)   # ACB

# Shoreline Fruit's orchard:
cherrylat, cherrylon = 44.83444, -85.44333

df_interp = interpolateTemp(
    df_tvc, df_acb,
    coord_tvc, coord_acb,
    cherrylat, cherrylon
)

#chill hour counter, this can be edited to your preferred model
interp_chill = pd.DataFrame(index=df_interp.index)
interp_chill['chill_score'] = np.select(
    [
        (df_interp['temp_interp'] >= 0) & (df_interp['temp_interp'] < 7.22222),
        (df_interp['temp_interp'] >= 7.22222) & (df_interp['temp_interp'] < 15.5556),
        (df_interp['temp_interp'] >= 15.5556)
    ],
    [1, 0, -1],
    default=0
)
interp_chill['chill_accum'] = interp_chill['chill_score'].cumsum()

#GDD counter
interp_chill['temp_interp_F'] = C_to_F(df_interp['temp_interp'])
interp_chill['gdd_hr'] = np.maximum(interp_chill['temp_interp_F'] - 41, 0) / 24

# chill_threshold = 250
# if (interp_chill['chill_accum'] >= chill_threshold).any():
#     first_gdd_time = interp_chill[interp_chill['chill_accum'] >= chill_threshold].index[0]
#     interp_chill['gdd_hr_masked'] = np.where(interp_chill.index >= first_gdd_time, interp_chill['gdd_hr'], 0)
# else:
#     interp_chill['gdd_hr_masked'] = 0.0

# interp_chill['gdd_cum'] = interp_chill['gdd_hr_masked'].cumsum()

df_interp.index = pd.to_datetime(df_interp.index)
df_tvc.index = pd.to_datetime(df_tvc.index)
df_acb.index = pd.to_datetime(df_acb.index)

unique_dates = pd.to_datetime(df_interp.index.date).unique()

def _selected_five_days(idx):
    if idx > len(unique_dates) - 5:
        idx = len(unique_dates) - 5
    days = unique_dates[idx:idx+5]
    # Convert to list of `datetime.date`
    days_list = [d.date() if hasattr(d, 'date') else d for d in days]
    df_days_interp = df_interp[np.isin(df_interp.index.date, days_list)]
    df_days_tvc   = df_tvc[np.isin(df_tvc.index.date, days_list)]
    df_days_acb   = df_acb[np.isin(df_acb.index.date, days_list)]
    return df_days_interp, df_days_tvc, df_days_acb, days

def _dayplot(idx):
    df_days_interp, df_days_tvc, df_days_acb, days = _selected_five_days(idx)
    fig, ax = plt.subplots(figsize=(4,4))
    
    # plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leaves space for legend
    # ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True)
    
    # Plot, using full datetime so the x-axis spans 5*24 points
    if not df_days_interp.empty:
        ax.plot(df_days_interp.index, C_to_F(df_days_interp['temp_interp']), label='Interpolated Temp (Orchard)', color='red', linewidth=1.5)
    if not df_days_tvc.empty:
        ax.plot(df_days_tvc.index, C_to_F(df_days_tvc['temp']), alpha=0.4, label='TVC (Cherry Capital Airport)', color='red', linewidth=1)
    if not df_days_acb.empty:
        ax.plot(df_days_acb.index, C_to_F(df_days_acb['temp']), alpha=0.4, label='ACB (Antrim County Airport)', color='red', linewidth=1)
    ax.set_xlabel("Date/Hour")
    
    x_vals = df_days_interp.index
    day_starts = [i for i, t in enumerate(x_vals) if t.hour == 0]
    ax.set_xticks(x_vals)
    labels = ['' for _ in x_vals]
    for i in day_starts:
        labels[i] = x_vals[i].strftime('%m-%d')
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    
    ax.set_ylabel("Temperature (F)")
    ax.set_title(f"{str(days[0])[:10]} to {str(days[-1])[:10]}")
    ax.axhspan(ymin=C_to_F(7.2222), ymax=C_to_F(15.5556), facecolor='grey', alpha=0.3, label="moderate temp, zero weight")
    ax.axhspan(ymin=C_to_F(15.5556), ymax=C_to_F(35), facecolor='pink', alpha=0.3, label="high temp, negative weight")
    ax.axhspan(ymin=C_to_F(0), ymax=C_to_F(7.22222), facecolor='lightblue', alpha=0.3, label="optimal chill temp, positive weight")
    ax.axhline(C_to_F(0), color='darkblue', alpha=0.4, linestyle=':', linewidth=1.5, label='32°F (Freezing)')
    ax.grid(True)
    ax.xaxis.grid(True, color='lightgray', alpha=0.2)
    ax.yaxis.grid(True, color='lightgray', alpha=0.2)
    # plt.tight_layout()

def compute_gdd_cum(threshold=250):
    if (interp_chill['chill_accum'] >= threshold).any():
        first_gdd_time = interp_chill[interp_chill['chill_accum'] >= threshold].index[0]
        interp_chill['gdd_hr_masked'] = np.where(interp_chill.index >= first_gdd_time, interp_chill['gdd_hr'], 0)
    else:
        interp_chill['gdd_hr_masked'] = 0.0

    gdd_cum = interp_chill['gdd_hr_masked'].cumsum()
    return gdd_cum


def _longterm_plot(threshold=250, idx=0, ideal_gdd=230):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, hspace=0.1)
    ax1 = plt.subplot(gs[0]) #regular temp v time subplot,
    ax2 = plt.subplot(gs[1], sharex=ax1)  #chill hours subplot
    ax3 = plt.subplot(gs[2], sharex=ax1)

    gdd_cum = compute_gdd_cum(threshold=threshold)

    # --------- Highlight 5-day window ---------
    if idx > len(unique_dates) - 5:
        idx = len(unique_dates) - 5
    window_start = pd.to_datetime(unique_dates[idx])
    window_end = pd.to_datetime(unique_dates[idx + 5 - 1]) + pd.Timedelta(hours=23)
    for ax in (ax1, ax2, ax3):
        ax.axvspan(window_start, window_end, color="gold", alpha=0.15, zorder=0)
        
    # --------- Highlight Bloom windows --------
    bloom_windows = _bloom_windows()
    for window_dates in bloom_windows:
        start = pd.to_datetime(window_dates[0])
        end = pd.to_datetime(window_dates[-1]) + pd.Timedelta(hours=23)
        for ax in (ax1, ax2, ax3):
            ax.axvspan(start, end, color="purple", alpha=0.12, zorder=0)

    ax1.plot(df_interp.index, C_to_F(df_interp['temp_interp']), label='Interpolated temp. of orchard)', color='red', linewidth=1 )
    ax1.axhspan(ymin=C_to_F(7.2222), ymax=C_to_F(15.5556), facecolor='grey', alpha=0.3, label="moderate temp, zero weight")
    ax1.axhspan(ymin=C_to_F(15.5556), ymax=C_to_F(35), facecolor='pink', alpha=0.3, label="high temp, negative weight")
    ax1.axhspan(ymin=C_to_F(0), ymax=C_to_F(7.22222), facecolor='lightblue', alpha=0.3, label="optimal chill temp, positive weight")
    ax1.axhline(C_to_F(0), color='darkblue', alpha=0.4, linestyle=':', linewidth=1.5, label='32°F (Freezing)')
    
    ax2.plot(interp_chill.index, interp_chill['chill_accum'], label="Accumulated Chill hours", color="cornflowerblue")
    # ax2.axhline(threshold, color='blue', linestyle='--', label='1200 Chill hours')
    reached = interp_chill[interp_chill['chill_accum'] >= threshold]
    if not reached.empty:
     first_time = reached.index[0]
     first_accum = reached['chill_accum'].iloc[0]
    else:
     first_time = None
    if first_time is not None:
      ax2.plot(first_time, first_accum, 'ro', label=f"First Reached 250 ({first_time.strftime('%Y-%m-%d %H:%M')})")
      ax2.axvline(first_time, color='red', linestyle=':', alpha=0.6)

    ax3.plot(interp_chill.index, gdd_cum, color='olive', label='Accumulated GDD (F>41)', linewidth=2)
    ax3.axhline(ideal_gdd, color='red', linestyle=':', alpha=0.6, label=f'Ideal GDD ({ideal_gdd})')
    
    for ax in (ax1, ax2, ax3):
        plt.setp(ax.get_xticklabels(), fontsize=5, ha='center')
    ax3.set_xlabel("Time")
    ax1.set_ylabel("°F", rotation=0, labelpad=20, fontsize=7, ha='center')
    ax2.set_ylabel("CP", rotation=0, labelpad=15, fontsize=7, ha='center')
    ax3.set_ylabel("GDD", rotation=0, labelpad=15, fontsize=7, ha='center')

def _plot_ebi_planetscope(file="20250311.tif"):
    epsilon = 1e-6
    fdat = rasterio.open(file)
    blue  = fdat.read(1).astype(float) / 10000.0
    green = fdat.read(2).astype(float) / 10000.0
    red   = fdat.read(3).astype(float) / 10000.0
    nir   = fdat.read(4).astype(float) / 10000.0

    # Compute EBI
    ebi = (red + green + blue) / (green / (blue + epsilon) + epsilon) / (red - blue + 256)
    ebi_norm = (ebi - np.nanmin(ebi)) / (np.nanmax(ebi) - np.nanmin(ebi) + epsilon)

    # Compute NDVI for vegetation mask
    ndvi = (nir - red) / (nir + red + epsilon)

    # Vegetation mask
    veg_mask = ndvi > 0.5  # adjust threshold if needed

    # Apply mask: keep EBI where vegetation, set non-veg to 0 (black)
    ebi_masked = np.zeros_like(ebi_norm)
    ebi_masked[veg_mask] = ebi_norm[veg_mask]

    # Rotation correction
    ebi_rotated = rotate(ebi_masked, -1, reshape=False, order=1, mode='constant', cval=0)
    ebi_plot = np.ma.masked_equal(ebi_rotated, 0)

    # Custom cyan -> orange colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("cyan_orange", ["cyan", "red"])

    # Plot with automatic rotation correction
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)  # let matplotlib handle axes placement automatically
    # im = ax.imshow(ebi_plot, cmap=cmap, vmin=0.0, vmax=0.5)
    # cbar = fig.colorbar(im, ax=ax, fraction=0.02, label="EBI (Blossom Intensity)")
    # ax.set_title(f"EBI with Vegetation Mask - {os.path.basename(tif_path)}")
    # ax.axis("off")

    fig = plt.figure(figsize=(8, 8))
    im = plt.imshow(ebi_plot, cmap=cmap, vmin=0.0, vmax=0.5)
    cbar = fig.colorbar(im, fraction=0.02, label="EBI (Effective Bloom Index)")
    # plt.title("Blossom Intensity")
    plt.axis("off")
    total_ebi = np.sum(ebi_norm[veg_mask])
    return total_ebi

def _bloom_windows():
    
    epsilon = 10e-6
    
    bloom_windows = []
    n = len(unique_dates)
    for idx in range(n - 4):
        window = unique_dates[idx:idx+5]
        window_dates = unique_dates[idx:idx+5]
        found_file = None
        for d in window_dates:
            date_str = pd.to_datetime(d).strftime("%Y%m%d")
            candidate = f"./data/planetscope/{date_str}.tif"
            if os.path.exists(candidate):
                found_file = candidate
                break
        if found_file is not None:
            fdat = rasterio.open(found_file)
            blue  = fdat.read(1).astype(float) / 10000.0
            green = fdat.read(2).astype(float) / 10000.0
            red   = fdat.read(3).astype(float) / 10000.0
            nir   = fdat.read(4).astype(float) / 10000.0
            no_of_pixels = blue.size

            # Compute EBI
            ebi = (red + green + blue) / (green / (blue + epsilon) + epsilon) / (red - blue + 256)
            ebi_norm = (ebi - np.nanmin(ebi)) / (np.nanmax(ebi) - np.nanmin(ebi) + epsilon)

            # Compute NDVI for vegetation mask
            ndvi = (nir - red) / (nir + red + epsilon)

            # Vegetation mask
            veg_mask = ndvi > 0.5  # adjust threshold if needed

            # Apply mask: keep EBI where vegetation, set non-veg to 0 (black)
            ebi_masked = np.zeros_like(ebi_norm)
            ebi_masked[veg_mask] = ebi_norm[veg_mask]
            total_ebi = np.sum(ebi_norm[veg_mask])
            avg_ebi = total_ebi / no_of_pixels
            
            if (avg_ebi > 0.25 and avg_ebi < 0.27): bloom_windows.append(window)
            
    intervals = []
    for window_dates in bloom_windows:
        start = pd.to_datetime(window_dates[0])
        end = pd.to_datetime(window_dates[-1]) + pd.Timedelta(hours=23)
        intervals.append((start, end))

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    # Merge overlapping intervals
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(list(interval))
        else:
            last = merged[-1]
            # Calculate the gap in days between last[1] and interval[0]
            gap_days = (interval[0] - last[1]).days
            if interval[0] <= last[1] or gap_days < 7:
                # Overlap or less than 10 days apart: merge
                last[1] = max(last[1], interval[1])
            else:
                merged.append(list(interval))
            
            
    return merged
            
            
def _plot_ebi_sentinel2(date="20240426"):
    from pyproj import Transformer
    from rasterio.windows import Window

    # Some constants for this function
    veg_threshold = 0.5
    # plotting
    window_size = 100 # pixels around orchard (~100x100 pixel window)
    figure_size = 8
    # Latitude/Longitude of orchard (decimal degrees)
    lat, lon = 44.83444, -85.44333 # approximately 44-50-04 N, 85-26-36 W

    # other variables (don't change these)
    resolution = '10' # unfortunately only 10 works because resolutions 20 and 60 don't have B08 vegetation data
    # how many standard deviations away count as an outlier in the data
    outlier_stdevs_away = 40 # the outliers are really far away (> 100 stdevs)

    # folder path for this date
    folder_start = 'data/sentinel2/cherry_'
    folder = f'{folder_start}{date}/R10m/'

    bluepath = folder + 'B02.jp2'
    greenpath = folder + 'B03.jp2'
    redpath = folder + 'B04.jp2'
    vegpath = folder + 'B08.jp2'

    ### Set up window
    # Open one band to get raster CRS
    with rasterio.open(bluepath) as src:
        raster_crs = src.crs
        width = src.width
        height = src.height
    
    # Transform lat/lon to raster CRS
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    # Find pixel row/col in the raster
    row, col = src.index(x, y)
    # Define window bounds around orchard
    window_size = 100
    row_start = max(row - window_size, 0)
    row_end = min(row + window_size, src.height)
    col_start = max(col - window_size, 0)
    col_end = min(col + window_size, src.width)
    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

    ### Read only the window for each band
    with rasterio.open(bluepath) as src:
        initial = src.read(1, window=window)
        diff = float(np.nanmax(initial) - np.nanmin(initial))
        blue = initial - np.nanmin(initial)
        if diff:
            blue = blue / diff
    with rasterio.open(greenpath) as src:
        initial = src.read(1, window=window)
        diff = float(np.nanmax(initial) - np.nanmin(initial))
        green = initial - np.nanmin(initial)
        if diff:
            green = green / diff
    with rasterio.open(redpath) as src:
        initial = src.read(1, window=window)
        diff = float(np.nanmax(initial) - np.nanmin(initial))
        red = initial - np.nanmin(initial)
        if diff:
            red = red / diff
    with rasterio.open(vegpath) as src:
        initial = src.read(1, window=window)
        diff = float(np.nanmax(initial) - np.nanmin(initial))
        nri = initial - np.nanmin(initial)
        if diff:
            nri = nri / diff

    ### EBI
    # avoid division by 0
    epsilon = 1e-6
    # Compute EBI and normalize
    ebi = (red + green + blue) / ((green + epsilon) / (blue + epsilon)) / (red - blue + 256)
    ebi_norm = (ebi - np.nanmin(ebi)) / (np.nanmax(ebi) - np.nanmin(ebi) + epsilon)
    # adjusting for outliers
    outliers = ebi_norm > 40*np.std(ebi_norm)
    ebi_norm_adj = ebi_norm.copy()
    ebi_norm_adj[outliers] = 0

    ### Vegetation mask
    # Compute NDVI for vegetation mask
    ndvi = (nri - red) / (nri + red + epsilon)
    # Vegetation mask
    veg_mask = ndvi > veg_threshold  # adjust threshold if needed
    # Apply mask to EBI
    ebi_veg = ebi_norm_adj.copy()
    ebi_veg[~veg_mask] = 0 # zero out non-vegetation areas
    # renormalize newly masked EBI
    ebi_veg_norm = ebi_veg - np.nanmin(ebi_veg)
    diff = float(np.nanmax(ebi_veg) - np.nanmin(ebi_veg))
    if diff:
        ebi_veg_norm = ebi_veg_norm / diff

    ### Plot
    # Custom cyan -> orange colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("cyan_orange", ["cyan", "red"])
    # Plot figure
    plt.figure(figsize=(8, 8))
    plt.imshow(ebi_veg_norm, cmap=cmap)
    plt.colorbar(label='EBI - vegetation only')
    plt.title(date + ' EBI filtered for vegetation')
    plt.axis('off')
    # plt.show()



## ------- App UI --------


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_slider("date_slider", "Select the start day of 5 day period:",
                min=0, max=len(unique_dates)-5, value=90, step=5, ticks=False),
        ui.input_slider("chill_threshold", "Chill Points Threshold:",
                min=0, max=1000, value=250, step=10, ticks=False),
        ui.input_slider("ideal_gdd", "Ideal GDD Value:",
                min=0, max=800, value=300, step=5, ticks=False),
        ui.input_radio_buttons(
            "satellite",
            "Satellite data",
            choices=["PlanetScope", "Sentinel-2"],
            selected="PlanetScope"
        )#, title="Date Filter",
    ),
    ui.layout_columns(
        # ui.card(
            # ui.card_header("Temperature Detail of 5 day Period"),
            ui.output_plot("dayplot"),
            # full_screen=True,
        # ),
        # ui.card(
            # ui.card_header("Enhanced Bloom Index (EBI)"),
            ui.output_plot("plot_ebi"),
            # full_screen=True,
        # ),
    ),
    ui.layout_columns(
        # ui.card(
            # ui.card_header("Blossoming Season"),
            ui.output_plot("longterm_plot"),
            # full_screen=True,
        # ),
    ),
    ui.layout_column_wrap(
        ui.value_box(
            "Chill Points accumulated",
            ui.output_text("chill_point_count"),
            showcase=icon_svg("snowflake"),
        ),
        ui.value_box(
            "GDD accumulated",
            ui.output_text("gdd_cum"),
            showcase=icon_svg("leaf"),
        ),
        ui.value_box(
            "Total EBI",
            ui.output_text("ebi_total"),
            showcase=icon_svg("clover"),
        ),
        fill=False,
    ),
    ui.include_css(app_dir / "styles.css"),
    title="MI cherry blooming versus temperature",
    fillable=True,
)


def server(input, output, session):
    
    curr_total_ebi = reactive.Value(0.0)
    
    @reactive.calc
    def get_ebi_total():
        return curr_total_ebi.get()
    
    @output
    @render.plot
    def dayplot():
        idx = input.date_slider()
        _dayplot(idx)

    @output
    @render.plot
    def longterm_plot():
     _longterm_plot(input.chill_threshold(), input.date_slider(), input.ideal_gdd())

    @reactive.calc
    def filtered_df():
        filt_df = df[df["species"].isin(input.species())]
        filt_df = filt_df.loc[filt_df["body_mass_g"] < input.mass()]
        return filt_df
    
    def plot_ebi_planetscope():
        idx = input.date_slider()
        
        if idx > len(unique_dates) - 5:
            idx = len(unique_dates)
            
        window_dates = unique_dates[idx:idx+5]
        found_file = None
        for d in window_dates:
            date_str = pd.to_datetime(d).strftime("%Y%m%d")
            candidate = f"./data/planetscope/{date_str}.tif"
            if os.path.exists(candidate):
                found_file = candidate
                break
        if found_file is None:
            plt.text(0, 0.5, "No satellite data found")
            plt.axis('off')
            curr_total_ebi.set(0.0)
            return
    
        total_ebi = _plot_ebi_planetscope(file=found_file)
        curr_total_ebi.set(total_ebi)

    @output
    @render.plot
    def plot_ebi():
        if input.satellite() == "PlanetScope":
            plot_ebi_planetscope()
        if input.satellite() == "Sentinel-2":
            _plot_ebi_sentinel2()

    @output
    @render.plot
    def temperature_detail():
        n = input.time()
        x = np.linspace(0, 2 * np.pi, n//100)
        plt.scatter(x, np.sin(x))
        plt.ylim(-2, 2)

    @render.text
    def chill_point_count():
        # Get idx of 5-day window start
        idx = input.date_slider()
        if idx > len(unique_dates) - 5:
            idx = len(unique_dates) - 5
        # Calculate last date in window
        end_day = unique_dates[idx + 5 - 1]  # Inclusive last day of window

        # Find the last timestamp (hour) within the last day of window
        mask = interp_chill.index.date == end_day.date() if hasattr(end_day, 'date') else end_day
        sub = interp_chill[mask]
        if len(sub) > 0:
            last_chill = sub['chill_accum'].iloc[-1]
            return f"{last_chill:.0f}"
        else:
            return "N/A"

    @render.text
    def gdd_cum():
        gdd_cum_value = compute_gdd_cum(threshold=input.chill_threshold())
        idx = input.date_slider()
        if idx > len(unique_dates) - 5:
            idx = len(unique_dates) - 5
        end_day = unique_dates[idx + 5 - 1]
        mask = interp_chill.index.date == end_day.date() if hasattr(end_day, 'date') else end_day
        sub = interp_chill[mask]
        if len(sub) > 0:
            last_gdd = gdd_cum_value.iloc[-1]
            return f"{last_gdd:.1f}"
        else:
            return "N/A"
        
    @render.text
    def ebi_total():
        return f"{get_ebi_total():.2f}"

    @render.data_frame
    def summary_statistics():
        cols = [
            "species",
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "body_mass_g",
        ]
        return render.DataGrid(filtered_df()[cols], filters=True)


app = App(app_ui, server)
