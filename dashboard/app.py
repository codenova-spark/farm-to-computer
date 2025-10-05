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
cherrylat, cherrylon = 44.8324, -85.442

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

chill_threshold = 250
if (interp_chill['chill_accum'] >= chill_threshold).any():
    first_gdd_time = interp_chill[interp_chill['chill_accum'] >= chill_threshold].index[0]
    interp_chill['gdd_hr_masked'] = np.where(interp_chill.index >= first_gdd_time, interp_chill['gdd_hr'], 0)
else:
    interp_chill['gdd_hr_masked'] = 0.0

interp_chill['gdd_cum'] = interp_chill['gdd_hr_masked'].cumsum()

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
    ax.set_title(f"Temperature (F), {str(days[0])[:10]} to {str(days[-1])[:10]}")
    ax.axhspan(ymin=C_to_F(7.2222), ymax=C_to_F(15.5556), facecolor='grey', alpha=0.3, label="moderate temp, zero weight")
    ax.axhspan(ymin=C_to_F(15.5556), ymax=C_to_F(35), facecolor='pink', alpha=0.3, label="high temp, negative weight")
    ax.axhspan(ymin=C_to_F(0), ymax=C_to_F(7.22222), facecolor='lightblue', alpha=0.3, label="optimal chill temp, positive weight")
    ax.axhline(C_to_F(0), color='darkblue', alpha=0.4, linestyle=':', linewidth=1.5, label='32°F (Freezing)')
    ax.grid(True)
    ax.xaxis.grid(True, color='lightgray', alpha=0.2)
    ax.yaxis.grid(True, color='lightgray', alpha=0.2)
    # plt.tight_layout()

def _longterm_plot(threshold=250, idx=0):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, hspace=0.1)
    ax1 = plt.subplot(gs[0]) #regular temp v time subplot,
    ax2 = plt.subplot(gs[1], sharex=ax1)  #chill hours subplot
    ax3 = plt.subplot(gs[2], sharex=ax1)

    # --------- Highlight 5-day window ---------
    if idx > len(unique_dates) - 5:
        idx = len(unique_dates) - 5
    window_start = pd.to_datetime(unique_dates[idx])
    window_end = pd.to_datetime(unique_dates[idx + 5 - 1]) + pd.Timedelta(hours=23)
    for ax in (ax1, ax2, ax3):
        ax.axvspan(window_start, window_end, color="gold", alpha=0.15, zorder=0)

    ax1.plot(df_interp.index, C_to_F(df_interp['temp_interp']), label='Interpolated temp. of orchard)', color='red', linewidth=1.5 )
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

    ax3.plot(interp_chill.index, interp_chill['gdd_cum'], color='olive', label='Accumulated GDD (F>41)', linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temp (F)")
    ax2.set_ylabel("Chill Points (CP)")
    ax3.set_ylabel("GDD")




## ------- App UI --------


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_slider("date_slider", "Select the start day of 5 day period:",
                min=0, max=len(unique_dates)-5, value=0, step=5, ticks=False),
        ui.input_slider("chill_threshold", "Chill Points Threshold:",
                min=0, max=1000, value=250, step=10, ticks=False),
        ui.input_radio_buttons(
            "satellite",
            "Satellite data",
            choices=["PlanetScope", "Sentinel-2"],
            selected="PlanetScope"
        )#, title="Date Filter",
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Temperature Detail of 5 day Period"),
            ui.output_plot("dayplot"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Enhanced Bloom Index (EBI)"),
            ui.output_plot("test_plot_run"),
            full_screen=True,
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Blossoming Season"),
            ui.output_plot("longterm_plot"),
            full_screen=True,
        ),
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
    @output
    @render.plot
    def dayplot():
        idx = input.date_slider()
        _dayplot(idx)

    @output
    @render.plot
    def longterm_plot():
        _longterm_plot(input.chill_threshold(), input.date_slider())

    @reactive.calc
    def filtered_df():
        filt_df = df[df["species"].isin(input.species())]
        filt_df = filt_df.loc[filt_df["body_mass_g"] < input.mass()]
        return filt_df
    
    @output
    @render.plot
    def test_plot_run(file="20250311.tif"):
    # def test_plot_run(file="../data/planetscope/20250311.tif"):
        # from ebi_plot import test_plot  # Import the test_plot function
        # x = np.linspace(0, 2 * np.pi, 100)
        # s = input.satellite()
        # if s == "Sentinel-2":
        #     test_plot(x, np.cos(x), label="Sentinel-2")
        # elif s == "PlanetScope":
        #     test_plot(x, np.sin(x), label="PlanetScope")
        # plt.ylim(-2, 2)
        # plt.title("Selected Functions")
        # plt.legend()
        # from ebilib import plot_ebi_planetscope  # Import the test_plot function
        # plot_ebi_planetscope(app_dir / "../data/planetscope" / file)  # Call the function to plot EBI
        epsilon = 1e-6
        fdat = rasterio.open(app_dir / "../data/planetscope" / file)
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
        cbar = fig.colorbar(im, fraction=0.02, label="EBI (Blossom Intensity)")
        plt.title(f"EBI with Vegetation Mask - {os.path.basename(file)}")
        plt.axis("off")


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
        idx = input.date_slider()
        if idx > len(unique_dates) - 5:
            idx = len(unique_dates) - 5
        end_day = unique_dates[idx + 5 - 1]
        mask = interp_chill.index.date == end_day.date() if hasattr(end_day, 'date') else end_day
        sub = interp_chill[mask]
        if len(sub) > 0:
            last_gdd = sub['gdd_cum'].iloc[-1]
            return f"{last_gdd:.1f}"
        else:
            return "N/A"
        
    @render.text
    def ebi_total():
        return f"{filtered_df()['ebi_total'].mean():.1f} mm"

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
