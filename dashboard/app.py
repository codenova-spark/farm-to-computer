import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from faicons import icon_svg

from meteostat import Stations, Hourly
from datetime import datetime
from geopy.distance import geodesic

import pandas as pd

# Import data from shared.py
from shared import app_dir, df
from shiny import App, reactive, render, ui

## ----- Temperature information ------

# create a time frame, we are looking at april 2025 - june 2025
start = datetime(2025, 4, 1)
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


# Example data (replace with your actual DataFrames)

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
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leaves space for legend
    # ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True)
    
    # Plot, using full datetime so the x-axis spans 5*24 points
    if not df_days_interp.empty:
        ax.plot(df_days_interp.index, df_days_interp['temp_interp'], label='Interpolated Temp (Orchard)', color='brown', linewidth=2)
    if not df_days_tvc.empty:
        ax.plot(df_days_tvc.index, df_days_tvc['temp'], alpha=0.4, label='TVC (Cherry Capital Airport)', color='brown', linewidth=1)
    if not df_days_acb.empty:
        ax.plot(df_days_acb.index, df_days_acb['temp'], alpha=0.4, label='ACB (Antrim County Airport)', color='brown', linewidth=1)
    ax.set_xlabel("Date/Hour")
    
    x_vals = df_days_interp.index
    day_starts = [i for i, t in enumerate(x_vals) if t.hour == 0]
    ax.set_xticks(x_vals)
    labels = ['' for _ in x_vals]
    for i in day_starts:
        labels[i] = x_vals[i].strftime('%m-%d')
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    
    ax.set_ylabel("Temperature (C)")
    ax.set_title(f"Temperature (C), {str(days[0])[:10]} to {str(days[-1])[:10]}")
    ax.axhspan(ymin=7.2222, ymax=15.5556, facecolor='orchid', alpha=0.3, label="moderate temp, zero weight")
    ax.axhspan(ymin=15.5556, ymax=35, facecolor='pink', alpha=0.3, label="high temp, negative weight")
    ax.axhspan(ymin=0, ymax=7.22222, facecolor='lightblue', alpha=0.3, label="optimal chill temp, positive weight")
    ax.axhline(0, color='blue', linestyle='--', linewidth=1.5, label='0°C (Freezing)')
    ax.grid(True)
    ax.xaxis.grid(True, color='lightgray', alpha=0.2)
    ax.yaxis.grid(True, color='lightgray', alpha=0.2)
    plt.tight_layout()

## ------- App UI --------


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_slider("date_slider", "Start Day:",
                min=0, max=len(unique_dates)-5, value=0, step=1, ticks=False),
        ui.input_radio_buttons(
            "satellite",
            "Satellite data",
            choices=["PlanetScope", "Sentinel-2"],
            selected="PlanetScope"
        )#, title="Date Filter",
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Temperature Detail"),
            ui.output_plot("dayplot"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Enhanced Bloom Index (EBI)"),
            ui.output_plot("test_plot_run"),
            full_screen=True,
        ),
    ),
    ui.layout_column_wrap(
        ui.value_box(
            "Number of penguins",
            ui.output_text("count"),
            showcase=icon_svg("earlybirds"),
        ),
        ui.value_box(
            "Average bill length",
            ui.output_text("bill_length"),
            showcase=icon_svg("ruler-horizontal"),
        ),
        ui.value_box(
            "Average bill depth",
            ui.output_text("bill_depth"),
            showcase=icon_svg("ruler-vertical"),
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
    
    @reactive.calc
    def filtered_df():
        filt_df = df[df["species"].isin(input.species())]
        filt_df = filt_df.loc[filt_df["body_mass_g"] < input.mass()]
        return filt_df
    
    @output
    @render.plot
    def test_plot_run():
        from test_plot import test_plot  # Import the test_plot function
        x = np.linspace(0, 2 * np.pi, 100)
        s = input.satellite()
        if s == "Sentinel-2":
            test_plot(x, np.cos(x), label="Sentinel-2")
        elif s == "PlanetScope":
            test_plot(x, np.sin(x), label="PlanetScope")
        plt.ylim(-2, 2)
        plt.title("Selected Functions")
        plt.legend()

    @output
    @render.plot
    def temperature_detail():
        n = input.time()
        x = np.linspace(0, 2 * np.pi, n//100)
        plt.scatter(x, np.sin(x))
        plt.ylim(-2, 2)

    @render.text
    def count():
        return filtered_df().shape[0]

    @render.text
    def bill_length():
        return f"{filtered_df()['bill_length_mm'].mean():.1f} mm"

    @render.text
    def bill_depth():
        return f"{filtered_df()['bill_depth_mm'].mean():.1f} mm"

    @render.plot
    def length_depth():
        return sns.scatterplot(
            data=filtered_df(),
            x="bill_length_mm",
            y="bill_depth_mm",
            hue="species",
        )

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
