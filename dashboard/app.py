import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faicons import icon_svg

# Import data from shared.py
from shared import app_dir, df

from shiny import App, reactive, render, ui

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_slider("time", "Date", 2000, 6000, 6000),
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
            ui.output_plot("temperature_detail"),
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
