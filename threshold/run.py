import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import moyal

SCRIPT_DIR = Path(__file__).resolve().parent

PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def save_plot(filename=None):
    figure = plt.gcf()
    if filename is None:
        title = figure.axes[0].get_title() if figure.axes else "plot"
        filename = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            title
        ).strip("_").lower()
        filename = filename or "plot"

    figure.savefig(
        PLOTS_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

files = ["4mm.csv", "14mm.csv", "100mm.csv", "70mm.csv", "140mm.csv", "180mm.csv"]

all_valid_fits = {}

# Je ein Bild für alle Längen, ein Teilplot pro Länge (nach Länge sortiert)
files = sorted(files, key=lambda name: float(name.removesuffix("mm.csv")))

scatter_specs = [
    ("TrackLenght", "track length / mm", "tracklength"),
    ("NSideReflections", "number of side reflections", "side_reflections"),
]
scatter_n_cols = 3
scatter_n_rows = int(np.ceil(len(files) / scatter_n_cols))
scatter_figures = {}

for column, _, _ in scatter_specs:
    scatter_fig, scatter_axes = plt.subplots(
        scatter_n_rows, scatter_n_cols,
        figsize=(5 * scatter_n_cols, 4 * scatter_n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    scatter_figures[column] = (scatter_fig, scatter_axes.ravel())

for file_index, file in enumerate(files):
    CSV_PATH = SCRIPT_DIR.parent / file

    print(f"Loading CSV file: {CSV_PATH}")

    COLUMN_NAMES = [
        "RunID",
        "EventID",
        "TrackID",
        "Hit",
        "GlobalTime",
        "TimeSinceZero",
        "TrackLenght",
        "PhotonEnergy",
        "PhotonFlightTime",
        "PhotonCreationTime",
        "Process",
        "NSideReflections",
        "ScintillatorLenght",
        "EmissionAngle"
    ]

    DTYPES = {
        "RunID": "int32",
        "EventID": "int32",
        "TrackID": "int32",
        "Hit": "category",
        "GlobalTime": "float64",
        "TimeSinceZero": "float64",
        "TrackLenght": "float32",
        "PhotonEnergy": "float32",
        "PhotonFlightTime": "float32",
        "PhotonCreationTime": "float32",
        "Process": "category",
        "NSideReflections": "int32",
        "ScintillatorLenght": "float32",
        "EmissionAngle": "float32",
    }

    data = pd.read_csv(CSV_PATH, comment="#", header=None, names=COLUMN_NAMES, dtype=DTYPES)

    counts_data = len(data)
    counts_events = data['EventID'].nunique()

    print(f"{counts_data:,} Photon-Treffer geladen, {counts_events:,} Events")
    data.head()

    data["WavelengthNM"] = (1239.841984 / data["PhotonEnergy"])

    print("CSV File loaded!")

    # masks
    mask_process_scintillation = data["Process"] == "Scintillation"
    mask_process_cerenkov = data["Process"] == "Cerenkov"

    mask_hit_top = data["Hit"] == "top"
    mask_hit_bottom = data["Hit"] == "bottom"

    # all data
    data_all = data.copy()

    # only Scintillation
    data_all_scintillation = data[
        mask_process_scintillation
    ].copy()

    # only Cerenkov
    data_all_cerenkov = data[
        mask_process_cerenkov
    ].copy()

    # only top data all
    data_all_top = data[
        mask_hit_top
    ].copy()

    # only top data scintillation
    data_all_top_scintillation = data[
        (mask_hit_top)
        &
        (mask_process_scintillation)
    ].copy()

    # only top data cerenkov
    data_all_top_cerenkov = data[
        (mask_hit_top)
        &
        (mask_process_cerenkov)
    ].copy()

    # only bottom data all
    data_all_bottom = data[
        mask_hit_bottom
    ].copy()

    # only bottom data scintillation
    data_all_bottom_scintillation = data[
        (mask_hit_bottom)
        &
        (mask_process_scintillation)
    ].copy()

    # only top data cerenkov
    data_all_bottom_cerenkov = data[
        (mask_hit_bottom)
        &
        (mask_process_cerenkov)
    ].copy()

    # arrival times
    data_arr_times_all = (
        data_all["TimeSinceZero"].dropna().to_numpy()
    )

    data_arr_times_all_scintillation = (
        data_all_scintillation["TimeSinceZero"].dropna().to_numpy()
    )

    data_arr_times_all_cerenkov = (
        data_all_cerenkov["TimeSinceZero"].dropna().to_numpy()
    )

    data_arr_times_all_top = (
        data_all_top["TimeSinceZero"].dropna().to_numpy()
    )

    data_arr_times_all_bottom = (
        data_all_bottom["TimeSinceZero"].dropna().to_numpy()
    )

    data_arr_times_top_scintillation = (
        data_all_top_scintillation["TimeSinceZero"].dropna().to_numpy()
    )

    data_arr_times_top_cerenkov = (
        data_all_top_cerenkov["TimeSinceZero"].dropna().to_numpy()
    )

    data_arr_times_bottom_scintillation = (
        data_all_bottom_scintillation["TimeSinceZero"].dropna().to_numpy()
    )

    data_arr_times_bottom_cerenkov = (
        data_all_bottom_cerenkov["TimeSinceZero"].dropna().to_numpy()
    )

    data_wavelength_all = (
        data_all["WavelengthNM"].dropna().to_numpy()
    )

    data_wavelength_scintillation = (
        data_all_scintillation["WavelengthNM"].dropna().to_numpy()
    )

    data_wavelength_cerenkov = (
        data_all_cerenkov["WavelengthNM"].dropna().to_numpy()
    )

    # creat groups for every event: access via event17 = events[17]
    data_events = {
        event_id: event_data.copy()
        for event_id, event_data in data.groupby("EventID")
    }

    print("Data loaded!")

    cumulative_photon_thresholds = [
        1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 220, 250, 300
    ]

    # Define functions for Gaussian fitting and threshold computation

    def gauss(x, mean, sigma):
        return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mean)**2 / (2 * sigma**2))

    def fit_gauss(times, bins="auto", return_errors=False):
        # Gauss-Fit (def gauss) an das normierte Histogramm der Schwellenzeiten
        hist, edges = np.histogram(times, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        p0 = [np.mean(times), np.std(times, ddof=1)]
        try:
            popt, pcov = curve_fit(gauss, centers, hist, p0=p0)
        except (RuntimeError, ValueError):
            return (np.nan,) * (4 if return_errors else 2)

        if return_errors:
            # Unsichere Kovarianz (inf) wird zu NaN, dann fehlt nur der Fehlerbalken
            perr = np.sqrt(np.diag(pcov))
            perr = np.where(np.isfinite(perr), perr, np.nan)
            return popt[0], abs(popt[1]), perr[0], perr[1]

        return popt[0], abs(popt[1])

    def compute_threshold_times(df, column_name):
        # Zeit, zu der das N-te Photon pro Event eintrifft
        rows = []

        for event_id, event_data in df.groupby("EventID"):
            event_times = event_data["TimeSinceZero"].dropna().to_numpy(dtype=float)
            event_times = np.sort(event_times[event_times >= 0])

            row = {"EventID": event_id, "NPhotons": len(event_times)}

            for threshold in cumulative_photon_thresholds:
                row[f"threshold_{threshold}"] = (
                    event_times[threshold - 1]
                    if len(event_times) >= threshold
                    else np.nan
                )

            rows.append(row)

        return pd.DataFrame(rows)

    def compute_threshold_statistics(times_df, cumulative_photon_thresholds):
        # Berechnung der statistischen Kenngrößen für die Schwellenzeiten
        rows = []

        for threshold in cumulative_photon_thresholds:
            times = times_df[f"threshold_{threshold}"].dropna().to_numpy(dtype=float)

            if len(times) > 1:
                gauss_mean, gauss_sigma = fit_gauss(times)
                mean_time = np.mean(times)
                sigma_time = np.std(times, ddof=1)
            else:
                mean_time = sigma_time = gauss_mean = gauss_sigma = np.nan

            rows.append({
                "Threshold": threshold,
                "Events": len(times),
                "MeanTime_ns": mean_time,
                "Sigma_ps": sigma_time * 1000,
                "GaussianMean_ns": gauss_mean,
                "GaussianSigma_ps": gauss_sigma * 1000,
            })

        return pd.DataFrame(rows)

    def threshold_table(side_data):
        # Schwellenzeiten pro Event, Spalten = Schwelle
        table = compute_threshold_times(side_data, cumulative_photon_thresholds).set_index("EventID")
        table = table.drop(columns="NPhotons")
        table.columns = cumulative_photon_thresholds
        return table

    threshold_top = threshold_table(data_all_top_scintillation)
    threshold_bottom = threshold_table(data_all_bottom_scintillation)

    # Events ohne Treffer auf einer Seite bleiben als NaN erhalten
    threshold_bottom = threshold_bottom.reindex(threshold_top.index.union(threshold_bottom.index))
    threshold_top = threshold_top.reindex(threshold_bottom.index)

    threshold_times = pd.concat(
        {
            "Top": threshold_top,
            "Bottom": threshold_bottom,
            "Difference": threshold_top - threshold_bottom,
        },
        axis=1,
    )

    # Zugriff: threshold_times["Top"][50], threshold_times["Difference"][50], threshold_times.loc[event_id]
    threshold_times.head()

    difference_fits = []

    for threshold in cumulative_photon_thresholds:
        diff = threshold_times["Difference"][threshold].dropna().to_numpy(dtype=float)
        fit_mean, fit_sigma, mean_err, sigma_err = (
            fit_gauss(diff, return_errors=True) if len(diff) > 1 else (np.nan,) * 4
        )
        edges = np.histogram_bin_edges(diff, bins="auto") if len(diff) > 1 else [np.nan, np.nan]
        difference_fits.append({
            "Threshold": threshold,
            "Events": len(diff),
            "Mean_ns": fit_mean,
            "MeanErr_ns": mean_err,
            "Sigma_ps": fit_sigma * 1000,
            "SigmaErr_ps": sigma_err * 1000,
            "BinWidth_ns": edges[1] - edges[0],
        })

    difference_fits = pd.DataFrame(difference_fits)
    valid_fits = difference_fits.dropna(subset=["Mean_ns", "Sigma_ps"])

    # Plot Sigma of Difference against Threshold

    fig, ax_sigma = plt.subplots(figsize=(7, 5))

    ax_sigma.errorbar(
        valid_fits["Threshold"],
        valid_fits["Sigma_ps"],
        yerr=valid_fits["SigmaErr_ps"],
        fmt="o",
        color="steelblue",
        capsize=3,
    )
    first_event_id = data["EventID"].min()
    scintillator_length = data.loc[
        data["EventID"] == first_event_id, "ScintillatorLenght"
    ].iloc[0]
    all_valid_fits[float(scintillator_length)] = valid_fits
    ax_sigma.set_title(f"Sigma of Difference against Threshold (Length = {scintillator_length} mm)")
    ax_sigma.set_xlabel("photon threshold N")
    ax_sigma.set_ylabel("$\sigma$ / ps")
    ax_sigma.grid(alpha=0.3)

    stats_text = " N    Mean / ns    Sigma / ps    Err / ps\n"
    stats_text += "-" * 44 + "\n"

    for _, row in valid_fits.iterrows():
        stats_text += (
            f"{int(row['Threshold']):3d}   "
            f"{row['Mean_ns']:8.4f}     "
            f"{row['Sigma_ps']:8.2f}     "
            f"{row['SigmaErr_ps']:8.2f}\n"
        )

    ax_sigma.text(
        0.02, 0.98,
        stats_text,
        transform=ax_sigma.transAxes,
        fontsize=7,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85
        )
    )

    # Platz oben für die Tabelle: Daten belegen nur die untere Hälfte
    y_low, y_high = ax_sigma.get_ylim()
    ax_sigma.set_ylim(y_low, y_low + (y_high - y_low) / 0.5)

    fig.tight_layout()
    save_plot(f"threshold_difference_gauss_sigma_{scintillator_length}mm")
    # plt.show()

    print(f"Finished plotting threshold difference Gaussian sigma: {scintillator_length} mm")

    for column, x_label, _ in scatter_specs:
        ax = scatter_figures[column][1][file_index]
        ax.scatter(data[column], data["TimeSinceZero"], s=2, alpha=0.1)
        ax.set_title(f"Length = {scintillator_length:g} mm")
        ax.set_xlabel(x_label)
        ax.set_ylabel("time since zero / ns")
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(alpha=0.3)

    difference_fits.round(4).to_string(index=False)

for column, x_label, name in scatter_specs:
    scatter_fig, scatter_axes = scatter_figures[column]

    for unused_ax in scatter_axes[len(files):]:
        unused_ax.set_visible(False)

    scatter_fig.suptitle(f"Arrival time vs {x_label.split(' /')[0]}")
    scatter_fig.tight_layout()
    plt.figure(scatter_fig.number)
    save_plot(f"time_vs_{name}_all_lengths")
    plt.close(scatter_fig)

# Feste Farbe und Marker pro Länge, damit sie in beiden Plots gleich sind
length_styles = {
    length: (plt.cm.tab10(i), "osv^Dp"[i % 6])
    for i, length in enumerate(sorted(all_valid_fits))
}

def plot_lengths(lengths, title, filename):
    fig, ax = plt.subplots(figsize=(9, 6))

    for length in lengths:
        color, marker = length_styles[length]
        fits = all_valid_fits[length]
        ax.errorbar(
            fits["Threshold"],
            fits["Sigma_ps"],
            yerr=fits["SigmaErr_ps"],
            fmt=marker,
            color=color,
            capsize=3,
            label=f"{length:g} mm",
        )

    ax.set_title(title)
    ax.set_xlabel("photon threshold N")
    ax.set_ylabel(r"$\sigma$ / ps")
    ax.grid(alpha=0.3)
    ax.legend(title="Scintillator length")

    fig.tight_layout()
    save_plot(filename)
    # plt.show()

# Alle Längen in einem Plot
plot_lengths(
    sorted(all_valid_fits),
    "Sigma of Difference against Threshold (all lengths)",
    "threshold_difference_gauss_sigma_all_lengths",
)

# Auswahl an Längen
selected_lengths = [14.0, 70.0, 140.0, 180.0]
plot_lengths(
    [length for length in selected_lengths if length in all_valid_fits],
    "Sigma of Difference against Threshold (selected lengths)",
    "threshold_difference_gauss_sigma_selected_lengths",
)