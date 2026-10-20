import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import moyal

PLOTS_DIR = Path("plots")
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

files = ["4mm.csv", "14mm.csv", "100mm.csv", "140mm.csv"]

for file in files:
    CSV_PATH = f"../{file}"

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

    def fit_gauss(times, bins="auto"):
        # Gauss-Fit (def gauss) an das normierte Histogramm der Schwellenzeiten
        hist, edges = np.histogram(times, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        p0 = [np.mean(times), np.std(times, ddof=1)]
        try:
            popt, _ = curve_fit(gauss, centers, hist, p0=p0)
        except (RuntimeError, ValueError):
            return np.nan, np.nan

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
        fit_mean, fit_sigma = fit_gauss(diff) if len(diff) > 1 else (np.nan, np.nan)
        edges = np.histogram_bin_edges(diff, bins="auto") if len(diff) > 1 else [np.nan, np.nan]
        difference_fits.append({
            "Threshold": threshold,
            "Events": len(diff),
            "Mean_ns": fit_mean,
            "Sigma_ps": fit_sigma * 1000,
            "BinWidth_ns": edges[1] - edges[0],
        })

    difference_fits = pd.DataFrame(difference_fits)
    valid_fits = difference_fits.dropna(subset=["Mean_ns", "Sigma_ps"])

    # Plot Sigma of Difference against Threshold

    fig, ax_sigma = plt.subplots(figsize=(7, 5))

    ax_sigma.plot(valid_fits["Threshold"], valid_fits["Sigma_ps"], "o-", color="steelblue", linewidth=0.0)
    first_event_id = data["EventID"].min()
    scintillator_length = data.loc[
        data["EventID"] == first_event_id, "ScintillatorLenght"
    ].iloc[0]
    ax_sigma.set_title(f"Sigma of Difference against Threshold (Length = {scintillator_length} mm)")
    ax_sigma.set_xlabel("photon threshold N")
    ax_sigma.set_ylabel("$\sigma$ / ps")
    ax_sigma.grid(alpha=0.3)

    stats_text = " N    Mean / ns    Sigma / ps\n"
    stats_text += "-" * 32 + "\n"

    for _, row in valid_fits.iterrows():
        stats_text += (
            f"{int(row['Threshold']):3d}   "
            f"{row['Mean_ns']:8.4f}     "
            f"{row['Sigma_ps']:8.2f}\n"
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

    fig.tight_layout()
    save_plot(f"threshold_difference_gauss_sigma_{scintillator_length}mm")
    # plt.show()

    print(f"Finished plotting threshold difference Gaussian sigma: {scintillator_length} mm")

    difference_fits.round(4).to_string(index=False)