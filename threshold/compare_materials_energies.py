import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/derksen/Desktop/geant4/Scintillation_Simulation_build/data")

# Daten liegen unter data/<Material>/<Energie>/<Länge>mm.csv
OUT_DIR = SCRIPT_DIR / "plots" / "plots_comparison"

SIPM_ACTIVE_AREA = "3.72 x 3.62 mm"
PDE_APPLIED = True

CUMULATIVE_PHOTON_THRESHOLDS = [1, 5] + list(range(10, 141, 10))

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
    "EmissionAngle",
    "Material",
    "ElectronEnergy",
]

# Nur diese Spalten werden gelesen, das spart Speicher
USECOLS = ["EventID", "Hit", "TimeSinceZero", "Process", "Material", "ElectronEnergy"]

DTYPES = {
    "EventID": "int32",
    "Hit": "category",
    "TimeSinceZero": "float64",
    "Process": "category",
    "Material": "category",
    "ElectronEnergy": "float32",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "legend.frameon": False,
    "errorbar.capsize": 3,
})


def format_energy(energy_mev):
    return f"{energy_mev / 1000:g} GeV" if energy_mev >= 1000 else f"{energy_mev:g} MeV"


def gauss(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def fit_gauss(times):
    # Gauss-Fit (def gauss) an das normierte Histogramm, Rückgabe: mean, sigma, mean_err, sigma_err
    hist, edges = np.histogram(times, bins="auto", density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    p0 = [np.mean(times), np.std(times, ddof=1)]
    try:
        popt, pcov = curve_fit(gauss, centers, hist, p0=p0)
    except (RuntimeError, ValueError):
        return (np.nan,) * 4

    # Unsichere Kovarianz (inf) wird zu NaN, dann fehlt nur der Fehlerbalken
    perr = np.sqrt(np.diag(pcov))
    perr = np.where(np.isfinite(perr), perr, np.nan)
    return popt[0], abs(popt[1]), perr[0], perr[1]


def threshold_times(df):
    # Zeit des N-ten Photons pro Event, Zeilen = EventID, Spalten = Schwelle
    times = df.loc[df["TimeSinceZero"] >= 0, ["EventID", "TimeSinceZero"]]
    times = times.sort_values(["EventID", "TimeSinceZero"])
    times["rank"] = times.groupby("EventID").cumcount() + 1
    times = times[times["rank"].isin(CUMULATIVE_PHOTON_THRESHOLDS)]
    table = times.pivot(index="EventID", columns="rank", values="TimeSinceZero")
    return table.reindex(columns=CUMULATIVE_PHOTON_THRESHOLDS)


def analyse_file(path):
    data = pd.read_csv(path, comment="#", header=None, names=COLUMN_NAMES, usecols=USECOLS, dtype=DTYPES)

    scintillation = data[data["Process"] == "Scintillation"]
    top = threshold_times(scintillation[scintillation["Hit"] == "top"])
    bottom = threshold_times(scintillation[scintillation["Hit"] == "bottom"])

    # Events ohne Treffer auf einer Seite ergeben NaN
    difference = top - bottom

    rows = []
    for threshold in CUMULATIVE_PHOTON_THRESHOLDS:
        diff = difference[threshold].dropna().to_numpy(dtype=float)
        _, sigma, _, sigma_err = fit_gauss(diff) if len(diff) > 1 else (np.nan,) * 4
        rows.append({
            "Threshold": threshold,
            "Events": len(diff),
            "Sigma_ps": sigma * 1000,
            "SigmaErr_ps": sigma_err * 1000,
        })

    return {
        "material": str(data["Material"].iloc[0]),
        "energy_mev": float(data["ElectronEnergy"].iloc[0]),
        "n_events": int(data["EventID"].nunique()),
        "fits": pd.DataFrame(rows),
    }


def plot_title(what, length, energy_mev=None, material=None):
    # Erste Zeile: Inhalt, zweite Zeile: Setup
    setup = [f"L = {length:g} mm"]
    if material is not None:
        setup.append(f"Scintillator: {material}")
    if energy_mev is not None:
        setup.append(f"E = {format_energy(energy_mev)}")
    setup += [f"PDE: {PDE_APPLIED}", f"SiPM active area: {SIPM_ACTIVE_AREA}"]
    return f"{what}\n" + ", ".join(setup)


def plot_curves(curves, title, path):
    # curves: Liste aus (Label, fits, Farbe, Marker)
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, fits, color, marker in curves:
        valid = fits.dropna(subset=["Sigma_ps"])
        ax.errorbar(
            valid["Threshold"],
            valid["Sigma_ps"],
            yerr=valid["SigmaErr_ps"],
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
        )

    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("photon threshold N")
    ax.set_ylabel(r"$\sigma$ / ps")
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


results = {}

for path in sorted(DATA_ROOT.glob("*/*/*mm.csv")):
    length = float(path.name.removesuffix("mm.csv"))
    print(f"Loading CSV file: {path}")
    result = analyse_file(path)
    results[(length, result["material"], result["energy_mev"])] = result

if not results:
    raise SystemExit(f"No CSV files found under {DATA_ROOT}")

lengths = sorted({key[0] for key in results})
materials = sorted({key[1] for key in results})
energies = sorted({key[2] for key in results})

# Feste Farbe pro Material bzw. Energie und fester Marker pro Position
material_colors = {name: plt.cm.tab10(i) for i, name in enumerate(materials)}
energy_colors = {energy: plt.cm.tab10(i) for i, energy in enumerate(energies)}
markers = "osv^Dp"

for length in lengths:
    length_dir = OUT_DIR / f"{length:g}mm"

    # Länge und Energie fest, Material variiert
    for energy in energies:
        curves = [
            (
                f"{material} ({results[(length, material, energy)]['n_events']} events)",
                results[(length, material, energy)]["fits"],
                material_colors[material],
                markers[i % len(markers)],
            )
            for i, material in enumerate(materials)
            if (length, material, energy) in results
        ]
        if len(curves) < 2:
            continue

        plot_curves(
            curves,
            plot_title(
                r"Time resolution $\sigma(t_{top} - t_{bottom})$ vs photon threshold, different materials",
                length,
                energy_mev=energy,
            ),
            length_dir / f"materials_{format_energy(energy).replace(' ', '')}.png",
        )

    # Länge und Material fest, Energie variiert
    for material in materials:
        curves = [
            (
                f"{format_energy(energy)} ({results[(length, material, energy)]['n_events']} events)",
                results[(length, material, energy)]["fits"],
                energy_colors[energy],
                markers[i % len(markers)],
            )
            for i, energy in enumerate(energies)
            if (length, material, energy) in results
        ]
        if len(curves) < 2:
            continue

        plot_curves(
            curves,
            plot_title(
                r"Time resolution $\sigma(t_{top} - t_{bottom})$ vs photon threshold, different energies",
                length,
                material=material,
            ),
            length_dir / f"energies_{material}.png",
        )
