from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from materials import material_label

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/derksen/Desktop/geant4/Scintillation_Simulation_build/data")

# Daten liegen unter data/<Material>/<Energie>/<Länge>mm.csv
OUT_DIR = SCRIPT_DIR / "plots" / "plots_comparison"

MATERIAL_FOLDERS = ["EJ228", "EJ230", "EJ232", "EJ232Q05"] # BC422

SIPM_ACTIVE_AREA = "3 x (3.72 x 3.62 mm)"
PDE_APPLIED = True

ABSOLUTE_PHOTON_THRESHOLDS = [1, 5] + list(range(10, 141, 10))
CUMULATIVE_PHOTON_THRESHOLDS = [1, 5] + list(range(10, 301, 10))

# Bereich der x-Achse (photon threshold N) in den Plots
THRESHOLD_PLOT_RANGE = (0, 125)

# Bereich der x- und y-Achse (relative brightness / %) in den Plots
RELATIVE_PLOT_RANGE = {
    "energies": {
        "EJ228": {"x": (0, 30), "y": (30, 270)},
        "EJ230": {"x": (0, 11), "y": (25, 75)},
        "EJ232": {"x": (0, 25), "y": (20, 210)},
        "EJ232Q05": {"x": (0, 30), "y": (20, 200)},
    },
    "materials": {
        "1GeV": {"x": (0, 26), "y": (20, 200)},
        "1MeV": {"x": (0, 30), "y": (20, 250)},
        "2MeV": {"x": (0, 30), "y": (20, 250)},
    },
}

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

    detected = scintillation[scintillation["Hit"].isin(["top", "bottom"])]
    photons_per_event = detected.groupby("EventID").size()
    total_detected = photons_per_event.sum()
    # Mittel über Events mit mindestens einem Treffer, jeweils beide Seiten zusammen.
    average_detected = photons_per_event.mean()

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
            "RelativeBrightnessToMax": threshold / total_detected * 100 if total_detected > 0 else np.nan,
            "RelativeBrightnessToAverage": threshold / average_detected * 100 if average_detected > 0 else np.nan,
        })

    return {
        "material": str(data["Material"].iloc[0]),
        "energy_mev": float(data["ElectronEnergy"].iloc[0]),
        "n_events": int(data["EventID"].nunique()),
        "fits": pd.DataFrame(rows),
        "average_detected_photons_over_all_events": average_detected,
    }


def plot_title(what, length, relative, energy_mev=None, material=None):
    setup = [f"L = {length:g} mm"]
    if material is not None:
        label = material_label(material)
        setup.append(label if relative else f"Scintillator: {label}")
    if energy_mev is not None:
        setup.append(f"E = {format_energy(energy_mev)}")
    if relative:
        setup += [f"PDE: {'applied' if PDE_APPLIED else 'not applied'}", f"SiPM: {SIPM_ACTIVE_AREA}"]
        return f"{what}\n" + " | ".join(setup)
    setup += [f"PDE: {PDE_APPLIED}", f"SiPM active area: {SIPM_ACTIVE_AREA}"]
    return f"{what}\n" + ", ".join(setup)


def event_statistics_label(result):
    average = result["average_detected_photons_over_all_events"]
    return f"{result['n_events']:,} events | " + rf"$\langle N_\gamma\rangle$={average:,.1f}"


def plot_curves(curves, title, path, relative, plot_type, plot_key):
    # curves: Liste aus (Label, fits, Farbe, Marker)
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, fits, color, marker in curves:
        if not relative:
            fits = fits[fits["Threshold"].isin(ABSOLUTE_PHOTON_THRESHOLDS)]
        valid = fits.dropna(subset=["Sigma_ps"])
        ax.errorbar(
            valid["RelativeBrightnessToAverage" if relative else "Threshold"],
            valid["Sigma_ps"],
            yerr=valid["SigmaErr_ps"],
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
        )

    ax.set_title(title, loc="left", fontsize=10)
    if relative:
        ax.set_xlabel("threshold brightness / %")
        ranges = RELATIVE_PLOT_RANGE.get(plot_type, {}).get(plot_key)
        if ranges is not None:
            ax.set_xlim(*ranges["x"])
            ax.set_ylim(*ranges["y"])
        else:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
    else:
        ax.set_xlabel("photon threshold N")
        ax.set_xlim(*THRESHOLD_PLOT_RANGE)
    ax.set_ylabel(r"$\sigma$ / ps")

    handles, labels = ax.get_legend_handles_labels()
    for n_columns in range(len(labels), 0, -1):
        # Matplotlib füllt spaltenweise; die sichtbare Reihenfolge soll zeilenweise sein.
        order = [i for column in range(n_columns) for i in range(column, len(labels), n_columns)]
        legend = fig.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=n_columns,
            fontsize=8,
            frameon=False,
            columnspacing=0.8,
            labelspacing=0.3,
            handlelength=1.2,
            handletextpad=0.4,
            borderaxespad=0,
        )
        fig.canvas.draw()
        legend_bounds = legend.get_window_extent(fig.canvas.get_renderer())
        if legend_bounds.width <= 0.96 * fig.bbox.width or n_columns == 1:
            break
        legend.remove()

    fig.tight_layout(rect=(0, legend_bounds.height / fig.bbox.height + 0.01, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_comparisons(results, relative):
    suffix = "rel" if relative else "abs"
    lengths = sorted({key[0] for key in results})
    material_order = {name: i for i, name in enumerate(MATERIAL_FOLDERS)}
    materials = sorted(
        {key[1] for key in results},
        key=lambda name: (material_order.get(name, len(material_order)), name),
    )
    energies = sorted({key[2] for key in results})

    # Feste Farbe pro Material bzw. Energie und fester Marker pro Position
    material_colors = {name: plt.cm.tab10(i) for i, name in enumerate(materials)}
    energy_colors = {energy: plt.cm.tab10(i) for i, energy in enumerate(energies)}
    markers = "osv^Dp"

    for length in lengths:
        length_dir = OUT_DIR / f"{length:g}mm"

        # Länge und Energie fest, Material variiert
        for energy in energies:
            curves = []
            for i, material in enumerate(materials):
                if (length, material, energy) not in results:
                    continue
                result = results[(length, material, energy)]
                label = f"{material_label(material, multiline=True)}\n{event_statistics_label(result)}"
                curves.append((label, result["fits"], material_colors[material], markers[i % len(markers)]))

            if len(curves) < 2:
                continue

            what = (
                r"Time resolution $\sigma(t_{\mathrm{top}}-t_{\mathrm{bottom}})$ vs relative brightness of threshold (diff. materials)"
                if relative else
                r"Time resolution $\sigma(t_{top} - t_{bottom})$ vs photon threshold, different materials"
            )
            energy_key = format_energy(energy).replace(" ", "")
            plot_curves(
                curves,
                plot_title(what, length, relative, energy_mev=energy),
                length_dir / f"materials_{energy_key}_{suffix}.png",
                relative,
                "materials",
                energy_key,
            )

        # Länge und Material fest, Energie variiert
        for material in materials:
            curves = [
                (
                    f"{format_energy(energy)}\n{event_statistics_label(results[(length, material, energy)])}",
                    results[(length, material, energy)]["fits"],
                    energy_colors[energy],
                    markers[i % len(markers)],
                )
                for i, energy in enumerate(energies)
                if (length, material, energy) in results
            ]
            if len(curves) < 2:
                continue

            what = (
                r"Time resolution $\sigma(t_{top} - t_{bottom})$ vs relative brightness of threshold (diff. energies)"
                if relative else
                r"Time resolution $\sigma(t_{top} - t_{bottom})$ vs photon threshold, different energies"
            )
            plot_curves(
                curves,
                plot_title(what, length, relative, material=material),
                length_dir / f"energies_{material}_{suffix}.png",
                relative,
                "energies",
                material,
            )


def main():
    if not MATERIAL_FOLDERS:
        raise SystemExit("No materials selected in MATERIAL_FOLDERS")
    files = []
    for material_folder in sorted(set(MATERIAL_FOLDERS)):
        material_dir = DATA_ROOT / material_folder
        material_files = list(material_dir.glob("*/*mm.csv"))
        if not material_files:
            raise SystemExit(f"No CSV files found for material folder {material_dir}")
        files.extend(material_files)

    results = {}
    for path in sorted(files):
        length = float(path.name.removesuffix("mm.csv"))
        result = analyse_file(path)
        key = (length, result["material"], result["energy_mev"])
        results[key] = result
        print(f"Material: {result['material']}")
        print(f"Length: {length} mm")
        print(f"Energy: {result['energy_mev']} MeV")
        print(f"Average detected photons over all events: {result['average_detected_photons_over_all_events']}")
        print("---------------------------------")

    plot_comparisons(results, relative=False)
    plot_comparisons(results, relative=True)


if __name__ == "__main__":
    main()
