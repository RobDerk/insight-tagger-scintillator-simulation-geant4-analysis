# Materialkonstanten, Quelle: Eljen-Datenblätter
# EJ-232/EJ-232Q: https://eljentechnology.com/products/plastic-scintillators/ej-232-ej-232q
# EJ-228/EJ-230:  https://eljentechnology.com/products/plastic-scintillators/ej-228-ej-230
# BC-422: https://luxiumsolutions.com/radiation-detection-scintillators/plastic-scintillators/fast-timing-bc-418-bc-420-bc-422-bc-422q
MATERIALS = {
    "EJ228": {
        "rise_time_ps": 500,
        "decay_time_ps": 1400,
        "light_output_pct_anthracene": 67,
        "photons_per_mev": 10200,
        "emission_max_nm": 391,
        "attenuation_length_cm": 20,
        "refractive_index": 1.58,
    },
    "EJ230": {
        "rise_time_ps": 500,
        "decay_time_ps": 1500,
        "light_output_pct_anthracene": 64,
        "photons_per_mev": 9700,
        "emission_max_nm": 391,
        "attenuation_length_cm": 120,
        "refractive_index": 1.58,
    },
    "EJ232": {
        "rise_time_ps": 350,
        "decay_time_ps": 1600,
        "light_output_pct_anthracene": 55,
        "photons_per_mev": 8400,
        "emission_max_nm": 370,
        "attenuation_length_cm": 10,
        "refractive_index": 1.58,
    },
    # EJ-232Q mit 0.5 % Benzophenon
    "EJ232Q05": {
        "rise_time_ps": 110,
        "decay_time_ps": 700,
        "light_output_pct_anthracene": 19,
        "photons_per_mev": 2900,
        "emission_max_nm": 370,
        "attenuation_length_cm": 10,
        "refractive_index": 1.58,
    },
    # Bulk light attenuation length; the typical 1/e length is not applicable.
    "BC422": {
        "rise_time_ps": 350,
        "decay_time_ps": 1600,
        "light_output_pct_anthracene": 55,
        "emission_max_nm": 370,
        "attenuation_length_cm": 8,
        "refractive_index": 1.58,
    }
}


def material_label(name, multiline=False):
    # Materialname mit Rise und Decay Zeit, für Titel und Legenden
    properties = MATERIALS.get(name)
    if properties is None:
        return name
    if multiline:
        return (
            f"{name}\n"
            rf"$(\tau_r,\tau_d)$ = ({properties['rise_time_ps']:g}, {properties['decay_time_ps']:g}) ps"
        )
    return (
        rf"{name} ($\tau_r$ = {properties['rise_time_ps']:g} ps, "
        rf"$\tau_d$ = {properties['decay_time_ps']:g} ps)"
    )
