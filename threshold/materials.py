# Materialkonstanten, Quelle: Eljen-Datenblätter
# EJ-232/EJ-232Q: https://eljentechnology.com/products/plastic-scintillators/ej-232-ej-232q
# EJ-228/EJ-230:  https://eljentechnology.com/products/plastic-scintillators/ej-228-ej-230
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
    "BC422": {
        "rise_time_ps": 500,
        "decay_time_ps": 1400,
        "light_output_pct_anthracene": 64,
        "photons_per_mev": 9700,
        "emission_max_nm": 391,
        "attenuation_length_cm": 120,
        "refractive_index": 1.58,
    }
}


def material_label(name):
    # Materialname mit Rise und Decay Zeit, für Titel und Legenden
    properties = MATERIALS.get(name)
    if properties is None:
        return name
    return (
        rf"{name} ($\tau_{{rise}}$ = {properties['rise_time_ps']:g} ps, "
        rf"$\tau_{{decay}}$ = {properties['decay_time_ps']:g} ps)"
    )
