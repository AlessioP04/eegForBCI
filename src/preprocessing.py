# Utilities per il pre-processing

import numpy as np

# Crea sliding window sui dati raw. Restituisce una matrice tridimensionale con tutte le finestre, la dimensione della finestra in campioni, la dimensione dello step in campioni e il numero totale di campioni.
def create_sliding_windows(raw, window_size, step_size):
    sfreq = raw.info['sfreq']

    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    total_samples = raw.n_times
    windows = []

    for start in range(0, total_samples - window_samples, step_samples):
        end = start + window_samples
        windows.append(raw.get_data(start=start, stop=end))

    return (
        np.array(windows),
        window_samples,
        step_samples,
        total_samples
    )

# Crea un array di etichette per ogni finestra. Strettamente legato alla funzione sopra (richiede gli stessi valori di x_samples per essere allineato)
# Testato manualmente: le etichette coincidono con gli eventi. (Test effettuato su soggetto 1, 2)
def create_window_labels(
    events,
    event_map,
    total_samples,
    window_samples,
    step_samples,
    threshold=0.5
):
    labels_per_sample = np.zeros(total_samples, dtype=int)
    event_samples = events[:, 0]
    event_codes = events[:, 2]

    for i in range(len(event_samples) - 1):
        start = event_samples[i]
        end = event_samples[i + 1]
        labels_per_sample[start:end] = event_map[event_codes[i]]

    y = []

    for start in range(0, total_samples - window_samples, step_samples):
        end = start + window_samples
        window_labels = labels_per_sample[start:end]

        frac_left = np.mean(window_labels == 2)
        frac_right = np.mean(window_labels == 3)

        if frac_left >= threshold:
            y.append(2)

        elif frac_right >= threshold:
            y.append(3)

        else:
            y.append(1)

    return np.array(y)