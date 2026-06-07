# File esempio con tutte le funzionalità MNE

import mne
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
mne.set_log_level('WARNING')

root = Path("data").resolve()

bids_path = BIDSPath(
    subject="001",
    task="motion",
    run="5",
    datatype="eeg",
    root=root,
)

raw = read_raw_bids(bids_path, verbose=False)
raw.load_data(verbose=False)

# ── INFO ─────────────────────────────────────────────────────────────────────
#print(raw.info)
print(f"Canali:    {len(raw.ch_names)}")
print(f"Fs:        {raw.info['sfreq']} Hz")
print(f"Durata:    {raw.times[-1]:.2f} s")
print(f"Campioni:  {raw.n_times}")

events, event_id = mne.events_from_annotations(raw, verbose=False)
print(f"Eventi:    {event_id}")

# ── VISUALIZZAZIONE SEGNALE GREZZO ────────────────────────────────────────────
# Apre una finestra interattiva: puoi scorrere il segnale, zoomare, escludere canali
raw.plot(
    n_channels=10,       # canali visibili contemporaneamente
    duration=10,         # secondi visibili per pagina
    scalings="auto",
    title="Segnale grezzo",
    block=True           # blocca l'esecuzione finché non chiudi la finestra
)

# ── PREPROCESSING ─────────────────────────────────────────────────────────────

# Riferimento medio
raw.set_eeg_reference('average', projection=False, verbose=False)

# Filtro notch (rimozione interferenza di rete 50 Hz)
raw.notch_filter(freqs=50, verbose=False)

# Filtro passa banda
raw.filter(l_freq=8, h_freq=30, verbose=False)

# Visualizza dopo il filtraggio
raw.plot(
    n_channels=10,
    duration=10,
    scalings="auto",
    title="Segnale filtrato (8-30 Hz)",
    block=True
)

# ── PSD (densità spettrale di potenza) ────────────────────────────────────────
# Mostra la distribuzione di potenza per frequenza — utile per verificare
# che il filtro abbia funzionato e per individuare artefatti spettrali
raw.compute_psd(fmin=1, fmax=60).plot(
    picks=["C3", "C4", "Cz"],
)

# ── EPOCHING ──────────────────────────────────────────────────────────────────
# Ritaglia il segnale attorno agli eventi: [-1s, +4s] rispetto all'onset
events, event_id = mne.events_from_annotations(raw, verbose=False) 
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-1.0,
    tmax=4.0,
    baseline=(-1.0, 0),   # baseline correction automatica
    preload=True,
    verbose=False
)

print(epochs)

# Visualizza le epoche: ogni riga è un trial
epochs.plot(
    n_epochs=5,
    n_channels=10,
    scalings="auto",
    title="Epoche",
    block=True
)

# ── ERP (media delle epoche per classe) ───────────────────────────────────────
evoked_left  = epochs["TASK3T1"].average()
evoked_right = epochs["TASK3T2"].average()

# Plot ERP per canale singolo
mne.viz.plot_compare_evokeds(
    {"Sinistra": evoked_left, "Destra": evoked_right},
    picks=["C3", "C4", "Cz"],
    title="ERP medio - C3, C4, Cz"
)
