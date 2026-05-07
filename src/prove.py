# Script personale per testare varie funzionalità (generalmente su singoli pazienti)
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
from pathlib import Path

# Path per il dadaset, serve a risolvere il percorso richiesto da BIDS.
root = Path(__file__).resolve().parent.parent / "data" 



# Eseguiamo la scansione di tutti i soggetti 
for i in range(1, 2):
        
    bids_path = BIDSPath(
        subject="001",
        task="motion",
        run="4",
        datatype="eeg",
        root=root
    )
    
    try:
        raw = read_raw_bids(bids_path, verbose=False)  # Memorizzo in raw i dati EEG del soggetto
        events, event_id = mne.events_from_annotations(raw, verbose=False) # Memorizzo in events gli eventi e in event_id le corrispondenti etichette
        raw.plot(scalings='auto', show=True, block=True)
        # Plotto il contenuto in frequenza. Faccio in modo che non si chiusa la pagina.
        raw.load_data(verbose=False)
        raw.notch_filter(freqs=60, method='spectrum_fit', verbose=False)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        print(raw.info['bads'])
        psds = raw.compute_psd(
            method="welch",
            fmin=8,
            fmax=70,
            n_fft=256
        )
        # psds.plot()
        # plt.show()  
        

    except Exception as e:
        print(f"Errore {"1"}: {e}")
        
 