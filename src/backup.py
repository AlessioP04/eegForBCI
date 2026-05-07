# Prima pipeline completata (non revisionata). Backup del 6/5/2024. Effettuato prima della pubblicazione della repository su GitHub,
# dopo probabilmente non sarà più aggiornato. 
# Viene eseguito PSD con classificatore LDA solo su 3 elettrodi. 
# Sliding window di 1s con passo di 0.1s. 
# Etichette assegnate se un evento è presente per oltre il 40% della finestra, altrimenti rilassamento.
import mne 
from mne_bids import BIDSPath, read_raw_bids
from scipy.signal import welch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path


# Utility per estrarre bande di potenza a precise frequenze
def band_power(psds, freqs, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return psds[:, :, idx].mean(axis=2)

# Path per il dataset, serve a risolvere il percorso richiesto da BIDS.
root = Path(__file__).resolve().parent.parent / "data" 
runs = ["4", "8", "12"]                     # Le run che vengono prese in considerazione

# Eseguiamo la scansione di tutti i soggetti 
for i in range(1, 2):
    subject = f"{i:03d}"
    # Per ogni soggetto eseguiamo la scansione sulle run di nostro interesse
    for run in runs:
        # Il dataset utilizza il protocollo BIDS, cioè le informazioni sono distribuite su più file. 
        # Per leggere i dati, dobbiamo specificare il percorso del file BIDS e BIDSPath si occupa di ricavare tutte le informazioni:
        # channles.tsv, coordinatesystem.json, events.tsv, ecc.
        bids_path = BIDSPath(
            subject=subject,
            task="motion",
            run=run,
            datatype="eeg",
            root=root,
        )
        
        try:
            raw = read_raw_bids(bids_path, verbose=False)  # Memorizzo in raw i dati EEG del soggetto
            events, event_id = mne.events_from_annotations(raw, verbose=False) # Memorizzo in events gli eventi e in event_id le corrispondenti etichette
            # Mapping per praticità (Task2 è l'immaginazione di pugno destro/sinistro. Tx rappresenta lo stato)
            event_map = {
                event_id['TASK2T0']: 1,
                event_id['TASK2T1']: 2,
                event_id['TASK2T2']: 3
            }
            #raw.filter(l_freq=1, h_freq=40, verbose=False)  # Filtro passa banda 1-40 Hz
            #raw.set_eeg_reference('average', projection=False, verbose=False)  # Riferimento medio
            #raw.plot(scalings='auto', show=True, block=True)

            # Genero la sliding window
            sfreq = raw.info['sfreq']  # frequenza di campionamento
            window_size = 1.0  # lunghezza finestra in secondi
            step_size = 0.1  # passo in secondi

            n_samples_window = int(window_size * sfreq)  # numero di campioni per finestra
            n_samples_step = int(step_size * sfreq)  # numero di campioni per passo

            n_samples = raw.n_times # Numero totale di campioni nel segnale

            windows = []    # Lista per memorizzare le finestre

            for start in range(0, n_samples - n_samples_window, n_samples_step):
                end = start + n_samples_window
                window = raw.get_data(start=start, stop=end)    # Ogni finestra contiene 1s di dati di raw
                windows.append(window)

            # Windows è una lista di array, ognuno con i dati della finestra
            windows = np.array(windows)  # forma: (n_windows, n_channels, n_samples_window)

            print(f"Numero finestre: {windows.shape[0]}")
            print(f"Numero canali e campioni per finestra:{windows.shape[1]}, {windows.shape[2]}")
            
            # Selezioniamo solo i canali centrali (TODO: eseguire la segmentazione solo sui canali di interesse)
            central_channels = ["C3", "Cz", "C4"]
            picks = mne.pick_channels(raw.ch_names, central_channels)
            windows_central = windows[:, picks, :]
            # Inizializzo l'array dove memorizzo i psd
            psds = []
            freqs = None
            # Scorro sulle singole finestre
            for w in windows_central:
                psd_window = []
                # Scorro sui canali di ogni finestra (saranno solamente i canali di interesse, non tutti e 64)
                for ch in w:
                    f, p = welch(ch, fs=sfreq, nperseg=160) 
                    psd_window.append(p)
                
                psds.append(psd_window)
                
                if freqs is None:
                    freqs = f

            psds = np.array(psds)
            alpha = band_power(psds, freqs, 8, 12)
            beta  = band_power(psds, freqs, 12, 30)
            gamma = band_power(psds, freqs, 30, 45)

            # Concateno i segnali così ottenuti 
            X = np.concatenate([alpha, beta, gamma], axis=1)  
            X = np.log(X)
            #print(X)  


            event_times = events[:, 0] / sfreq  # Tempi degli eventi in secondi
            event_codes = events[:, 2]  # Codici degli eventi
            event_labels = np.array([event_map[e] for e in event_codes])    # Etichette corrispondenti ai codici degli eventi
            # Costruisco gli intervalli di tempo di ogni evento
            intervals = []
            for i in range(len(event_times) - 1):   # Per il numero di eventi - 1
                start = event_times[i]      # Prendo l'istante iniziale
                end = event_times[i + 1]    # L'istante finale 
                label = event_labels[i]     # La relativa etichetta
                intervals.append((start, end, label))   # Aggiungo l'intervallo alla lista

            # Assegno le etichette solo se nella finestra un evento è presente per oltre il 40%, altrimenti assegno rilassamento
            y = []
            threshold = 0.4

            labels_per_sample = np.zeros(raw.n_times, dtype=int)
            
            for istart, iend, ilabel in intervals:
                s = int(istart * sfreq)
                e = int(iend * sfreq)
                labels_per_sample[s:e] = ilabel

            for start in range(0, raw.n_times - n_samples_window, n_samples_step):
                end = start + n_samples_window
                window_labels = labels_per_sample[start:end]
                
                frac_left  = np.mean(window_labels == 2)
                frac_right = np.mean(window_labels == 3)
                
                if frac_left >= threshold:
                    y.append(2)
                elif frac_right >= threshold:
                    y.append(3)
                else:
                    y.append(1)

            y = np.array(y)
            print(f"Numero di campioni in X e di feature per campione: {X.shape}")
            print(f"Numero di etichette: {y.shape[0]}")
            #print(y.tolist())
        except Exception as e:
            print(f"Errore {subject}: {e}")
        

clf = make_pipeline(
    StandardScaler(),
    LinearDiscriminantAnalysis()
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv)
print("Accuracy media:", scores.mean())
