import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path


# Utility per estrarre bande di potenza a precise frequenze
def band_power(psds, freqs, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs < fmax)
    band_freqs = freqs[idx]
    if band_freqs.size == 0:
        return np.zeros(psds.shape[:2])
    band_psd = psds[:, :, idx]
    return np.trapezoid(band_psd, band_freqs, axis=2)

root = Path(__file__).resolve().parent.parent / "data"      # Path per il dataset
runs = ["4", "8", "12"]   # Le run che vengono prese in considerazione
X_all = []  # Vettore X con tutti i dati di tutte le finestre
y_all = []  # Vettore y con tutte le etichette di tutte le finestre

# Eseguiamo la scansione di tutti i soggetti 
for i in range(1, 4):
    subject = f"{i:03d}"
    # Per ogni soggetto eseguiamo la scansione sulle run di nostro interesse
    for run in runs:
        bids_path = BIDSPath(   # Specifichiamo il percorso del dataset e BIDS eseguirà correttamente la scansione 
            subject=subject,
            task="motion",
            run=run,
            datatype="eeg",
            root=root,
        )
        
        try:
            # Fase 1: lettura dei dati
            # Memorizzo i dati EEG del soggetto. Raw contiene tutte le informazioni della run "a" di un certo paziente "b"
            raw = read_raw_bids(bids_path, verbose=False)  
            # Memorizzo ogni oggetto event (con tempi, etichette...) nell'array events. Memorizzo le etichette usate in event_id
            events, event_id = mne.events_from_annotations(raw, verbose=False) 
            # Mappo le etichette per praticità
            event_map = {
                event_id['TASK2T0']: 1,
                event_id['TASK2T1']: 2,
                event_id['TASK2T2']: 3
            }
            sfreq = raw.info['sfreq']  # frequenza di campionamento

            # Fase 2: pre-processing
            raw.load_data(verbose=False) # Carico i dati in memoria per poter filtrare ecc.
            raw.filter(l_freq=1, h_freq=40, verbose=False)  # Filtro passa banda 1-40 Hz
            raw.set_eeg_reference('average', projection=False, verbose=False)  # Riferimento medio
            #raw.plot(scalings='auto', show=True, block=True)

            # Genero la sliding window: l'output è la lista windows contenente tutte le window della run
            window_size = 1.0  # Lunghezza finestra in secondi
            step_size = 0.1  # Lunghezza passo in secondi
            window_samples = int(window_size * sfreq)  # Numero di campioni per finestra
            step_samples = int(step_size * sfreq)  # Numero di campioni per passo
            total_samples = raw.n_times # Numero totale di campioni nel segnale

            windows = []    # Lista per memorizzare le finestre
            # Scorro dal campione 0 a quasi la fine saltando di step_samples alla volta
            for start in range(0, total_samples - window_samples, step_samples):
                end = start + window_samples    # Calcolo l'end della finestra
                window = raw.get_data(start=start, stop=end)    # Ogni finestra contiene 1s di dati di raw. Contiene SOLO i dati, non è più possibile eseguire operazioni come plot.
                windows.append(window)
            # La finestra contiene i campioni da start a end: ogni campione contiene a sua volta i valori dei 64 elettrodi.
            # Quindi window è una matrice con righe = 64 elettrodi e colonne = valori negli istanti di campionamento
            # Concatenando le finestre si ottiene una matrice tridimensionale 
            windows = np.array(windows)  # forma: (n_windows, n_channels, window_samples)

            # print(f"Numero canali e campioni per finestra:{windows.shape[1]}, {windows.shape[2]}")
            
            # Selezioniamo solo i canali di nostro interesse
            # preferred_channels = ["C3", "Cz", "C4"]
            # picks = mne.pick_channels(raw.ch_names, preferred_channels)
            # windows_central = windows[:, picks, :]

            # Fase 3: feature extraction (potenza di banda)
            n_fft = min(256, window_samples)
            psds, freqs = mne.time_frequency.psd_array_welch(
                windows,
                sfreq=sfreq,
                fmin=1,
                fmax=45,
                n_fft=n_fft,
                n_overlap=0,
                average="mean",
                verbose=False,
            )

            theta = band_power(psds, freqs, 4, 8)
            alpha = band_power(psds, freqs, 8, 13)
            beta = band_power(psds, freqs, 13, 30)

            # Concateno le bande (n_windows, n_channels * 3)
            features = np.concatenate([theta, alpha, beta], axis=1)
            features = np.log10(features + 1e-12)
            X = features
            

            # Fase 3.B Assegno le etichette ad ogni finestra temporale
            event_times = events[:, 0] / sfreq  # Tempi degli eventi in secondi
            event_codes = events[:, 2]  # Codici degli eventi (es. Task2T0)
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
            labels_per_sample = np.zeros(total_samples, dtype=int)
            for istart, iend, ilabel in intervals:
                s = int(istart * sfreq)
                e = int(iend * sfreq)
                labels_per_sample[s:e] = ilabel
            # Ritondante lo so, ma scorro nuovamente tutti i campioni a salti di step_samples
            for start in range(0, total_samples - window_samples, step_samples):
                end = start + window_samples
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
            #print(f"Numero di campioni in X e di feature per campione: {X.shape}")
            #print(f"Numero di etichette: {y.shape[0]}")

            # Aggiungo tutto agli array complessivi 
            X_all.append(X)
            y_all.append(y)
            print(f"Soggetto {subject} run {run} - Campioni: {X.shape[0]}, Feature per campione: {X.shape[1]}, Etichette: {y.shape[0]}")

        except Exception as e:
            print(f"Errore {subject}: {e}")
        

# Concatenazione finale
X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

# Controllo distribuzione classi
unique, counts = np.unique(y_all, return_counts=True)
print("Distribuzione classi:", dict(zip(unique, counts)))

# Pipeline: scaling + LDA
clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=10, random_state=0),
    LinearDiscriminantAnalysis(solver='svd')  # robusto e stabile
)

# Cross-validation stratificata
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Valutazione con più metriche
scores = cross_validate(
    clf,
    X_all,
    y_all,
    cv=cv,
    scoring=["accuracy", "balanced_accuracy"],
    return_train_score=False
)

print("Accuracy per fold:", scores["test_accuracy"])
print("Accuracy media:", scores["test_accuracy"].mean())

print("Balanced accuracy per fold:", scores["test_balanced_accuracy"])
print("Balanced accuracy media:", scores["test_balanced_accuracy"].mean())
