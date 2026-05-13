CSP: common spatial pattern, date due classi massimizza la varianza di una mentre minimizza l'altra.
Mette in evidenza la discriminabilità tra le classi, cioè il loro grado di separazione. Enfatizza quindi le differenze spaziali, 
ottimo per EEG e motor imagery.
Contro: non si adatta alla non stazionaretà del segnale, rischia di overfittare facilmente, è sensibile al rumore e nella forma 
standard permette solo una classificazione binaria. 
Esistono però varianti quale il regularized, filter bank e multiclass CSP.


PCA: principal component analisys, una tecnica statistica di elaborazione del segnale utilizzata per ridurre la dimensionalità 
dei dati e separare le componenti principali dell'attività cerebrale.
Riduce la dimensionalità ed il rumore ma è poco indicato per ambiente motor imagery. E' spesso usata in combinazione con
l'ICA che permette una migliore rimozione degli artefatti, poichè in grado di separare in maniera più ottimale le sorgenti
indipendenti.


------------------------------------------------------------------
Test e risultati
CSP tende a fornire feature lineari e si presta quindi maggiormente ad un classificatore lineare quale LDA.
PCA richiede invece un classificatore più complesso, come SVM con kernel non lineare.
Ad un primo sguardo, visualizzando graficamente alcune componenti delle feature, risulta evidente, in generale, la migliore 
separazione ottenuta tramite CSP. 
Usando le feature estratte per addestrare classificatori LDA e SVM, a parità di tutte le altre caratteristiche (threshold, valutazione, canali estratti, etc.) ed effettuando una girdSearch con il fine di massimizzare l'accuratezza dei modelli, si ottengono, in media, prestazioni circa il 10% superiori utilizzando CSP piuttosto che PCA.