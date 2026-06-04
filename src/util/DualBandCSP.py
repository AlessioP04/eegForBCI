from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
import numpy as np

class DualBandCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg='ledoit_wolf', log=True, sfreq=160):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.sfreq = sfreq
        self.csp_mu = None
        self.csp_beta = None

    def _bandpass(self, X, l_freq, h_freq):
        # X shape: (n_windows, n_channels, n_times)
        from mne.filter import filter_data
        return filter_data(
            X.astype(np.float64),
            sfreq=self.sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=False
        )

    def fit(self, X, y):
        X_mu   = self._bandpass(X, 8, 12)
        X_beta = self._bandpass(X, 13, 30)
        
        self.csp_mu   = CSP(n_components=self.n_components, reg=self.reg, log=self.log)
        self.csp_beta = CSP(n_components=self.n_components, reg=self.reg, log=self.log)
        
        self.csp_mu.fit(X_mu, y)
        self.csp_beta.fit(X_beta, y)
        return self

    def transform(self, X):
        X_mu   = self._bandpass(X, 8, 12)
        X_beta = self._bandpass(X, 13, 30)
        
        feat_mu   = self.csp_mu.transform(X_mu)
        feat_beta = self.csp_beta.transform(X_beta)
        
        return np.concatenate([feat_mu, feat_beta], axis=1)