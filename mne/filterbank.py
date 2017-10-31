from __future__ import division
"""A module for filter bank.

"""
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)
from copy import deepcopy

import numpy as np
from scipy.signal import firwin
import pyfftw.interfaces.numpy_fft as fft

class FilterBank(object):

    def __init__(self, data,
                 bw=None, cf=None, foi=None,
                 order=None, **kwargs):

        self._data = data
        self.nch, self.nsamp = self._data.shape

        self.sfreq = kwargs['sfreq']
        self.delta_f_ix = int(self.nsamp // self.sfreq)

        error_freqs = ValueError('foi is not defined. If foi is not known, define cf and bw instead.')
        if bw is None and cf is None:
            if foi is None:
                raise error_freqs
            else:
                cf, bw = self._get_center_frequencies(foi)

        elif bw is not None and cf is not None:
            foi = self._get_frequency_of_interests(cf, bw)
        else:
            raise error_freqs

        self.bandwidth = bw if bw is not None else self.bandwidth
        self.center_f = cf if cf is not None else self.center_f
        self.freqs = foi if foi is not None else self.freqs
        self.nfreqs = self.freqs.shape[0]

        nsamp_ = int(self.bandwidth * 2 * self.delta_f_ix)

        # Create a prototype filter
        self.order = order if order is not None else self.order

        _, filts = self.create_filter(self.order, self.bandwidth/2., self.sfreq/2., self.nsamp, shift=True)

        cf0 = self.nsamp//2
        idx_ = slice(cf0-int(self.delta_f_ix * self.bandwidth), cf0+int(self.delta_f_ix * self.bandwidth))
        self.filts = np.atleast_2d(filts[idx_])

        fois_ix = self._get_frequency_of_interests(cf, self.bandwidth * 2) * self.delta_f_ix

        # Crop out and index of the fft signal, and demodulate them to DC centered
        tmpwr = np.arange(self.nsamp // 2)
        idx0 = np.logical_and(tmpwr[:,np.newaxis]>=fois_ix[:,0][np.newaxis,:], tmpwr[:,np.newaxis]<fois_ix[:,-1][np.newaxis,:]).T
        win_idx0 = np.ones((self.nfreqs, 1)) * np.arange(tmpwr.size)

        tmp = win_idx0[idx0]
        diff = int(self.nfreqs * tmp.size // self.nfreqs - tmp.size)
        if diff < 0:
            tmp = tmp[:diff]
        elif diff > 0:
            tmp = np.hstack([tmp, np.arange(diff)])

        self._idx1 = np.array(tmp.reshape(self.nfreqs, tmp.size // self.nfreqs), dtype=np.int64)

        # Shift the cropped out signal back to the modulating frequencies
        self._idx2 = np.array(np.arange(self.nfreqs)[:, np.newaxis] + np.zeros((1, nsamp_)), dtype=np.int64)

    def process(self, data):
        return self.filter_data(data, self.filts)

    def filter_data(self, data, filts):

        data_f = np.atleast_2d(fft.rfft(data, planner_effort='FFTW_ESTIMATE'))

        _X = np.zeros((1, self.nfreqs, data_f.size), dtype=np.complex64)
        _X[:, self._idx2, self._idx1] = data_f[:,self._idx1] * filts

        return fft.irfft(_X, axis=-1, planner_effort='FFTW_ESTIMATE')

    @staticmethod
    def _get_center_frequencies(fois):
        cf = np.atleast_2d(fois.mean(axis=-1)).T
        bw = np.diff(fois, axis=-1)
        return cf, bw

    @staticmethod
    def _get_frequency_of_interests(cf, bw):
        bw_ = bw * np.ones((cf.shape[0], 2))
        bw_[:,0] *= -.5
        bw_[:,1] *= .5

        return cf + bw_

    @staticmethod
    def create_filter(order, cutoff, nyquist, N, ftype='fir', output='freq', shift=True):
        """
        Create a prototype filter.
        """
        h = firwin(order, cutoff, nyq=nyquist)

        if output == 'freq':
            w = fft.fftfreq(N)
            w *= (nyquist*2)

            H    = fft.fft(h, n=N, axis=-1, planner_effort='FFTW_ESTIMATE')

            if shift:
                return fft.fftshift(w), fft.fftshift(H)
            else:
                return w, H

        else:
            return h
