"""
Implements Theano Window Estimation as described in the original DCGAN paper,
See: https://github.com/goodfeli/adversarial/blob/master/parzen_ll.py
"""

import datetime
import gc

import numpy as np

import theano
import theano.tensor as T


def get_nll(x, parzen, batch_size=10):
    """Calculate the Negative Log-Liklihood over X using parzen function"""

    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    nlls = []
    for i in range(n_batches):
        nll = parzen(x[inds[i::n_batches]])
        nlls.extend(nll)

    return np.array(nlls)


def log_mean_exp(a):
    max_ = a.max(1)
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, "x")).mean(1))


def theano_parzen(mu: np.array, sigma: float) -> thano.function:
    """
    Create Parzen function from sample of Mu (i.e. Samples from G)

    Args:
    --------
        - mu: np.Array: Samples from G cast to NDArray and reshaped

        - sigma: float32: proposed sigma value for Parzen Kernel
    """

    x = T.matrix()
    mu = theano.shared(mu)

    a = (x.dimshuffle(0, "x", 1) - mu.dimshuffle("x", 0, 1)) / sigma
    E = log_mean_exp(-0.5 * (a ** 2).sum(2))
    Z = mu.shape[1] * T.log(sigma * np.sqrt(np.pi * 2))

    return theano.function([x], E - Z)


def cross_validate_sigma(
    g_samples: np.array, data: np.array, sigmas: np.array, batch_size: int
) -> float:
    """
    Select optimal kernel size for Parzen

    Args:
    --------
        - g_samples: np.ndarray: Sample images from G

        - data: np.ndarray: Sample images from MSLS

        - sigmas: np.ndarray: array of sigmas to test
    """

    lls = []
    for sigma in sigmas:
        parzen = theano_parzen(g_samples, sigma)
        nll = get_nll(data, parzen, batch_size=batch_size).mean()
        lls.append(nll)
        print(
            f"[{datetime.datetime.utcnow().__str__()}]\t[σ = {sigma:.4f}]\t[nll: {nll:.4f}]"
        )

        del parzen
        gc.collect()

    ind = np.argmax(lls)
    print(f"[{datetime.datetime.utcnow().__str__()}]\t[σ = {sigmas[ind]:.4f}]")
    return sigmas[ind]
