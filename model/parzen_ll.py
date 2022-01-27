# Derived from the DCGAN paper's Parzen Estimation LL calculations:
#
# Fundamentally, no changes to the method; some updates for Python2 -> Python3.7+, adding
# verbose comments on LL methods, cutting CLI wrapper, and handling for sending Pytorch.datasets
# data to Theano
#
# See: https://github.com/goodfeli/adversarial/blob/master/parzen_ll.py
#
# NLL Functions from DCGAN Paper: Credit: Yann N. Dauphin

import datetime
import gc

import numpy as np

import theano
import theano.tensor as T
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

    
def get_nll(x, parzen, batch_size=10):
    """
    Calculate the Negative Log-Liklihood over X using parzen function
    -------
    Args:
        X -
        parzen - theano.function, see `theano_parzen`
        batch_size - int - # of images to use for each NLL sample
    """

    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    nlls = []
    for i in range(n_batches):
        nll = parzen(x[inds[i::n_batches]])
        nlls.extend(nll)
        if i % 10 == 0:
            print(
                f"[{datetime.datetime.utcnow().__str__()}]\t[{i}/{n_batches}]\tMean NLL: {np.mean(nlls)}"
            )

    return np.array(nlls)


def log_mean_exp(a):
    max_ = a.max(1)
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, "x")).mean(1))


def theano_parzen(mu, sigma):
    """
    Create Parzen function from sample of Mu (i.e. Samples from G)
    -------
    Args:
        - mu - np.Array - Samples from G cast to NDArray and reshaped
        - sigma - float32 - proposed sigma value for Parzen Kernel
    """

    x = T.matrix()
    mu = theano.shared(mu)

    a = (x.dimshuffle(0, "x", 1) - mu.dimshuffle("x", 0, 1)) / sigma
    E = log_mean_exp(-0.5 * (a ** 2).sum(2))
    Z = mu.shape[1] * T.log(sigma * np.sqrt(np.pi * 2))

    return theano.function([x], E - Z)


def cross_validate_sigma(g_samples, data, sigmas, batch_size):
    """
    Select optimal kernel size for Parzen
    -------
    Args:
        g_samples - numpy.ndarray - Sample images from G
        data - numpy.ndarray - Sample images from MSLS
        sigmas - numpy.ndarray - array of sigmas to test
    """

    lls = []
    for sigma in sigmas:
        print(f"[{datetime.datetime.utcnow().__str__()}]\t[σ = {sigma}]")

        parzen = theano_parzen(g_samples, sigma)
        tmp = get_nll(data, parzen, batch_size=batch_size)

        lls.append(np.asarray(tmp).mean())
        del parzen
        gc.collect()

    ind = np.argmax(lls)
    print(f"[{datetime.datetime.utcnow().__str__()}]\t[Using: σ = {sigma}]")
    return sigmas[ind]
