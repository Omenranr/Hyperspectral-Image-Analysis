import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk, erosion, reconstruction
from skimage.util import invert

def imshow_stretch(I, stretchval=0.1):
    """Show a grayscale or RGB image after performing a stretching of its dynamic"""
    nbins = 1000
    tol_low = stretchval
    tol_high = 1-stretchval
    Iadj = np.zeros(I.shape)
    
    n_bands = 1 if len(I.shape) < 3 else I.shape[2]
    for i in range(n_bands):
        tmp = I if n_bands == 1 else I[:,:,i]
        tmp = tmp - np.min(tmp.ravel())
        tmp = tmp / np.max(tmp.ravel())
        N, _ = np.histogram(tmp.ravel(), nbins)
        cdf = np.cumsum(N)/np.sum(N)  # cumulative distribution function
        ilow = np.where(cdf > tol_low)[0][0]
        ihigh = np.where(cdf >= tol_high)[0][0]
        ilow = (ilow - 1)/(nbins-1)
        ihigh = (ihigh - 1)/(nbins-1)
        li = ilow
        lo = 0
        hi = ihigh
        ho = 255
        shape = tmp.shape
        out = (tmp < li) * lo
        out = out + np.logical_and(tmp >= li, tmp < hi) * (lo + (ho - lo) * ((tmp - li) / (hi - li)))
        out = out + (tmp >= hi) * ho
        if n_bands == 1:
            Iadj = out
        else:
            Iadj[:,:,i] = out
    plt.imshow(Iadj.astype('uint8'))

def gen_train_test(orig_labels, labeled_classes, classes, n_samples_per_class):
    """function generating a training and a test set from the labelled set of 
    samples L (typically an image with labelled pixels set to values greater
    than zero and correspondent to their class index).
    Warning! It is assumed that the given number of samples is not less than
    the minimum available number of labelled samples for each class."""
    n_classes = classes.shape[0]
    # generate a random set of labelled samples used for training
    tr_idx = np.zeros((n_samples_per_class * n_classes,), dtype=int)
    for i in range(n_classes):
        idx = np.where(orig_labels == classes[i])[0]
        idx = idx[np.random.permutation(idx.shape[0])]
        tr_idx[i * n_samples_per_class : (i + 1) * n_samples_per_class] = idx[:n_samples_per_class]
    # remove samples used for training from the test
    ts = np.copy(orig_labels)
    tr_idx = tr_idx.astype(int)
    ts[tr_idx] = 0
    ts_idx = np.where(ts > 0)[0]
    return tr_idx, ts_idx
    
def morph_prof(im, si, st, nb):
    """Computes the morphological profile by using a circular SE (disk)
    and returns a solution in multispectral format: [h w nb_b]
    #
    # INPUTS
    #
    # im: the images to processed
    # si: the size of the se
    # st: the step of the size
    # nb the number of opening/closing
    #
    # OUTPUTS
    #
    # MP: the morphological profile."""
    H, W = im.shape
    imc = invert(im)
    MP = np.zeros((H, W, 2 * nb + 1))
    i=1
    MP[:, :, nb + 1] = im
    # Computing MP
    for j in range(nb):
        se = disk(si)
        temp0 = erosion(im, selem=se)
        tempC = erosion(imc, selem=se)
        temp0 = reconstruction(temp0, im)
        tempC = reconstruction(tempC, imc)
        MP[:,:,nb + i] = temp0
        MP[:,:,nb - i] = invert(tempC)
        si += st
        i += 1
    return MP

