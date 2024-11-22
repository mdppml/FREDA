"""
This file is adapted from [Wenda: Weighted Elastic Net for Unsupervised Domain Adaptation].
Original Repository: https://github.com/PfeiferLabTue/wenda
License: GNU General Public License v3.0 (GPL-3.0)
"""

import numpy as np


def printTestErrors(pred_raw, test_y_raw, heading=None, indent=0):
    prefix = " " * indent
    errors = np.abs(test_y_raw - pred_raw)
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    corr = np.corrcoef(pred_raw, test_y_raw)[0, 1]
    std = np.std(errors)
    q75, q25 = np.percentile(errors, [75, 25])
    iqr = q75 - q25
    if heading is not None:
        print(prefix + heading)
        print(prefix + len(heading) * '-')
    print(prefix + "Mean abs. error:", mean_err)
    print(prefix + "Median abs. error:", median_err)
    print(prefix + "Correlation:", corr)
    print()
    return [mean_err, median_err, corr, std, iqr]


class StandardNormalizer:
    """
    Class to (de-)standardize data before/after model generation.

    Standardization is performed by subtracting the mean from each variable
    and dividing by its standard deviation (plus a small constant for numerical
    stability). Thus, each normalized variable has mean zero and a standard
    deviation of approx. 1.

    Attributes:
    -----------
    @var _eps:   epsilon used for numerical stability
                 (when dividing by small standard deviations)
    @var _axis:  axis along which data is normalized
    """

    def __init__(self, x, epsilon=1e-6, axis=0):
        """
        Constructs a normalizer object.

        Parameters:
        -----------
        @param x:         input used to tune the normalizer, i.e., to estimate feature
                          means and standard deviations
        @param epsilon:   (optional) epsilon for numerical stability, default: 1e-6
        @param axis:      (optional) axis index, along which the data should be normalized,
                          (i.e., after normalization the mean and std of x along axis are
                          0 and 1, respectively), default: 0
        """
        self._eps = epsilon
        self._axis = axis
        self._means = np.mean(x, axis=axis)
        self._stds = np.std(x, axis=axis) + self._eps

    def normalize(self, x):
        """
        Normalizes x with means and standard deviations estimated during initialization.

        Parameters:
        -----------
        @param x:     input to normalize, is expected to be a 1d or 2d numpy array

        @return: normalized x
        """
        # transform x depending on axis
        if self._axis == 0:
            x_norm = (x - self._means) / self._stds
        elif self._axis == 1:
            x_norm = ((x.T - self._means) / self._stds).T
        else:
            raise RuntimeError("Axis must be in {0, 1}!")
        # return normalized x (and mean + std if they were estimated)
        return x_norm

    def denormalize(self, x):
        """
        Denormalizes data.

        Parameters:
        -----------
        @param x:     data to denormalize, is expected to be a 1d or 2d numpy array
        """
        if self._axis == 0:
            return self._stds * x + self._means
        elif self._axis == 1:
            return (self._stds * x.T + self._means).T
        else:
            raise RuntimeError("Axis must be in {0, 1}!")

    def __repr__(self):
        return "StandardNormalizer(_eps={0!r}, _axis={1!r})".format(self._eps, self._axis)


class HorvathNormalizer:
    """
    Combines Horvath's age transformation with a StandardNormalizer.
    """

    def __init__(self, x, epsilon=1e-6, adult_age=20):
        """
        Constructs a normalizer object.

        Parameters:
        -----------
        @param x:         input used to tune the normalizer, i.e., to estimate feature
                          means and standard deviations, is expected to be a 1d numpy array
        @param epsilon:   (optional) parameter of StandardNormalizer
        @param adult_age: (optional) parameter of age transformation
        """
        x_trans = age_transform(x, adult_age=adult_age)
        self._adult_age = adult_age
        self._std_normalizer = StandardNormalizer(x_trans, epsilon=epsilon)

    def normalize(self, x):
        """
        Normalizes data.

        Parameters:
        -----------
        @param x:     input to normalize

        @return: normalized x
        """
        x_trans = age_transform(x, adult_age=self._adult_age)
        return self._std_normalizer.normalize(x_trans)

    def denormalize(self, x):
        """
        Denormalizes data.

        Parameters:
        -----------
        @param x:     data to denormalize

        @return: denormalized x
        """
        x_back = self._std_normalizer.denormalize(x)
        return age_back_transform(x_back, adult_age=self._adult_age)

    def __repr__(self):
        return "HorvathNormalizer(_adult_age={0!r}, _std_normalizer={1!r})".format(self._adult_age,
                                                                                   self._std_normalizer)


def age_transform(age, adult_age=20):
    """
    Transforms an age as according to Horvath (2013).

    Reference:
    S. Horvath, "DNA methylation age of human tissue and cell types",
    Genome Biology, vol. 14, no. 10, 2013.
    """
    age = (age + 1) / (1 + adult_age)
    y = np.where(age <= 1, np.log(age), age - 1)
    return (y)


def age_back_transform(trans_age, adult_age=20):
    """
    Inverse function to age_transform(age, adult_age=20).
    """
    y = np.where(trans_age < 0,
                 (1 + adult_age) * np.exp(trans_age) - 1,
                 (1 + adult_age) * trans_age + adult_age)
    return y


def sec2str(seconds):
    """
    Converts a number of seconds (float or integer) into a human readable string.
    """
    assert seconds > 0, 'A negative number of seconds is not allowed!'
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return "{0:.0f} days, {1:.0f} h, {2:.0f} min, {3:.2f} sec".format(d, h, m, s)
    elif h > 0:
        return "{0:.0f} h, {1:.0f} min, {2:.2f} sec".format(h, m, s)
    elif m > 0:
        return "{0:.0f} min, {1:.2f} sec".format(m, s)
    else:
        return "{0:.2f} sec".format(s)
