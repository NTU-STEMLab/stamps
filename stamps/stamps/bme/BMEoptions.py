# -*- coding: utf-8 -*-
"""BME options

This module has a same name class for store 
parameters for BME calculation.
"""

import numpy as np

class BMEoptions(object):
    """BME options for calculation.

    """

    def __init__(self):
        """Initialize the BMEoptions instance.

        You can use this instance like a dictionary, get or set custom
        key and value.

        Examples:
            >>> bmeoptions = BMEoptions()
            >>> bmeoptions[2]  # The maximum number of evaluation of the integral
            10000 
            >>> bmeoptions[28] = 10 # set #29 to 10

        """

        options = np.array([
            [0.],  # 1
            [0.0001],  # 2
            [32768.],  # 3 is the maximum number of evaluation of the integral
            [0.001],  # 4 is the relative error on the estimation of the integral
            [0.],  # 5 (test)
            [25.],  # 6
            [0.001],  # 7
            [2.],  # 8 number of moments calculated
                  # 1 for mean, 2 for mean and std dev, 3 for all three moments)
            [0.],  # 9 (test)
            [0.],  # 10 (test)
            [0.],  # 11 (test)
            [0.],  # 12 (test)
            [0.],  # 13 (test)
            [100.],  # 14
            [0.],  # 15 (test)
            [0.],  # 16 (test)
            [0.],  # 17 (test)
            [0.],  # 18 (test)
            [0.],  # 19 (test)
            [0.68],  # 20
            [0.],  # 21
            [0.],  # 22
            [0.],  # 23
            [0.],  # 24
            [0.],  # 25
            [0.],  # 26
            [0.],  # 27
            [0.],  # 28
            [0.],  # 29
            ])

        #make options pythonic
        self.options_dict = {}
        for idx, k in enumerate(options):
            self.options_dict[idx] = k
            self.options_dict[idx,0] = k[0]

        #we can add more options with key and value here
        #and old code still compatible
        self.options_dict['integration method'] = 'qmc'
        self.options_dict['qmc_showinfo'] = False
        self.options_dict['ck pdf debug'] = False
        self.options_dict['prior_pdf_at_ck'] = None # New option for external prior PDF (1D numpy array)
        self.options_dict['use_neighbor_average_as_prior'] = False # New option to enable/disable neighbor averaging for prior
        self.options_dict['debug'] = False

        # Acceleration option: when True, non-Gaussian soft data PDFs are
        # approximated as Gaussian (using their mean and variance) so that
        # the fast analytical path is used instead of expensive QMC integration.
        # This trades a small amount of accuracy for a large speed-up.
        self.options_dict['soft_approx_gaussian'] = False

        # Outlier handling for soft data that are statistically far from the
        # conditional prior E[Zs|Zh] (i.e. the interval is incompatible with
        # what the surrounding hard data imply):
        #
        #   'inflate'  (default) - expand the interval toward the conditional
        #                prior mean until the datum becomes compatible.  The
        #                datum stays in the neighbourhood with reduced, but
        #                non-zero, information content.  This is also applied
        #                as a *pre-QMC screening* step so that the IS proposal
        #                is always well-supported inside the interval, avoiding
        #                unnecessary Gaussian fallbacks.  This is appropriate
        #                when the interval bounds themselves carry measurement
        #                uncertainty.
        #
        #   'exclude'  - remove the datum from the neighbourhood entirely.
        #                The posterior uses only the remaining compatible data.
        #
        #   'ignore'   - keep all soft data as-is with no outlier check.
        self.options_dict['outlier_handling'] = 'inflate'

        # Fractional margin added on top of the minimum inflation required to
        # pass the 3-sigma compatibility test.  A value of 0.10 inflates the
        # interval to 110% of the bare minimum.  Only used when
        # outlier_handling = 'inflate'.
        self.options_dict['outlier_inflate_margin'] = 0.10

        # Minimum marginal overlap probability (per dimension) below which the
        # pre-QMC screening triggers an inflation (or exclusion, depending on
        # outlier_handling).  For a uniform soft datum [lo, hi] and conditional
        # prior N(m, sigma^2), the overlap is:
        #   P = Phi((hi-m)/sigma) - Phi((lo-m)/sigma)
        # When P < min_overlap_prob AND m is OUTSIDE [lo, hi], the interval is
        # expanded toward m before QMC is attempted, so the Fused-Gaussian IS
        # proposal has good support inside the inflated interval.
        # Default 0.10 (10%) triggers inflation for intervals that have less
        # than 10% probability mass under the conditional prior.
        self.options_dict['min_overlap_prob'] = 0.10

        # PCA dimension-reduction for importance-sampling QMC integration.
        # When the conditional covariance Σ_{s|h} has eigenvalues spanning
        # several orders of magnitude, the effective dimensionality of the
        # integral is much lower than ns.  By keeping only the principal
        # components that explain at least this fraction of the total
        # variance, we reduce the QMC integration domain from [0,1]^ns to
        # [0,1]^ns_eff (ns_eff ≤ ns).  Set to 1.0 to disable truncation.
        self.options_dict['pca_integration_threshold'] = 0.999

        # Non-negativity constraint: when True, estimation points whose
        # posterior mean is negative are corrected using truncated-normal
        # theory so that the final estimate is always >= 0.  This is
        # appropriate when the variable of interest is physically
        # non-negative (e.g. concentration, precipitation).
        self.options_dict['nonneg_estimate'] = False

    def __setitem__(self, key, value):
        if type(key) == tuple:
            self.options_dict[key] = value
            self.options_dict[key[0]] = [value]
        elif type(key) == str:
            self.options_dict[key] = value
        else:
            raise TypeError(
                'Not a valid key type: {t}'.format(t=type(key))
                )

    def __getitem__ (self, key):
        return self.options_dict[key]

    def get(self, key, default=None):
        """Dict-compatible get() with default value."""
        return self.options_dict.get(key, default)
