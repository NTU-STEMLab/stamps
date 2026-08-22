# -*- coding: utf-8 -*-
import numpy as np

from .coord2dist import coord2dist
from .isspacetime import isspacetime
from ..models.covmodel import get_model


def coord2K(c1, c2, models, params, aniso_angle=None, aniso_ratio=None):
    """Compute a nested covariance matrix between two sets of space/space-time coordinates.

    Evaluates a nested covariance model (sum of structured components) at
    the Euclidean distances between all pairs (c1[i], c2[j]), returning the
    combined covariance matrix and the individual component matrices.

    This is the fundamental building block for kriging and BME estimation:
    it constructs the kriging covariance matrices K(c_h, c_h), k(c_k, c_h),
    and K(c_s, c_s) from the model specification.

    Parameters
    ----------
    c1 : np.ndarray, shape (m1, nd)
        Space/space-time coordinates of the first set of *m1* locations.
        ``nd=1`` for purely temporal, ``nd=2`` for 2-D spatial,
        ``nd=3`` for space-time (easting, northing, time).
    c2 : np.ndarray, shape (m2, nd)
        Space/space-time coordinates of the second set of *m2* locations.
    models : list of str
        Covariance model names for each nested component.  The format
        depends on the space-time structure (detected by ``isspacetime``):

        * **Pure spatial** ``nd ≤ 2``::

              ['exponentialC', 'nuggetC', ...]

        * **Separable S/T** (``'/'`` in name or list-of-lists structure)::

              ['exponentialC/exponentialC', ...]   # forward-slash notation
              ['exponentialC', 'exponentialC', ...]  # paired with ST params

        * **Non-separable S/T** (``'CST'`` suffix)::

              ['gaussianCST', 'exponentialCST', ...]

    params : list of tuple
        Covariance parameters for each nested component.  The expected
        content depends on the model type:

        * **Pure spatial** — ``(sill, ar)`` or ``(sill,)`` for nugget.
        * **Separable S/T** — ``(sill, ar_spatial, ar_temporal)`` per component.
        * **Non-separable S/T** — ``(sill, ar_combined, s_t_ratio)`` per component,
          where the combined distance is ``dist_s + s_t_ratio * dist_t``.

    Returns
    -------
    Ki_sum : np.ndarray, shape (m1, m2)
        Sum of all nested covariance components: the total covariance matrix.
    Ki : list of np.ndarray
        Individual component matrices, each of shape (m1, m2), one per
        entry in *models*.

    Other Parameters
    ----------------
    aniso_angle : float or None
        Principal-axis angle of geometric anisotropy in **degrees**,
        counter-clockwise from the x-axis.  When provided together with
        ``aniso_ratio``, spatial coordinates are transformed via
        :func:`~stamps.stamps.stats.dependence.anisotropy.aniso2iso` before
        distance computation.  Ignored for the temporal axis.
    aniso_ratio : float or None
        Anisotropy ratio in ``(0, 1]``.  ``1.0`` means isotropy.

    Notes
    -----
    Empty coordinate arrays (size 0) are handled gracefully: the returned
    matrices have the correct shape but contain no finite elements.

    The ``isspacetime`` helper inspects *models* to determine the correct
    distance formulation (spatial-only, non-separable ST, or separable ST).

    When ``aniso_angle`` and ``aniso_ratio`` are provided the 2-D spatial
    component of the coordinates is transformed by :func:`aniso2iso` before
    computing Euclidean distances.  This implements geometric anisotropy
    consistently across all model types.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.general.coord2K import coord2K
    >>> c1 = np.array([[10., 15.], [6., 4.], [-14., -6.], [8., -4.]])
    >>> c2 = np.array([[5., -5.], [12., 8.], [0., -8.], [6., 10.]])
    >>> models = ['exponentialC']
    >>> params = [(10., 7.)]
    >>> Ki_sum, Ki = coord2K(c1, c2, models, params)
    >>> Ki_sum.shape
    (4, 4)

    References
    ----------
    Journel, A.G. & Huijbregts, C.J. (1978). *Mining Geostatistics*.
        Academic Press, London. Section I.4 (anisotropic coordinate
        transformation applied before distance computation).
    Christakos, G. (1992). *Random Field Models in Earth Sciences*.
        Academic Press. Chapter 3 (covariance matrix construction).

    ### Original Reference ###
    Formats of covariance model and parameters

    isST / isSTsep:
        models: ['exponentialC','exponentialC','...']
        params: [(3,None), (21.9, 35.8)]
    not isST:
        models: ["gaussian", "exponential", "..."]
        params: [(sill1, bs1), (sill2, bs2)]
    """
    if c1.size == 0 or c2.size == 0:
        Ki = []
        for model in models:
            Ki.append(np.array([]).reshape((c1.shape[0], c2.shape[0])))
        return sum(Ki), Ki

    # Apply anisotropic coordinate transform to the spatial component before
    # computing distances. For ST coordinates (nd=3), only columns 0:2 are
    # spatial; column 2 is time and is left unchanged.
    if aniso_angle is not None and aniso_ratio is not None:
        from ..stats.dependence.anisotropy import aniso2iso as _aniso2iso
        nd = c1.shape[1]
        if nd >= 2:
            c1 = c1.copy()
            c2 = c2.copy()
            c1[:, 0:2] = _aniso2iso(c1[:, 0:2], aniso_angle, aniso_ratio)
            c2[:, 0:2] = _aniso2iso(c2[:, 0:2], aniso_angle, aniso_ratio)

    isST, isSTsep, model_res = isspacetime(models)
    if isST:
        if isSTsep:
            modelS, modelT = model_res
            dist_s = coord2dist(c1[:, 0:2], c2[:, 0:2])
            dist_t = coord2dist(c1[:, 2:3], c2[:, 2:3])
            Ki = []
            for model_s, model_t, param_i in zip(modelS, modelT, params):
                sill, param_s, param_t = param_i
                model_s = get_model(model_s)
                model_t = get_model(model_t)
                Ki.append(
                    sill * model_s(dist_s, 1., param_s) * model_t(dist_t, 1., param_t))
            return sum(Ki), Ki  # K, KK in matlab
        else:
            (modelS,) = model_res
            dist_s = coord2dist(c1[:, 0:2], c2[:, 0:2])
            dist_t = coord2dist(c1[:, 2:3], c2[:, 2:3])
            Ki = []
            for model_s, param_i in zip(modelS, params):
                sill, param_s, s_t_ratio = param_i
                model_s = get_model(model_s)
                Ki.append(
                    sill * model_s(dist_s + s_t_ratio * dist_t, 1., param_s))
            return sum(Ki), Ki  # K, KK in matlab
    else:
        Ki = []
        dist_s = coord2dist(c1, c2)
        (modelS,) = model_res
        for model_s, param_i in zip(modelS, params):
            if len(param_i) == 1:        # nugget: only sill, no range
                sill = param_i[0]
                param_s = [None]
            else:                        # spatial model: sill + range param(s)
                sill, param_s = param_i[0], param_i[1]
            model_s = get_model(model_s)
            Ki.append(sill * model_s(dist_s, 1., param_s))
        return sum(Ki), Ki  # K, KK in matlab

def coord2Ksplit(c1_split, c2split, models, params,
                 aniso_angle=None, aniso_ratio=None):
    """Compute a block-structured covariance matrix for partitioned coordinate sets.

    Applies ``coord2K`` to every pair of coordinate blocks from two lists,
    producing a 2-D list (block matrix) of covariance sub-matrices.  This
    is used in BME estimation to organise the covariance system for
    estimation points, hard-data locations, and soft-data locations into a
    structured block form before assembling the full kriging system.

    Parameters
    ----------
    c1_split : list of np.ndarray or None
        List of *m* coordinate arrays, each of shape ``(ni, nd)``, or
        ``None`` for absent blocks (e.g., when a dataset type is empty).
    c2split : list of np.ndarray or None
        List of *n* coordinate arrays.
    models : list of str
        Covariance model names (passed directly to ``coord2K``).
    params : list of tuple
        Covariance parameters (passed directly to ``coord2K``).

    Returns
    -------
    sum_k_split : list of list of (np.ndarray or None)
        *m* × *n* nested list of total covariance sub-matrices.
        ``sum_k_split[i][j]`` is the covariance matrix between
        ``c1_split[i]`` and ``c2split[j]`` (``None`` if either block is
        ``None``).
    ki_split : list of list of (list or None)
        *m* × *n* nested list of individual component covariance sub-matrices
        (parallel to *sum_k_split*).

    Notes
    -----
    In the BME workflow, the standard call partitions coordinates as::

        c1_split = [ck, ch, cs]   # estimation, hard, soft
        c2split  = [ck, ch, cs]

    producing a 3 × 3 block covariance structure.  Blocks involving absent
    data types (e.g., ``cs = None``) are stored as ``None`` and skipped
    during assembly.

    The layout of the 2-D list is row-major:

    .. code-block:: text

        [ [K(c1a,c2a), K(c1a,c2b), K(c1a,c2c)],
          [K(c1b,c2a), K(c1b,c2b), K(c1b,c2c)],
          [K(c1c,c2a), K(c1c,c2b), K(c1c,c2c)] ]

    ### Original Reference ###
    split dataset for estimated/hard/soft data split.

    c1_split  list  list of m 2D numpy array of S/T coordinates.
    c2_split  list  list of n 2D numpy array of S/T coordinates.

    return
    sumK  2D list  m by n 2D list of covariances.
    Ki    list     m by n 2D list of nested covariances.
    """
    sum_k_split = []
    ki_split = []
    for c1_i in c1_split:
        sum_k_j = []
        ki_split_j = []
        for c2_j in c2split:
            if c1_i is not None and c2_j is not None:
                sum_k, ki = coord2K(c1_i, c2_j, models, params,
                                    aniso_angle=aniso_angle,
                                    aniso_ratio=aniso_ratio)
                sum_k_j.append(sum_k)
                ki_split_j.append(ki)
            else:
                sum_k_j.append(None)
                ki_split_j.append(None)
        sum_k_split.append(sum_k_j)
        ki_split.append(ki_split_j)
        
    return sum_k_split, ki_split

def coord2Kcombine(sum_k_split):
    """Reassemble a block covariance structure produced by ``coord2Ksplit``.

    Concatenates the non-None sub-matrices in *sum_k_split* (output of
    ``coord2Ksplit``) column-wise within each row and then row-wise across
    rows, reconstructing the full covariance matrix as a single 2-D array.

    Parameters
    ----------
    sum_k_split : list of list of (np.ndarray or None)
        Block covariance matrix as returned by ``coord2Ksplit``.

    Returns
    -------
    output : np.ndarray
        Full covariance matrix assembled from the non-None blocks via
        ``np.hstack`` (within rows) and ``np.vstack`` (across rows).
        Returns an empty list if all blocks are ``None``.

    Notes
    -----
    This is the inverse of the splitting operation: after modifying
    individual blocks (e.g., adding soft-data variance to the diagonal),
    ``coord2Kcombine`` can restore the monolithic matrix for use in the
    linear kriging system.
    """
    output = []
    for i in sum_k_split:
        r = [j for j in i if j is not None]
        if r:
            output.append(np.hstack(r))
    if output:
        output = np.vstack(output)
    return output
