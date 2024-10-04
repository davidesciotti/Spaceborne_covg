import warnings
import numpy as np
from scipy.interpolate import interp1d
import itertools
import os
import time
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.integrate import simpson
import pymaster as nmt
import healpy as hp
from tqdm import tqdm

DEG2_IN_SPHERE = 4 * np.pi * (180 / np.pi)**2

from scipy.integrate import simpson


def nmt_gaussian_cov_to_dict(cl_tt, cl_te, cl_ee, cl_tb, cl_eb, cl_bb, coupled, cw, w00, w02, w22, nbl_4covnmt):

    # * NOTE: the order of the arguments (in particular for the cls) is the following
    # * spin_a1, spin_a2, spin_b1, spin_b2,
    # * cla1b1, cla1b2, cla2b1, cla2b2
    # * the order of the output dimensions depends on the order of the input list. See below:
    # * [cl_te, cl_tb] - > TE=0, TB=1
    # * covar_TT_TE = covar_00_02[:, 0, :, 0]x
    # * covar_TT_TB = covar_00_02[:, 0, :, 1]
    # The next few lines show how to extract the covariance matrices
    # for different spin combinations.
    covar_00_00 = nmt.gaussian_covariance(cw,
                                        0, 0, 0, 0,  # Spins of the 4 fields
                                        [cl_tt],  # TT
                                        [cl_tt],  # TT
                                        [cl_tt],  # TT
                                        [cl_tt],  # TT
                                        coupled=coupled,
                                        wa=w00, wb=w00).reshape([nbl_4covnmt, 1,
                                                                nbl_4covnmt, 1])
    covar_TT_TT = covar_00_00[:, 0, :, 0]

    # TODO start - check this better - still new
    covar_00_02 = nmt.gaussian_covariance(cw,
                                        0, 0, 0, 2,  # Spins of the 4 fields
                                        [cl_tt],  # TT
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_tt],  # TT
                                        [cl_te, cl_tb],  # TE, TB
                                        coupled=coupled,
                                        wa=w00, wb=w02).reshape([nbl_4covnmt, 1,
                                                                nbl_4covnmt, 2])
    covar_TT_TE = covar_00_02[:, 0, :, 0]
    covar_TT_TB = covar_00_02[:, 0, :, 1]
    # TODO end - check this better - still new

    covar_02_02 = nmt.gaussian_covariance(cw,
                                        0, 2, 0, 2,  # Spins of the 4 fields
                                        [cl_tt],  # TT
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # ET, BT
                                        [cl_ee, cl_eb,
                                            cl_eb, cl_bb],  # EE, EB, BE, BB
                                        coupled=coupled,
                                        wa=w02, wb=w02).reshape([nbl_4covnmt, 2,
                                                                nbl_4covnmt, 2])
    covar_TE_TE = covar_02_02[:, 0, :, 0]
    covar_TE_TB = covar_02_02[:, 0, :, 1]
    covar_TB_TE = covar_02_02[:, 1, :, 0]
    covar_TB_TB = covar_02_02[:, 1, :, 1]

    covar_00_22 = nmt.gaussian_covariance(cw,
                                        0, 0, 2, 2,  # Spins of the 4 fields
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # TE, TB
                                        coupled=coupled,
                                        wa=w00, wb=w22).reshape([nbl_4covnmt, 1,
                                                                nbl_4covnmt, 4])
    covar_TT_EE = covar_00_22[:, 0, :, 0]
    covar_TT_EB = covar_00_22[:, 0, :, 1]
    covar_TT_BE = covar_00_22[:, 0, :, 2]
    covar_TT_BB = covar_00_22[:, 0, :, 3]

    covar_02_22 = nmt.gaussian_covariance(cw,
                                        0, 2, 2, 2,  # Spins of the 4 fields
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_ee, cl_eb,
                                            cl_eb, cl_bb],  # EE, EB, BE, BB
                                        [cl_ee, cl_eb,
                                            cl_eb, cl_bb],  # EE, EB, BE, BB
                                        coupled=coupled,
                                        wa=w02, wb=w22).reshape([nbl_4covnmt, 2,
                                                                nbl_4covnmt, 4])
    covar_TE_EE = covar_02_22[:, 0, :, 0]
    covar_TE_EB = covar_02_22[:, 0, :, 1]
    covar_TE_BE = covar_02_22[:, 0, :, 2]
    covar_TE_BB = covar_02_22[:, 0, :, 3]
    covar_TB_EE = covar_02_22[:, 1, :, 0]
    covar_TB_EB = covar_02_22[:, 1, :, 1]
    covar_TB_BE = covar_02_22[:, 1, :, 2]
    covar_TB_BB = covar_02_22[:, 1, :, 3]

    covar_22_22 = nmt.gaussian_covariance(cw,
                                        2, 2, 2, 2,  # Spins of the 4 fields
                                        [cl_ee, cl_eb,
                                        cl_eb, cl_bb],  # EE, EB, BE, BB
                                        [cl_ee, cl_eb,
                                            cl_eb, cl_bb],  # EE, EB, BE, BB
                                        [cl_ee, cl_eb,
                                            cl_eb, cl_bb],  # EE, EB, BE, BB
                                        [cl_ee, cl_eb,
                                            cl_eb, cl_bb],  # EE, EB, BE, BB
                                        coupled=coupled,
                                        wa=w22, wb=w22).reshape([nbl_4covnmt, 4,
                                                                nbl_4covnmt, 4])

    covar_EE_EE = covar_22_22[:, 0, :, 0]
    covar_EE_EB = covar_22_22[:, 0, :, 1]
    covar_EE_BE = covar_22_22[:, 0, :, 2]
    covar_EE_BB = covar_22_22[:, 0, :, 3]
    covar_EB_EE = covar_22_22[:, 1, :, 0]
    covar_EB_EB = covar_22_22[:, 1, :, 1]
    covar_EB_BE = covar_22_22[:, 1, :, 2]
    covar_EB_BB = covar_22_22[:, 1, :, 3]
    covar_BE_EE = covar_22_22[:, 2, :, 0]
    covar_BE_EB = covar_22_22[:, 2, :, 1]
    covar_BE_BE = covar_22_22[:, 2, :, 2]
    covar_BE_BB = covar_22_22[:, 2, :, 3]
    covar_BB_EE = covar_22_22[:, 3, :, 0]
    covar_BB_EB = covar_22_22[:, 3, :, 1]
    covar_BB_BE = covar_22_22[:, 3, :, 2]
    covar_BB_BB = covar_22_22[:, 3, :, 3]

    # build dict with relevant covmats
    cov_nmt_dict = {
        'LLLL': covar_EE_EE,
        'GLLL': covar_TE_EE,
        'GGLL': covar_TT_EE,
        'GLGL': covar_TE_TE,
        'GGGL': covar_TT_TE,
        'GGGG': covar_TT_TT,
    }
    return cov_nmt_dict


def bin_cell(ells_in, ells_out, ells_out_edges, cls_in, weights, which_binning, ells_eff=None):
    """
    Bin the input power spectrum into the output bins.
    :param ells_in: array of input ells
    :param ells_out: array of output ells
    :param cls_in: array of input power spectrum
    :param weights: array of weights for the input power spectrum
    :return: array of binned power spectrum
    """
    if weights is None:
        weights = np.ones_like(ells_in)
    if len(ells_in) != len(cls_in):
        raise ValueError('ells_in and cls_in must have the same length')
    if len(ells_in) != len(weights):
        raise ValueError('ells_in and weights must have the same length')
    if np.any(ells_out < ells_in[0]) or np.any(ells_out > ells_in[-1]):
        raise ValueError('ells_out must be within the range of ells_in')
    if np.any(ells_out[1:] < ells_out[:-1]):
        raise ValueError('ells_out must be monotonically increasing')

    assert len(cls_in) == len(ells_in), "ells_in must be the same length as the covariance matrix"
    assert len(ells_out) == len(ells_out_edges) - 1, "ells_out must be the same length as the number of edges - 1"

    binned_cls = np.zeros(len(ells_out))
    spline = CubicSpline(ells_in, cls_in)

    ells_edges_low = ells_out_edges[:-1]
    ells_edges_high = ells_out_edges[1:]

    # Loop over the output bins
    for ell_idx in range(len(ells_out)):

        # Get ell min/max for the current bins
        ell_min = ells_edges_low[ell_idx]
        ell_max = ells_edges_high[ell_idx]

        # this mask returns a bool array True at the ells_in indices satisfying the condition
        ell_bool_mask = (ell_min <= ells_in) & (ells_in < ell_max)
        ell_masked_idxs = np.nonzero(ell_bool_mask)[0]

        # isolate the relevant ranges of ell values from the original ells_in grid, weights and cov
        ells_in_masked = ells_in[ell_masked_idxs]
        cls_masked = cls_in[ell_masked_idxs]

        if weights.shape == (len(ells_eff), len(ells_in)):
            ells_eff_idx = np.argmin(np.abs(ells_eff - ells_out[ell_idx]))
            weights_masked = weights[ells_eff_idx, ell_masked_idxs]
        elif weights.shape == (len(ells_in),):
            weights_masked = weights[ell_masked_idxs]

        # Calculate the bin widths
        if weights is None:
            delta_ell = ell_max - ell_min
            assert delta_ell == np.sum(weights_masked), "The weights must sum to the bin width"


        # Option 1: use the original grid for integration and no weights
        if which_binning == 'integral':
            integral = simpson(y=cls_masked * weights_masked, x=ells_in_masked)
            binned_cls[ell_idx] = integral / np.sum(weights_masked)
        
        elif which_binning == 'mean':
            binned_cls[ell_idx] = np.mean(cls_masked)

        else:
            raise ValueError('which_binning should be "mean" or "integral"')

        # # Option 2: create fine grids for integration over the ell ranges (GIVES GOOD RESULTS ONLY FOR nsteps=delta_ell!)
        # ell_fine = np.linspace(ell_min, ell_max, 50)
        # cls_interp = spline(ell_fine)

        # # Perform simpson integration over the ell ranges
        # integral = simpson(y=cls_interp * ell_fine, x=ell_fine)
        # binned_cls[ell_idx] = integral / (np.sum(ell_fine))

    return binned_cls


def bin_2d_matrix(cov, ells_in, ells_out, ells_out_edges, which_binning, weights):

    assert cov.shape[0] == cov.shape[1] == len(ells_in), "ells_in must be the same length as the covariance matrix"
    assert len(ells_out) == len(ells_out_edges) - 1, "ells_out must be the same length as the number of edges - 1"

    if weights is None:
        weights = np.ones_like(ells_in)

    assert len(weights) == len(ells_in)

    binned_cov = np.zeros((len(ells_out), len(ells_out)))
    # cov_interp_func = RectBivariateSpline(ells_in, ells_in, cov)

    ells_edges_low = ells_out_edges[:-1]
    ells_edges_high = ells_out_edges[1:]

    # Loop over the output bins
    for ell1_idx, _ in tqdm(enumerate(ells_out)):
        for ell2_idx, _ in enumerate(ells_out):

            # Get ell min/max for the current bins
            ell1_min = ells_edges_low[ell1_idx]
            ell1_max = ells_edges_high[ell1_idx]
            ell2_min = ells_edges_low[ell2_idx]
            ell2_max = ells_edges_high[ell2_idx]

            # this mask returns a bool array True at the ells_in indices satisfying the condition
            ell1_bool_mask = (ell1_min <= ells_in) & (ells_in < ell1_max)
            ell2_bool_mask = (ell2_min <= ells_in) & (ells_in < ell2_max)
            ell1_masked_idxs = np.nonzero(ell1_bool_mask)[0]
            ell2_masked_idxs = np.nonzero(ell2_bool_mask)[0]

            # isolate the relevant ranges of ell values from the original ells_in grid, weights and cov
            ell1_masked = ells_in[ell1_masked_idxs]
            ell2_masked = ells_in[ell2_masked_idxs]
            weights1_masked = weights[ell1_masked_idxs]
            weights2_masked = weights[ell2_masked_idxs]
            cov_masked = cov[np.ix_(ell1_masked_idxs, ell2_masked_idxs)]            

            # Calculate the bin widths
            if weights is None:
                delta_ell = ell1_max - ell1_min
                assert delta_ell == np.sum(weights1_masked), "The weights must sum to the bin width"


            if which_binning == 'integral':
                # Option 1a: use the original grid for integration and the ell values as weights
                partial_integral = simpson(
                    y=cov_masked * weights1_masked[:, None] * weights2_masked[None, :], x=ell2_masked, axis=1)
                integral = simpson(y=partial_integral, x=ell1_masked)
                binned_cov[ell1_idx, ell2_idx] = integral / (np.sum(weights1_masked) * np.sum(weights2_masked))

            elif which_binning == 'mean':
                # ! important note: taking the mean in 2D is equivalent to taking the mean and dividing by delta_ell in 1D
                # binned_cov[ell1_idx, ell2_idx] = np.mean(np.diag(cov_masked)) / delta_ell
                binned_cov[ell1_idx, ell2_idx] = np.mean(cov_masked) 

            else:
                raise ValueError('which_binning should be "mean" or "integral"')
            

            # # Option 2: create fine grids for integration over the ell ranges (GIVES GOOD RESULTS ONLY FOR nsteps=delta_ell!)
            # ell_fine_1 = np.linspace(ell1_min, ell1_max, 50)
            # ell_fine_2 = np.linspace(ell2_min, ell2_max, 50)

            # # Evaluate the spline on the fine grids
            # ell1_fine_xx, ell2_fine_yy = np.meshgrid(ell_fine_1, ell_fine_2, indexing='ij')
            # cov_interp_vals = cov_interp_func(ell_fine_1, ell_fine_2)

            # # Perform simpson integration over the ell ranges
            # partial_integral = simpson(y=cov_interp_vals * ell1_fine_xx * ell2_fine_yy, x=ell_fine_2, axis=1)
            # integral = simpson(y=partial_integral, x=ell_fine_1)
            # # Normalize by the bin areas
            # binned_cov[ell1_idx, ell2_idx] = integral / (np.sum(ell_fine_1) * np.sum(ell_fine_2))

    return binned_cov


def percent_diff(array_1, array_2, abs_value=False):
    diff = (array_1 / array_2 - 1) * 100
    if abs_value:
        return np.abs(diff)
    else:
        return diff


def matshow(array, title="title", log=True, abs_val=False, threshold=None, only_show_nans=False, matshow_kwargs={}):
    """
    :param array:
    :param title:
    :param log:
    :param abs_val:
    :param threshold: if None, do not mask the values; otherwise, keep only the elements above the threshold
    (i.e., mask the ones below the threshold)
    :return:
    """

    if only_show_nans:
        warnings.warn('only_show_nans is True, better switch off log and abs_val for the moment')
        # Set non-NaN elements to 0 and NaN elements to 1
        array = np.where(np.isnan(array), 1, 0)
        title += ' (only NaNs shown)'

    # the ordering of these is important: I want the log(abs), not abs(log)
    if abs_val:  # take the absolute value
        array = np.abs(array)
        title = 'abs ' + title
    if log:  # take the log
        array = np.log10(array)
        title = 'log10 ' + title

    if threshold is not None:
        array = np.ma.masked_where(array < threshold, array)
        title += f" \n(masked below {threshold} \%)"

    plt.matshow(array, **matshow_kwargs)
    plt.colorbar()
    plt.title(title)
    plt.show()


def deg2_to_fsky(survey_area_deg2):
    # deg2_in_sphere = 41252.96  # deg^2 in a spere
    # return survey_area_deg2 / deg2_in_sphere

    f_sky = survey_area_deg2 * (np.pi / 180) ** 2 / (4 * np.pi)
    return f_sky


def generate_polar_cap(area_deg2, nside):

    if np.isclose(area_deg2, DEG2_IN_SPHERE, rtol=1e-5, atol=0):
        print('area_deg2 is very close to the full sky, returning a full sky mask')
        mask = np.ones(hp.pixelfunc.nside2npix(nside))
        return mask

    print(f'Generating a polar cap mask with area {area_deg2} deg2 and resolution nside {nside}')

    # Expected sky fraction
    fsky_expected = deg2_to_fsky(area_deg2)
    print(f"Expected f_sky: {fsky_expected}")

    # Convert the area to radians squared for the angular radius calculation
    area_rad2 = area_deg2 * (np.pi / 180)**2

    # The area of a cap is given by A = 2*pi*(1 - cos(theta)),
    # so solving for theta gives us the angular radius of the cap
    theta_cap_rad = np.arccos(1 - area_rad2 / (2 * np.pi))

    # Convert the angular radius to degrees for visualization
    theta_cap_deg = np.degrees(theta_cap_rad)
    print(f"Angular radius of the cap in degrees: {theta_cap_deg}")

    # Calculate the corresponding nside for the HEALPix map
    # The resolution parameter nside should be chosen so lmax ~ 3*nside

    # Create an empty mask with the appropriate number of pixels for our nside
    mask = np.zeros(hp.nside2npix(nside))

    # Find the pixels within our cap
    # Vector pointing to the North Pole (θ=0, φ can be anything since θ=0 defines the pole)
    vec = hp.ang2vec(0, 0)
    pixels_in_cap = hp.query_disc(nside, vec, theta_cap_rad)

    # Set the pixels within the cap to 1
    mask[pixels_in_cap] = 1

    # Calculate the actual sky fraction of the generated mask
    fsky_actual = np.sum(mask) / len(mask)
    print(f"Actual f_sky from the mask: {fsky_actual}")

    return mask


def generate_ind(triu_tril_square, row_col_major, size):
    """
    Generates a list of indices for the upper triangular part of a matrix
    :param triu_tril_square: str. if 'triu', returns the indices for the upper triangular part of the matrix.
    If 'tril', returns the indices for the lower triangular part of the matrix
    If 'full_square', returns the indices for the whole matrix
    :param row_col_major: str. if True, the indices are returned in row-major order; otherwise, in column-major order
    :param size: int. size of the matrix to take the indices of
    :return: list of indices
    """
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'
    assert triu_tril_square in ['triu', 'tril', 'full_square'], 'triu_tril_square must be either "triu", "tril" or ' \
                                                                '"full_square"'

    if triu_tril_square == 'triu':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i, size)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i + 1)]
    elif triu_tril_square == 'tril':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i + 1)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i, size)]
    elif triu_tril_square == 'full_square':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(size)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(size)]
    else:
        raise ValueError('triu_tril_square must be either "triu", "tril" or "full_square"')

    return np.asarray(ind)


def build_full_ind(triu_tril, row_col_major, size):
    """
    Builds index array mapping the redshift indices zi, zj into the index of the independent redshift pairs, for all probes
    """

    assert triu_tril in ['triu', 'tril'], 'triu_tril must be either "triu" or "tril"'
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    LL_columns = np.zeros((zpairs_auto, 2))
    GL_columns = np.hstack((np.ones((zpairs_cross, 1)), np.zeros((zpairs_cross, 1))))
    GG_columns = np.ones((zpairs_auto, 2))

    LL_columns = np.hstack((LL_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)
    GL_columns = np.hstack((GL_columns, generate_ind('full_square', row_col_major, size))).astype(int)
    GG_columns = np.hstack((GG_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)

    ind = np.vstack((LL_columns, GL_columns, GG_columns))

    assert ind.shape[0] == zpairs_3x2pt, 'ind has the wrong number of rows'

    return ind


def get_zpairs(zbins):
    zpairs_auto = int((zbins * (zbins + 1)) / 2)  # = 55 for zbins = 10, cast it as int
    zpairs_cross = zbins ** 2
    zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross
    return zpairs_auto, zpairs_cross, zpairs_3x2pt


###############################################################################
#################### COVARIANCE MATRIX COMPUTATION ############################
###############################################################################
# TODO unify these 3 into a single function
# TODO workaround for start_index, stop_index (super easy)


def covariance(nbl, npairs, start_index, stop_index, Cij, noise, l_lin, delta_l, fsky, ind):
    # create covariance array
    covariance = np.zeros((nbl, nbl, npairs, npairs))
    # compute cov(ell, p, q)
    for ell in range(nbl):
        for p in range(start_index, stop_index):
            for q in range(start_index, stop_index):
                covariance[ell, ell, p - start_index, q - start_index] = \
                    ((Cij[ell, ind[p, 2], ind[q, 2]] + noise[ind[p, 0], ind[q, 0], ind[p, 2], ind[q, 2]]) *
                     (Cij[ell, ind[p, 3], ind[q, 3]] + noise[ind[p, 1], ind[q, 1], ind[p, 3], ind[q, 3]]) +
                     (Cij[ell, ind[p, 2], ind[q, 3]] + noise[ind[p, 0], ind[q, 1], ind[p, 2], ind[q, 3]]) *
                     (Cij[ell, ind[p, 3], ind[q, 2]] + noise[ind[p, 1], ind[q, 0], ind[p, 3], ind[q, 2]])) / \
                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return covariance


def covariance_einsum(cl_5d, noise_5d, f_sky, ell_values, delta_ell, return_only_diagonal_ells=False):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as noise_5d

    :param cl_5d:
    :param noise_5d:
    :param f_sky:
    :param ell_values:
    :param delta_ell:
    :param return_only_diagonal_ells:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.

    """
    assert cl_5d.shape[0] == 1 or cl_5d.shape[0] == 2, 'This funcion only works with 1 or two probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert noise_5d.shape == cl_5d.shape, 'noise_5d must have shape the same shape as cl_5d, although there ' \
                                          'is no ell dependence'

    nbl = cl_5d.shape[2]

    prefactor = 1 / ((2 * ell_values + 1) * f_sky * delta_ell)

    # considering ells off-diagonal (wrong for Gauss: I am not implementing the delta)
    # term_1 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # term_2 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # cov_10d = np.einsum('ABCDLMijkl, L -> ABCDLMijkl', term_1 + term_2, prefactor)

    # considering only ell diagonal
    term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    term_2 = np.einsum('ADLil, BCLjk -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    cov_9d = np.einsum('ABCDLijkl, L -> ABCDLijkl', term_1 + term_2, prefactor)

    if return_only_diagonal_ells:
        warnings.warn('return_only_diagonal_ells is True, the array will be 9-dimensional, potentially causing '
                      'problems when reshaping or summing to cov_SSC arrays')
        return cov_9d

    n_probes = cov_9d.shape[0]
    zbins = cov_9d.shape[-1]
    cov_10d = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    cov_10d[:, :, :, :, np.arange(nbl), np.arange(nbl), ...] = cov_9d[:, :, :, :, np.arange(nbl), ...]

    return cov_10d


def covariance_nmt(cl_5d, noise_5d, workspace_path, mask_path):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as noise_5d

    :param cl_5d:
    :param noise_5d:
    :param f_sky:
    :param ell_values:
    :param delta_ell:
    :param return_only_diagonal_ells:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.

    """
    assert cl_5d.shape[0] == 1 or cl_5d.shape[0] == 2, 'This funcion only works with 1 or two probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert noise_5d.shape == cl_5d.shape, 'noise_5d must have shape the same shape as cl_5d, although there ' \
                                          'is no ell dependence'

    # Transform array Cl's and noise to dictionnary to ease the loop over probes
    cl_dic, keys = Cl5D_to_ClDic(cl_5d)
    noise_dic, keys = Cl5D_to_ClDic(noise_5d)

    # Get symetric Cls for permutations in the covariance loop
    for key in keys:
        probeA, probeB = key.split('-')
        cl_dic['-'.join([probeB, probeA])] = cl_dic[key]
        noise_dic['-'.join([probeB, probeA])] = noise_dic[key]

    # Get the mask and make it a field
    print('Get mask')
    mask = hp.read_map(mask_path)
    fmask = nmt.NmtField(mask, [mask])

    # Get the workspace
    print('Get workspace')
    workspace = nmt.NmtWorkspace()
    workspace.read_from(workspace_path)
    nbl = workspace.get_bandpower_windows().shape[1]

    # Initialise covariance workspace
    print('Initialise covariance workspace')
    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(fmask, fmask)

    # Initialise the covariance matrix
    ncl = len(keys)
    covmat = np.zeros((int(ncl * nbl), int(ncl * nbl)))

    # Covariance computation loop
    print('Start loop')
    for (idx1, key1), (idx2, key2) in itertools.combinations_with_replacement(enumerate(keys), 2):
        probeA, probeB = key1.split('-')
        probeC, probeD = key2.split('-')

        covmat[idx1 * nbl:(idx1 + 1) * nbl, idx2 * nbl:(idx2 + 1) * nbl] =\
            nmt.gaussian_covariance(cov_workspace, 0, 0, 0, 0,  # Spins of the 4 fields
                                    [cl_dic['-'.join([probeA, probeC])] +\
                                     noise_dic['-'.join([probeA, probeC])]],
                                    [cl_dic['-'.join([probeB, probeC])] +\
                                     noise_dic['-'.join([probeB, probeC])]],
                                    [cl_dic['-'.join([probeA, probeD])] +\
                                     noise_dic['-'.join([probeA, probeD])]],
                                    [cl_dic['-'.join([probeB, probeD])] +\
                                     noise_dic['-'.join([probeB, probeD])]],
                                    workspace, wb=workspace)

    # Symmetric of the matrix
    covmat = covmat + covmat.T - np.diag(covmat.diagonal())

    # TODO transform 2D matrix to 10D so that it can enter ordering transformation functions.
    # For now the ordering is probe_zpair_ell

    return covmat


def Cl5D_to_ClDic(cl_5d):
    """
    Transform a 5D Cl array to a dictionary with different probes as keys.

    Parameters:
    cl_5d (ndarray): 5D array of Cl data. The dimensions should be arranged as follows: (n_l, n_cls, n_cls, n_bins, n_bins), where:
                      - n_l: Number of angular multipoles.
                      - n_cls: Number of Cl spectra (e.g., auto or cross spectra).
                      - n_bins: Number of bins or redshift slices.

    Returns:
    dict: Dictionary with keys representing different probes and values as Cl data.
         The keys are formatted as 'XY-i-j', where 'XY' represents the probe type ('GC', 'WL', or 'GGL'),
         and 'i', 'j' represent the bin indices.
    list: List of keys in the dictionary.
    """
    nbz = cl_5d.shape[3]  # Number of bins or redshift slices
    indices = range(nbz)

    # Define iteration schemes for different probes
    iter_probe = {'GC': itertools.combinations_with_replacement(indices, 2),
                  'WL': itertools.combinations_with_replacement(indices, 2),
                  'GGL': itertools.product(indices, repeat=2)}

    # Mapping between probe type and key format
    keymap = {'GC': ('D', 'D'),
              'WL': ('G', 'G'),
              'GGL': ('D', 'G')}

    # Mapping between probe type and indices for accessing the Cl data
    keymap_davide = {'GC': (1, 1),
                     'WL': (0, 0),
                     'GGL': (0, 1)}

    # Create the dictionary and the list of keys using dictionary comprehension and list comprehension
    dic = {}
    for probe in ['WL', 'GGL', 'GC']:
        for i, j in iter_probe[probe]:
            k = '{}{}-{}{}'.format(keymap[probe][0], i, keymap[probe][1], j)
            if probe == 'GGL':
                dic[k] = cl_5d[keymap_davide[probe][0], keymap_davide[probe][1], :, j, i]
            else:
                dic[k] = cl_5d[keymap_davide[probe][0], keymap_davide[probe][1], :, i, j]
    keys = list(dic.keys())

    return dic, keys


def linear_lmin_binning(NSIDE, lmin, bw):
    """
    Generate a linear binning scheme based on a minimum multipole 'lmin' and bin width 'bw'.

    Parameters:
    -----------
    NSIDE : int
        The NSIDE parameter of the HEALPix grid.

    lmin : int
        The minimum multipole to start the binning.

    bw : int
        The bin width, i.e., the number of multipoles in each bin.

    Returns:
    --------
    nmt_bins
        A binning scheme object defining linearly spaced bins starting from 'lmin' with
        a width of 'bw' multipoles.

    Notes:
    ------
    This function generates a binning scheme for the pseudo-Cl power spectrum estimation
    using the Namaster library. It divides the multipole range from 'lmin' to 2*NSIDE
    into bins of width 'bw'.

    Example:
    --------
    # Generate a linear binning scheme for an NSIDE of 64, starting from l=10, with bin width of 20
    bin_scheme = linear_lmin_binning(NSIDE=64, lmin=10, bw=20)
    """
    lmax = 2 * NSIDE
    nbl = (lmax - lmin) // bw + 1
    elli = np.zeros(nbl, int)
    elle = np.zeros(nbl, int)

    for i in range(nbl):
        elli[i] = lmin + i * bw
        elle[i] = lmin + (i + 1) * bw

    b = nmt.NmtBin.from_edges(elli, elle)
    return b


def coupling_matrix(bin_scheme, mask, wkspce_name):
    """
    Compute the mixing matrix for coupling spherical harmonic modes using
    the provided binning scheme and mask.

    Parameters:
    -----------
    bin_scheme : nmt_bins
        A binning scheme object defining the bins for the coupling matrix.

    mask : nmt_field
        A mask object defining the regions of the sky to include in the computation.

    wkspce_name : str
        The file name for storing or retrieving the computed workspace containing
        the coupling matrix.

    Returns:
    --------
    nmt_workspace
        A workspace object containing the computed coupling matrix.

    Notes:
    ------
    This function computes the coupling matrix necessary for the pseudo-Cl power
    spectrum estimation using the NmtField and NmtWorkspace objects from the
    Namaster library.

    If the workspace file specified by 'wkspce_name' exists, the function reads
    the coupling matrix from the file. Otherwise, it computes the matrix and
    writes it to the file.

    Example:
    --------
    # Generate a linear binning scheme for an NSIDE of 64, starting from l=10, with bin width of 20
    bin_scheme = linear_lmin_binning(NSIDE=64, lmin=10, bw=20)

    # Define the mask
    mask = nmt.NmtField(mask, [mask])

    # Compute the coupling matrix and store it in 'coupling_matrix.bin'
    coupling_matrix = coupling_matrix(bin_scheme, mask, 'coupling_matrix.bin')
    """
    print('Compute the mixing matrix')
    start = time.time()
    fmask = nmt.NmtField(mask, [mask])  # nmt field with only the mask
    w = nmt.NmtWorkspace()
    if os.path.isfile(wkspce_name):
        print('Mixing matrix has already been calculated and is in the workspace file : ', wkspce_name, '. Read it.')
        w.read_from(wkspce_name)
    else:
        print('The file : ', wkspce_name, ' does not exists. Calculating the mixing matrix and writing it.')
        w.compute_coupling_matrix(fmask, fmask, bin_scheme)
        w.write_to(wkspce_name)
    print('Done computing the mixing matrix. It took ', time.time() - start, 's.')
    return w


def cov_10D_dict_to_array(cov_10D_dict, nbl, zbins, n_probes=2):
    """ transforms a dictionary of "shape" [(A, B, C, D)][nbl, nbl, zbins, zbins, zbins, zbins] (where A, B, C, D is a
    tuple of strings, each one being either 'L' or 'G') to a numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)"""
    cov_10D_array = \
        np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))

    LG_idx_dict = {'L': 0, 'G': 1}
    for A, B, C, D in cov_10D_dict.keys():
        cov_10D_array[LG_idx_dict[A], LG_idx_dict[B], LG_idx_dict[C], LG_idx_dict[D], ...] = \
            cov_10D_dict[A, B, C, D]

    return cov_10D_array


def cov_10D_array_to_dict(cov_10D_array, n_probes=2):
    """ transforms a dictionary of "shape" [(A, B, C, D)][nbl, nbl, zbins, zbins, zbins, zbins] (where A, B, C, D is a
    tuple of strings, each one being either 'L' or 'G') to a numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)"""

    # cov_10D_dict = {}
    # for A in ('L', 'G'):
    #     for B in ('L', 'G'):
    #         for C in ('L', 'G'):
    #             for D in ('L', 'G'):
    #                 cov_10D_dict[A, B, C, D] = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))

    cov_10D_dict = {}
    LG_idx_tuple = ('L', 'G')
    for A in range(n_probes):
        for B in range(n_probes):
            for C in range(n_probes):
                for D in range(n_probes):
                    cov_10D_dict[LG_idx_tuple[A], LG_idx_tuple[B], LG_idx_tuple[C], LG_idx_tuple[D]] = \
                        cov_10D_array[A, B, C, D, ...]

    return cov_10D_dict


def cov_SSC(nbl, zpairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    if probe in ["WL", "WA"]:
        shift = 0
    elif probe == "GC":
        shift = zbins
    else:
        raise ValueError('probe must be "WL", "WA" or "GC"')

    cov_SSC = np.zeros((nbl, nbl, zpairs, zpairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(zpairs):
                for q in range(zpairs):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                    cov_SSC[ell1, ell2, p, q] = (Rl[ell1, i, j] * Rl[ell2, k, l] *
                                                 Cij[ell1, i, j] * Cij[ell2, k, l] *
                                                 Sijkl[i + shift, j + shift, k + shift, l + shift])
    cov_SSC /= fsky
    return cov_SSC


def build_Sijkl_dict(Sijkl, zbins):
    # build probe lookup dictionary, to set the right start and stop values
    probe_lookup = {
        'L': {
            'start': 0,
            'stop': zbins
        },
        'G': {
            'start': zbins,
            'stop': 2 * zbins
        }
    }

    # fill Sijkl dictionary
    Sijkl_dict = {}
    for probe_A in ['L', 'G']:
        for probe_B in ['L', 'G']:
            for probe_C in ['L', 'G']:
                for probe_D in ['L', 'G']:
                    Sijkl_dict[probe_A, probe_B, probe_C, probe_D] = \
                        Sijkl[probe_lookup[probe_A]['start']:probe_lookup[probe_A]['stop'],
                              probe_lookup[probe_B]['start']:probe_lookup[probe_B]['stop'],
                              probe_lookup[probe_C]['start']:probe_lookup[probe_C]['stop'],
                              probe_lookup[probe_D]['start']:probe_lookup[probe_D]['stop']]

    return Sijkl_dict


def build_3x2pt_dict(array_3x2pt):
    dict_3x2pt = {}
    if array_3x2pt.ndim == 5:
        dict_3x2pt['L', 'L'] = array_3x2pt[:, 0, 0, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[:, 0, 1, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[:, 1, 0, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[:, 1, 1, :, :]
    elif array_3x2pt.ndim == 4:
        dict_3x2pt['L', 'L'] = array_3x2pt[0, 0, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[0, 1, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[1, 0, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[1, 1, :, :]
    return dict_3x2pt


# ! to be deprecated

def cov_SS_3x2pt_10D_dict_nofsky_old(nbl, cl_3x2pt, Sijkl, zbins, response_3x2pt, probe_ordering):
    """Buil the 3x2pt covariance matrix using a dict for the response, the cls and Sijkl.
    Slightly slower (because of the use of dicts, I think) but much cleaner (no need for multiple if statements).
    """
    #
    # build and/or initialize the dictionaries
    Sijkl_dict = build_Sijkl_dict(Sijkl, zbins)
    cl_3x2pt_dict = build_3x2pt_dict(cl_3x2pt)
    response_3x2pt_dict = build_3x2pt_dict(response_3x2pt)
    cov_3x2pt_SS_10D = {}

    # compute the SS cov only for the relevant probe combinations
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_3x2pt_SS_10D[A, B, C, D] = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
            for ell1 in range(nbl):
                for ell2 in range(nbl):
                    for i in range(zbins):
                        for j in range(zbins):
                            for k in range(zbins):
                                for l in range(zbins):
                                    cov_3x2pt_SS_10D[A, B, C, D][ell1, ell2, i, j, k, l] = \
                                        (response_3x2pt_dict[A, B][ell1, i, j] *
                                         response_3x2pt_dict[C, D][ell2, k, l] *
                                         cl_3x2pt_dict[A, B][ell1, i, j] *
                                         cl_3x2pt_dict[C, D][ell2, k, l] *
                                         Sijkl_dict[A, B, C, D][i, j, k, l])
            print('computing SSC in blocks: working probe combination', A, B, C, D)

    return cov_3x2pt_SS_10D


def cov_G_10D_dict(cl_dict, noise_dict, nbl, zbins, l_lin, delta_l, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically.
    This one works with dictionaries, in particular for the cls and noise arrays.
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.

    This version is faster, it is a wrapper function for covariance_6D_blocks,
    which makes use of jit
    """

    cov_10D_dict = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_10D_dict[A, B, C, D] = cov_GO_6D_blocks(
                cl_dict[A, C], cl_dict[B, D], cl_dict[A, D], cl_dict[B, C],
                noise_dict[A, C], noise_dict[B, D], noise_dict[A, D], noise_dict[B, C],
                nbl, zbins, l_lin, delta_l, fsky)
    return cov_10D_dict


def cov_SS_10D_dict(Cl_dict, Rl_dict, Sijkl_dict, nbl, zbins, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically.
    This one works with dictionaries, in particular for the cls and noise arrays.
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.

    This version is faster, it is a wrapper function for covariance_6D_blocks,
    which makes use of jit
    """

    cov_SS_10D_dict = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_SS_10D_dict[A, B, C, D] = cov_SS_6D_blocks(Rl_dict[A, B], Cl_dict[A, B], Rl_dict[C, D], Cl_dict[C, D],
                                                           Sijkl_dict[A, B, C, D], nbl, zbins, fsky)

    return cov_SS_10D_dict


# This function does mix the indices, but not automatically: it only indicates which ones to use and where
# It can be used for the individual blocks of the 3x2pt (unlike the one above),
# but it has to be called once for each block combination (see cov_blocks_LG_4D
# and cov_blocks_GL_4D)
# best used in combination with cov_10D_dictionary

def cov_GO_6D_blocks(C_AC, C_BD, C_AD, C_BC, N_AC, N_BD, N_AD, N_BC, nbl, zbins, l_lin, delta_l, fsky):
    cov_GO_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                for k in range(zbins):
                    for l in range(zbins):
                        cov_GO_6D[ell, ell, i, j, k, l] = \
                            ((C_AC[ell, i, k] + N_AC[i, k]) *
                             (C_BD[ell, j, l] + N_BD[j, l]) +
                             (C_AD[ell, i, l] + N_AD[i, l]) *
                             (C_BC[ell, j, k] + N_BC[j, k])) / \
                            ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return cov_GO_6D


def cov_SS_6D_blocks(Rl_AB, Cl_AB, Rl_CD, Cl_CD, Sijkl_ABCD, nbl, zbins, fsky):
    """ experimental"""
    cov_SS_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for i in range(zbins):
                for j in range(zbins):
                    for k in range(zbins):
                        for l in range(zbins):
                            cov_SS_6D[ell1, ell2, i, j, k, l] = \
                                (Rl_AB[ell1, i, j] *
                                 Cl_AB[ell1, i, j] *
                                 Rl_CD[ell2, k, l] *
                                 Cl_CD[ell2, k, l] *
                                 Sijkl_ABCD[i, j, k, l])
    cov_SS_6D /= fsky
    return cov_SS_6D


def cov_3x2pt_dict_10D_to_4D(cov_3x2pt_dict_10D, probe_ordering, nbl, zbins, ind_copy, GL_or_LG):
    """
    Takes the cov_3x2pt_10D dictionary, reshapes each A, B, C, D block separately
    in 4D, then stacks the blocks in the right order to output cov_3x2pt_4D
    (which is not a dictionary but a numpy array)

    probe_ordering: e.g. ['L', 'L'], ['G', 'L'], ['G', 'G']]
    """

    ind_copy = ind_copy.copy()  # just to ensure the input ind file is not changed

    # Check that the cross-correlation is coherent with the probe_ordering list
    # this is a weak check, since I'm assuming that GL or LG will be the second
    # element of the datavector
    if GL_or_LG == 'GL':
        assert probe_ordering[1][0] == 'G' and probe_ordering[1][1] == 'L', \
            'probe_ordering[1] should be "GL", e.g. [LL, GL, GG]'
    elif GL_or_LG == 'LG':
        assert probe_ordering[1][0] == 'L' and probe_ordering[1][1] == 'G', \
            'probe_ordering[1] should be "LG", e.g. [LL, LG, GG]'

    # get npairs
    npairs_auto, npairs_cross, npairs_3x2pt = get_zpairs(zbins)

    # construct the ind dict
    ind_dict = {}
    ind_dict['L', 'L'] = ind_copy[:npairs_auto, :]
    ind_dict['G', 'G'] = ind_copy[(npairs_auto + npairs_cross):, :]
    if GL_or_LG == 'LG':
        ind_dict['L', 'G'] = ind_copy[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['G', 'L'] = ind_dict['L', 'G'].copy()  # copy and switch columns
        ind_dict['G', 'L'][:, [2, 3]] = ind_dict['G', 'L'][:, [3, 2]]
    elif GL_or_LG == 'GL':
        ind_dict['G', 'L'] = ind_copy[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['L', 'G'] = ind_dict['G', 'L'].copy()  # copy and switch columns
        ind_dict['L', 'G'][:, [2, 3]] = ind_dict['L', 'G'][:, [3, 2]]

    # construct the npairs dict
    npairs_dict = {}
    npairs_dict['L', 'L'] = npairs_auto
    npairs_dict['L', 'G'] = npairs_cross
    npairs_dict['G', 'L'] = npairs_cross
    npairs_dict['G', 'G'] = npairs_auto

    # initialize the 4D dictionary and list of probe combinations
    cov_3x2pt_dict_4D = {}
    combinations = []

    # make each block 4D and store it with the right 'A', 'B', 'C, 'D' key
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            combinations.append([A, B, C, D])
            cov_3x2pt_dict_4D[A, B, C, D] = cov_6D_to_4D_blocks(cov_3x2pt_dict_10D[A, B, C, D], nbl, npairs_dict[A, B],
                                                                npairs_dict[C, D], ind_dict[A, B], ind_dict[C, D])

    # take the correct combinations (stored in 'combinations') and construct
    # lists which will be converted to arrays
    row_1_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[:3]]
    row_2_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[3:6]]
    row_3_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[6:9]]

    # concatenate the lists to make rows
    row_1 = np.concatenate(row_1_list, axis=3)
    row_2 = np.concatenate(row_2_list, axis=3)
    row_3 = np.concatenate(row_3_list, axis=3)

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = np.concatenate((row_1, row_2, row_3), axis=2)

    return cov_3x2pt_4D


# ! to be deprecated
def symmetrize_ij(cov_6D, zbins):
    warnings.warn('THIS FUNCTION ONLY WORKS IF THE MATRIX TO SYMMETRIZE IS UPPER *OR* LOWER TRIANGULAR, NOT BOTH')
    # TODO thorough check?
    for i in range(zbins):
        for j in range(zbins):
            cov_6D[:, :, i, j, :, :] = cov_6D[:, :, j, i, :, :]
            cov_6D[:, :, :, :, i, j] = cov_6D[:, :, :, :, j, i]
    return cov_6D


# ! this function is new - still to be thouroughly tested
def cov_4D_to_6D(cov_4D, nbl, zbins, probe, ind):
    """transform the cov from shape (nbl, nbl, npairs, npairs)
    to (nbl, nbl, zbins, zbins, zbins, zbins). Not valid for 3x2pt, the total
    shape of the matrix is (nbl, nbl, zbins, zbins, zbins, zbins), not big
    enough to store 3 probes. Use cov_4D functions or cov_6D as a dictionary
    instead,
    """

    assert probe in ['LL', 'GG', 'LG', 'GL'], 'probe must be "LL", "LG", "GL" or "GG". 3x2pt is not supported'

    npairs_auto, npairs_cross, npairs_tot = get_zpairs(zbins)
    if probe in ['LL', 'GG']:
        npairs = npairs_auto
    elif probe in ['GL', 'LG']:
        npairs = npairs_cross

    assert ind.shape[0] == npairs, 'ind.shape[0] != npairs: maybe you\'re passing the whole ind file ' \
                                   'instead of ind[:npairs, :] - or similia'

    # TODO use jit
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ij in range(npairs):
        for kl in range(npairs):
            i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3]
            cov_6D[:, :, i, j, k, l] = cov_4D[:, :, ij, kl]

    # GL is not symmetric
    # ! this part makes this function very slow
    if probe in ['LL', 'GG']:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(cov_6D[ell1, ell2, :, :, i, j])
                        cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(cov_6D[ell1, ell2, i, j, :, :])

    return cov_6D


#
def cov_6D_to_4D(cov_6D, nbl, zpairs, ind):
    """transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins)
    to (nbl, nbl, zpairs, zpairs)"""
    assert ind.shape[0] == zpairs, "ind.shape[0] != zpairs: maybe you're passing the whole ind file " \
                                   "instead of ind[:zpairs, :] - or similia"
    cov_4D = np.zeros((nbl, nbl, zpairs, zpairs))
    for ij in range(zpairs):
        for kl in range(zpairs):
            # rename for better readability
            i, j, k, l = ind[ij, -2], ind[ij, -1], ind[kl, -2], ind[kl, -1]
            cov_4D[:, :, ij, kl] = cov_6D[:, :, i, j, k, l]
    return cov_4D


def cov_6D_to_4D_blocks(cov_6D, nbl, npairs_AB, npairs_CD, ind_AB, ind_CD):
    """ reshapes the covariance even for the non-diagonal (hence, non-square) blocks needed to build the 3x2pt.
    use npairs_AB = npairs_CD and ind_AB = ind_CD for the normal routine (valid for auto-covariance
    LL-LL, GG-GG, GL-GL and LG-LG). n_columns is used to determine whether the ind array has 2 or 4 columns
    (if it's given in the form of a dictionary or not)
    """
    assert ind_AB.shape[0] == npairs_AB, 'ind_AB.shape[0] != npairs_AB'
    assert ind_CD.shape[0] == npairs_CD, 'ind_CD.shape[0] != npairs_CD'

    # this is to ensure compatibility with both 4-columns and 2-columns ind arrays (dictionary)
    # the penultimante element is the first index, the last one the second index (see s - 1, s - 2 below)
    n_columns_AB = ind_AB.shape[1]  # of columns: this is to understand the format of the file
    n_columns_CD = ind_CD.shape[1]

    # check
    assert n_columns_AB == n_columns_CD, 'ind_AB and ind_CD must have the same number of columns'
    nc = n_columns_AB  # make the name shorter

    cov_4D = np.zeros((nbl, nbl, npairs_AB, npairs_CD))
    for ij in range(npairs_AB):
        for kl in range(npairs_CD):
            i, j, k, l = ind_AB[ij, nc - 2], ind_AB[ij, nc - 1], ind_CD[kl, nc - 2], ind_CD[kl, nc - 1]
            cov_4D[:, :, ij, kl] = cov_6D[:, :, i, j, k, l]
    return cov_4D


def return_combinations(A, B, C, D):
    print(f'C_{A}{C}, C_{B}{D}, C_{A}{D}, C_{B}{C}, N_{A}{C}, N_{B}{D}, N_{A}{D}, N_{B}{C}')


###########################
#
def check_symmetric(array_2d, exact, rtol=1e-05):
    """
    :param a: 2d array
    :param exact: bool
    :param rtol: relative tolerance
    :return: bool, whether the array is symmetric or not
    """
    # """check if the matrix is symmetric, either exactly or within a tolerance
    # """
    assert type(exact) == bool, 'parameter "exact" must be either True or False'
    assert array_2d.ndim == 2, 'the array is not square'
    if exact:
        return np.array_equal(array_2d, array_2d.T)
    else:
        return np.allclose(array_2d, array_2d.T, rtol=rtol, atol=0)


# reshape from 3 to 4 dimensions
def array_3D_to_4D(cov_3D, nbl, npairs):
    print('XXX THIS FUNCTION ONLY WORKS FOR GAUSS-ONLY COVARIANCE')
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    for ell in range(nbl):
        for p in range(npairs):
            for q in range(npairs):
                cov_4D[ell, ell, p, q] = cov_3D[ell, p, q]
    return cov_4D


def cov_2D_to_4D(cov_2D, nbl, block_index='vincenzo', optimize=True):
    """ new (more elegant) version of cov_2D_to_4D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops)
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'zpair_wise', me and Vincenzo block_index == 'ell':
    I add this distinction in the "if" to make it clearer.

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'
    assert cov_2D.ndim == 2, 'the input covariance must be 2-dimensional'

    zpairs_AB = cov_2D.shape[0] // nbl
    zpairs_CD = cov_2D.shape[1] // nbl

    cov_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))

    if optimize:
        if block_index in ['ell', 'vincenzo', 'C-style']:
            cov_4D = cov_2D.reshape((nbl, zpairs_AB, nbl, zpairs_CD)).transpose((0, 2, 1, 3))
        elif block_index in ['ij', 'sylvain', 'F-style']:
            cov_4D = cov_2D.reshape((zpairs_AB, nbl, nbl, zpairs_CD)).transpose((1, 2, 0, 3))
        return cov_4D

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[ipair * nbl + l1, jpair * nbl + l2]
    return cov_4D


def cov_4D_to_2D(cov_4D, block_index='vincenzo', optimize=True):
    """ new (more elegant) version of cov_4D_to_2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.

    this function can also convert to 2D non-square blocks; this is needed to build the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], 'the first two axes of the input covariance must have the same size'
    # assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    nbl = int(cov_4D.shape[0])
    zpairs_AB = int(cov_4D.shape[2])
    zpairs_CD = int(cov_4D.shape[3])

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if optimize:
        if block_index in ['ell', 'vincenzo', 'C-style']:
            cov_2D.reshape(nbl, zpairs_AB, nbl, zpairs_CD)[:, :, :, :] = cov_4D.transpose(0, 2, 1, 3)

        elif block_index in ['ij', 'sylvain', 'F-style']:
            cov_2D.reshape(zpairs_AB, nbl, zpairs_CD, nbl)[:, :, :, :] = cov_4D.transpose(2, 0, 3, 1)
        return cov_2D

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[l1, l2, ipair, jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]

    return cov_2D


def cov_4D_to_2DCLOE_3x2pt(cov_4D, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    print("the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
          " will work both for LG and GL) ")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    lim_1 = zpairs_auto
    lim_2 = zpairs_cross + zpairs_auto
    lim_3 = zpairs_3x2pt

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_4D_to_2D(cov_4D[:, :, :lim_1, :lim_1], block_index)
    cov_LL_LG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_1:lim_2], block_index)
    cov_LL_GG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_2:lim_3], block_index)

    cov_LG_LL = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, :lim_1], block_index)
    cov_LG_LG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], block_index)
    cov_LG_GG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], block_index)

    cov_GG_LL = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, :lim_1], block_index)
    cov_GG_LG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], block_index)
    cov_GG_GG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], block_index)

    # make long rows and stack together
    row_1 = np.hstack((cov_LL_LL, cov_LL_LG, cov_LL_GG))
    row_2 = np.hstack((cov_LG_LL, cov_LG_LG, cov_LG_GG))
    row_3 = np.hstack((cov_GG_LL, cov_GG_LG, cov_GG_GG))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


def cov_2DCLOE_to_4D_3x2pt(cov_2D, nbl, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    print("the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
          " will work both for LG and GL) ")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    # now I'm reshaping the full block diagonal matrix, not just the sub-blocks (cov_2D_to_4D works for both cases)
    lim_1 = zpairs_auto * nbl
    lim_2 = (zpairs_cross + zpairs_auto) * nbl
    lim_3 = zpairs_3x2pt * nbl

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_2D_to_4D(cov_2D[:lim_1, :lim_1], nbl, block_index)
    cov_LL_LG = cov_2D_to_4D(cov_2D[:lim_1, lim_1:lim_2], nbl, block_index)
    cov_LL_GG = cov_2D_to_4D(cov_2D[:lim_1, lim_2:lim_3], nbl, block_index)

    cov_LG_LL = cov_2D_to_4D(cov_2D[lim_1:lim_2, :lim_1], nbl, block_index)
    cov_LG_LG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_1:lim_2], nbl, block_index)
    cov_LG_GG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_2:lim_3], nbl, block_index)

    cov_GG_LL = cov_2D_to_4D(cov_2D[lim_2:lim_3, :lim_1], nbl, block_index)
    cov_GG_LG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_1:lim_2], nbl, block_index)
    cov_GG_GG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_2:lim_3], nbl, block_index)

    # here it is a little more difficult to visualize the stacking, but the probes are concatenated
    # along the 2 zpair_3x2pt-long axes
    cov_4D = np.zeros((nbl, nbl, zpairs_3x2pt, zpairs_3x2pt))

    zlim_1 = zpairs_auto
    zlim_2 = zpairs_cross + zpairs_auto
    zlim_3 = zpairs_3x2pt

    cov_4D[:, :, :zlim_1, :zlim_1] = cov_LL_LL
    cov_4D[:, :, :zlim_1, zlim_1:zlim_2] = cov_LL_LG
    cov_4D[:, :, :zlim_1, zlim_2:zlim_3] = cov_LL_GG

    cov_4D[:, :, zlim_1:zlim_2, :zlim_1] = cov_LG_LL
    cov_4D[:, :, zlim_1:zlim_2, zlim_1:zlim_2] = cov_LG_LG
    cov_4D[:, :, zlim_1:zlim_2, zlim_2:zlim_3] = cov_LG_GG

    cov_4D[:, :, zlim_2:zlim_3, :zlim_1] = cov_GG_LL
    cov_4D[:, :, zlim_2:zlim_3, zlim_1:zlim_2] = cov_GG_LG
    cov_4D[:, :, zlim_2:zlim_3, zlim_2:zlim_3] = cov_GG_GG

    return cov_4D


def cov2corr(cov):
    """ Credits:
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """

    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    correlation = cov / outer_v
    correlation[cov == 0] = 0
    return correlation


## build the noise matrices ##
def build_noise(zbins, nProbes, sigma_eps2, ng_shear, ng_clust, EP_or_ED='EP'):
    """
    function to build the noise power spectra.
    ng = number of galaxies per arcmin^2 (constant, = 30 in IST:F 2020)
    n_bar = # of gal per bin
    """
    conversion_factor = 11818102.860035626  # deg to arcmin^2

    assert isinstance(ng_shear, (int, float, np.ndarray)), 'ng_shear should be int, float or an array'
    assert isinstance(ng_clust, (int, float, np.ndarray)), 'ng_shear should be int, float or an array'
    # this may be relaxed in the future...
    assert type(ng_shear) == type(ng_clust), 'ng_shear and ng_clust should be the same type)'

    # if ng is a number, n_bar will be ng/zbins and the bins have to be equipopulated
    if np.isscalar(ng_shear) and np.isscalar(ng_clust):
        assert ng_shear > 0, 'ng_shear should be positive'
        assert ng_clust > 0, 'ng_clust should be positive'
        assert EP_or_ED == 'EP', 'if ng is a scalar (not a vector), the bins should be equipopulated'
        # assert ng > 20, 'ng should roughly be > 20 (this check is meant to make sure that ng is the cumulative galaxy ' \
        #                 'density, not the galaxy density in each bin)'
        n_bar_shear = ng_shear / zbins * conversion_factor
        n_bar_clust = ng_clust / zbins * conversion_factor

    # if ng is an array, n_bar == ng (this is a slight minomer, since ng is the cumulative galaxy density, while
    # n_bar the galaxy density in each bin). In this case, if the bins are quipopulated, the n_bar array should
    # have all entries almost identical.
    else:
        assert np.all(ng_shear > 0), 'ng_shear should be positive'
        # assert np.sum(ng_shear) > 20, 'ng should roughly be > 20 (this check is meant to make sure that ng is the cumulative galaxy ' \
        #                 'density, not the galaxy density in each bin)'
        if EP_or_ED == 'EP':
            assert np.allclose(np.ones_like(ng_shear) * ng_shear[0], ng_shear, rtol=0.05,
                               atol=0), 'if ng_shear is a vector and the bins are equipopulated, ' \
                                        'the value in each bin should be the same (or very similar)'

        n_bar_shear = ng_shear * conversion_factor
        n_bar_clust = ng_clust * conversion_factor

    # create and fill N
    N = np.zeros((nProbes, nProbes, zbins, zbins))
    np.fill_diagonal(N[0, 0, :, :], sigma_eps2 / (2 * n_bar_shear))
    np.fill_diagonal(N[1, 1, :, :], 1 / n_bar_clust)
    N[0, 1, :, :] = 0
    N[1, 0, :, :] = 0

    return N


def compute_ells(nbl: int, ell_min: int, ell_max: int, recipe, output_ell_bin_edges: bool = False):
    """Compute the ell values and the bin widths for a given recipe.

    Parameters
    ----------
    nbl : int
        Number of ell bins.
    ell_min : int
        Minimum ell value.
    ell_max : int
        Maximum ell value.
    recipe : str
        Recipe to use. Must be either "ISTF" or "ISTNL".
    output_ell_bin_edges : bool, optional
        If True, also return the ell bin edges, by default False

    Returns
    -------
    ells : np.ndarray
        Central ell values.
    deltas : np.ndarray
        Bin widths
    ell_bin_edges : np.ndarray, optional
        ell bin edges. Returned only if output_ell_bin_edges is True.
    """
    if recipe == 'ISTF':
        log10_ell_min = np.log10(ell_min) if ell_min != 0 else ell_min
        ell_bin_edges = np.logspace(log10_ell_min, np.log10(ell_max), nbl + 1)
        ells = (ell_bin_edges[1:] + ell_bin_edges[:-1]) / 2
        deltas = np.diff(ell_bin_edges)
    elif recipe == 'ISTNL':
        log_ell_min = np.log(ell_min) if ell_min != 0 else ell_min
        ell_bin_edges = np.linspace(log_ell_min, np.log(ell_max), nbl + 1)
        ells = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.
        ells = np.exp(ells)
        deltas = np.diff(np.exp(ell_bin_edges))
    else:
        raise ValueError('recipe must be either "ISTF" or "ISTNL"')

    if output_ell_bin_edges:
        return ells, deltas, ell_bin_edges

    return ells, deltas
