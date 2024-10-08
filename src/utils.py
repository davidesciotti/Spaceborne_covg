import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson

DEG2_IN_SPHERE = 4 * np.pi * (180 / np.pi)**2


def get_ngal(ngal_in, EP_or_ED, zbins, EP_check_tol):

    if isinstance(ngal_in, (int, float)):
        assert EP_or_ED == 'EP', 'n_gal must be a scalar in the equipopulated (EP) case'
        ngal_out = ngal_in

    elif type(ngal_in) == list:
        assert len(ngal_in) == zbins, 'n_gal must be a vector of length zbins'
        ngal_out = ngal_in

    elif type(ngal_in) == str:
        nofz = np.genfromtxt(ngal_in)
        assert nofz.shape[1] == zbins + 1, 'nz must be an array of shape (n_z_points, zbins + 1)'
        z_nofz = nofz[:, 0]
        nofz = nofz[:, 1:]
        ngal_out = simpson(y=nofz, x=z_nofz, axis=0)

    if EP_or_ED == 'EP' and not isinstance(ngal_out, (int, float)):
        for zi in range(zbins):
            assert np.allclose(ngal_out[0], ngal_out[zi], atol=0, rtol=EP_check_tol), \
                'n_gal must be the same for all zbins in the equipopulated (EP) case'

    return ngal_out


def cl_2D_to_3D_symmetric(Cl_2D, nbl, zpairs, zbins):
    """ reshape from (nbl, zpairs) to (nbl, zbins, zbins) according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    assert Cl_2D.shape == (nbl, zpairs), f'cl_2d must have shape (nbl, zpairs) = ({nbl}, {zpairs})'

    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for zpair_idx in range(zpairs):
            i, j = triu_idx[0][zpair_idx], triu_idx[1][zpair_idx]
            Cl_3D[ell, i, j] = Cl_2D[ell, zpair_idx]
    # fill lower diagonal (the matrix is symmetric!)
    Cl_3D = fill_3D_symmetric_array(Cl_3D, nbl, zbins)
    return Cl_3D


def cl_2D_to_3D_asymmetric(Cl_2D, nbl, zbins, order):
    """ reshape from (nbl, npairs) to (nbl, zbins, zbins), rows first
    (valid for asymmetric Cij, i.e. C_XC)
    """
    assert order in ['row-major', 'col-major', 'C', 'F'], 'order must be either "row-major", "C" (equivalently), or' \
                                                          '"col-major", "F" (equivalently)'
    if order == 'row-major':
        order = 'C'
    elif order == 'col-major':
        order = 'F'

    Cl_3D = np.zeros((nbl, zbins, zbins))
    Cl_3D = np.reshape(Cl_2D, Cl_3D.shape, order=order)
    return Cl_3D


def fill_3D_symmetric_array(array_3D, nbl, zbins):
    """ mirror the lower/upper triangle """
    assert array_3D.shape == (nbl, zbins, zbins), 'shape of input array must be (nbl, zbins, zbins)'

    array_diag_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        array_diag_3D[ell, :, :] = np.diag(np.diagonal(array_3D, 0, 1, 2)[ell, :])
    array_3D = array_3D + np.transpose(array_3D, (0, 2, 1)) - array_diag_3D
    return array_3D


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


def correlation_from_covariance(covariance):
    """ not thoroughly tested. Taken from 
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    does NOT work with 3x2pt
    """
    if covariance.shape[0] > 2000:
        print("this function doesn't work for 3x2pt")

    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def build_noise(zbins, n_probes, sigma_eps2, ng_shear, ng_clust, EP_or_ED, which_shape_noise='per_component'):
    """Builds the noise power spectra.

    Parameters
    ----------
    zbins : int
        Number of redshift bins.
    n_probes : int 
        Number of probes.
    sigma_eps2 : float
        Square of the *total* ellipticity dispersion.
        sigma_eps2 = sigma_eps ** 2, with
        sigma_eps = sigma_eps_i * sqrt(2),
        sigma_eps_i being the ellipticity dispersion *per component*
    ng_shear : int, float or numpy.ndarray
        Galaxy density of sources, relevant for cosmic shear
        If a scalar, cumulative galaxy density number density, per arcmin^2. 
        This will assume equipopulated bins. 
        If an array, galaxy number density, per arcmin^2, per redshift bin. 
        Must have length zbins.
    ng_clust : int, float or numpy.ndarray
        Galaxy density of lenses, relevant for galaxy clustering
        If a scalar, cumulative galaxy density number density, per arcmin^2. 
        This will assume equipopulated bins. 
        If an array, galaxy number density, per arcmin^2, per redshift bin. 
        Must have length zbins.
    EP_or_ED : str, optional
        Whether bins are equipopulated ('EP') or equidistant ('ED').
    which_shape_noise : str
        Which shape noise to use. 
        'ISTF' for the "incorrect" shape noise (used in ISTF paper), for backwars-compatibility.
        'per_component' for the per component shape noise, taking into account EE-only noise.

    Returns
    -------
    noise_4d : ndarray, shape (n_probes, n_probes, zbins, zbins)
        Noise power spectra matrices

    Notes
    -----
    The noise N is defined as:
        N_LL = sigma_eps^2 / (2 * n_bar) 
        N_GG = 1 / n_bar
        N_GL = N_LG = 0

    """

    conversion_factor = (180 / np.pi * 60)**2  # deg^2 to arcmin^2

    assert isinstance(ng_shear, (int, float, np.ndarray)), 'ng_shear should be int, float or an array'
    assert isinstance(ng_clust, (int, float, np.ndarray)), 'ng_shear should be int, float or an array'
    # this may be relaxed in the future...
    assert type(ng_shear) == type(ng_clust), 'ng_shear and ng_clust should be the same type)'

    # if ng is a scalar, n_bar will be ng/zbins and the bins have to be equipopulated
    if np.isscalar(ng_shear) and np.isscalar(ng_clust):
        assert ng_shear > 0, 'ng_shear should be positive'
        assert ng_clust > 0, 'ng_clust should be positive'
        assert EP_or_ED == 'EP', 'if ng is a scalar (not a vector), the bins should be equipopulated'
        # assert ng > 20, 'ng should roughly be > 20 (this check is meant to make sure that ng is the cumulative galaxy ' \
        #                 'density, not the galaxy density in each bin)'
        n_bar_shear = ng_shear / zbins * conversion_factor
        n_bar_clust = ng_clust / zbins * conversion_factor

    # if ng is an array, n_bar == ng (this is a slight misnomer, since ng is the cumulative galaxy density, while
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
    noise_4d = np.zeros((n_probes, n_probes, zbins, zbins))

    if which_shape_noise == 'ISTF':
        np.fill_diagonal(noise_4d[0, 0, :, :], sigma_eps2 / n_bar_shear)
    elif which_shape_noise == 'per_component':
        np.fill_diagonal(noise_4d[0, 0, :, :], sigma_eps2 / (2 * n_bar_shear))
    else:
        raise ValueError('which_shape_noise must be "ISTF" or "per_component"')

    np.fill_diagonal(noise_4d[1, 1, :, :], 1 / n_bar_clust)
    noise_4d[0, 1, :, :] = 0
    noise_4d[1, 0, :, :] = 0

    return noise_4d


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
        ell_bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
        ells = (ell_bin_edges[1:] + ell_bin_edges[:-1]) / 2
        deltas = np.diff(ell_bin_edges)
    elif recipe == 'ISTNL':
        ell_bin_edges = np.linspace(np.log(ell_min), np.log(ell_max), nbl + 1)
        ells = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.
        ells = np.exp(ells)
        deltas = np.diff(np.exp(ell_bin_edges))
    else:
        raise ValueError('recipe must be either "ISTF" or "ISTNL"')

    if output_ell_bin_edges:
        return ells, deltas, ell_bin_edges

    return ells, deltas
