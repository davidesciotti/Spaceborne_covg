import gc
import itertools
import json
import sys
import time
import warnings
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
from scipy.interpolate import interp1d, RectBivariateSpline, RegularGridInterpolator, CubicSpline, UnivariateSpline
from copy import deepcopy
import utils
import os
ROOT = os.getenv("ROOT")


def build_cl_ring_ordering(cl_3d):

    zbins = cl_3d.shape[1]
    assert cl_3d.shape[1] == cl_3d.shape[2], 'input cls should have shape (ell_bins, zbins, zbins)'
    cl_ring_list = []

    for offset in range(0, zbins):  # offset defines the distance from the main diagonal
        for zi in range(zbins - offset):
            zj = zi + offset
            cl_ring_list.append(cl_3d[:, zi, zj])

    return cl_ring_list


def build_cl_tomo_TEB_ring_ord(cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB, zbins, spectra_types=['T', 'E', 'B']):

    assert cl_TT.shape == cl_EE.shape == cl_BB.shape == cl_TE.shape == cl_EB.shape == cl_TB.shape, \
        'All input arrays must have the same shape.'
    assert cl_TT.ndim == 3, 'the ell axis should be present for all input arrays'

    # Iterate over redshift bins and spectra types to construct the matrix of combinations
    row_idx = 0
    matrix = []
    for zi in range(0, zbins):
        for s1 in spectra_types:
            row = []
            for zj in range(0, zbins):
                for s2 in spectra_types:
                    row.append(f'{s1}-{zi}-{s2}-{zj}')
            matrix.append(row)
            row_idx += 1

    assert len(row) == zbins * len(spectra_types), \
        "The number of elements in the row should be equal to the number of redshift bins times the number of spectra types."

    cl_ring_ord_list = []
    for offset in range(len(row)):
        for zi in range(len(row) - offset):
            zj = zi + offset

            probe_a, zi, probe_b, zj = matrix[zi][zj].split('-')

            if probe_a == 'T' and probe_b == 'T':
                cl = cl_TT
            elif probe_a == 'E' and probe_b == 'E':
                cl = cl_EE
            elif probe_a == 'B' and probe_b == 'B':
                cl = cl_BB
            elif probe_a == 'T' and probe_b == 'E':
                cl = cl_TE
            elif probe_a == 'E' and probe_b == 'B':
                cl = cl_EB
            elif probe_a == 'T' and probe_b == 'B':
                cl = cl_TB
            elif probe_a == 'B' and probe_b == 'T':
                cl = cl_TB.transpose(0, 2, 1)
            elif probe_a == 'B' and probe_b == 'E':
                cl = cl_EB.transpose(0, 2, 1)
            elif probe_a == 'E' and probe_b == 'T':
                cl = cl_TE.transpose(0, 2, 1)
            else:
                raise ValueError(f'Invalid combination: {probe_a}-{probe_b}')

            cl_ring_ord_list.append(cl[:, int(zi), int(zj)])

    return cl_ring_ord_list


def get_sample_field_bu(cl_TT, cl_EE, cl_BB, cl_TE, nside):
    """This routine generates a spin-0 and a spin-2 Gaussian random field based
    on these power spectra.
    From https://namaster.readthedocs.io/en/latest/source/sample_covariance.html
    """
    map_t, map_q, map_u = hp.synfast([cl_TT, cl_EE, cl_BB, cl_TE], nside)
    return nmt.NmtField(mask, [map_t], lite=False), nmt.NmtField(mask, [map_q, map_u], lite=False)


def cls_to_maps(cl_TT, cl_EE, cl_BB, cl_TE, nside):
    """
    This routine generates maps for spin-0 and a spin-2 Gaussian random field based
    on the input power spectra.

    Args:
        cl_TT (numpy.ndarray): Temperature power spectrum.
        cl_EE (numpy.ndarray): E-mode polarization power spectrum.
        cl_BB (numpy.ndarray): B-mode polarization power spectrum.
        cl_TE (numpy.ndarray): Temperature-E-mode cross power spectrum.
        nside (int): HEALPix resolution parameter.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: Temperature map, Q-mode polarization map, U-mode polarization map.
    """
    alm, Elm, Blm = hp.synalm([cl_TT, cl_EE, cl_BB, cl_TE, 0 * cl_TE, 0 * cl_TE],
                              lmax=3 * nside - 1, new=True)
    map_Q, map_U = hp.alm2map_spin([Elm, Blm], nside, 2, 3 * nside - 1)
    map_T = hp.alm2map(alm, nside)
    return map_T, map_Q, map_U


def masked_maps_to_nmtFields(map_T, map_Q, map_U, mask, n_iter=0, lite=True):
    """
    Create NmtField objects from masked maps.

    Args:
        map_T (numpy.ndarray): Temperature map.
        map_Q (numpy.ndarray): Q-mode polarization map.
        map_U (numpy.ndarray): U-mode polarization map.
        mask (numpy.ndarray): Mask to apply to the maps.

    Returns:
        nmt.NmtField, nmt.NmtField: NmtField objects for the temperature and polarization maps.
    """
    f0 = nmt.NmtField(mask, [map_T], n_iter=n_iter, lite=lite)
    f2 = nmt.NmtField(mask, [map_Q, map_U], spin=2, n_iter=n_iter, lite=lite)
    return f0, f2


def compute_master(f_a, f_b, wsp):
    """This function computes power spectra given a pair of fields and a workspace.
    From https://namaster.readthedocs.io/en/latest/source/sample_covariance.html
    NOTE THAT nmt.compute_full_master() does:
    NmtWorkspace.compute_coupling_matrix
    deprojection_bias
    compute_coupled_cell
    NmtWorkspace.decouple_cell
    and gives perfectly consistent results!
    """
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def find_ellmin_from_bpw(bpw, ells, threshold):

    # Calculate cumulative weight to find ell_min
    cumulative_weight = np.cumsum(bpw[0, :, 0, :], axis=-1)

    ell_min = []
    for i in range(bpw.shape[0]):
        idx = np.where(cumulative_weight[i] > threshold)[0]
        if len(idx) > 0:
            ell_min.append(ells[idx[0]])
        else:
            print(f"No index found for band {i} with cumulative weight > {threshold}")

    if ell_min:
        ell_min = int(np.ceil(np.mean(ell_min)))
        print(f"Estimated ell_min: {ell_min}")
    else:
        print("ell_min array is empty")

    return ell_min


def produce_gaussian_sims(cl_TT, cl_EE, cl_BB, cl_TE, nreal, nside, mask, coupled, which_cls):

    # both healpy anafast and nmt.compute_coupled_cell return the coupled cls. Dividing by fsky gives a rough
    # approximation of the true Cls
    correction_factor = 1. if coupled else fsky

    pseudo_cl_tt_list = []
    pseudo_cl_te_list = []
    pseudo_cl_ee_list = []

    print(f'Generating {nreal} maps for nside {nside} and computing cls with {which_cls}...')

    for _ in tqdm(range(nreal)):

        map_T, map_Q, map_U = cls_to_maps(cl_TT, cl_EE, cl_BB, cl_TE, nside)

        if which_cls == 'namaster':

            f0, f2 = masked_maps_to_nmtFields(map_T, map_Q, map_U, mask)

            if coupled:
                # pseudo-Cls. Becomes an ok estimator for the true Cls if divided by fsky
                pseudo_cl_tt = nmt.compute_coupled_cell(f0, f0)[0] / correction_factor
                pseudo_cl_te = nmt.compute_coupled_cell(f0, f2)[0] / correction_factor
                pseudo_cl_ee = nmt.compute_coupled_cell(f2, f2)[0] / correction_factor
            else:
                # best estimator for the true Cls
                pseudo_cl_tt = compute_master(f0, f0, w00)
                pseudo_cl_te = compute_master(f0, f2, w02)
                pseudo_cl_ee = compute_master(f2, f2, w22)

        elif which_cls == 'healpy':

            map_T = hp.remove_monopole(map_T)
            map_Q = hp.remove_monopole(map_Q)
            map_U = hp.remove_monopole(map_U)

            # pseudo-Cls. Becomes an ok estimator for the true Cls if divided by fsky
            pseudo_cl_hp_tot = hp.anafast([map_T * mask, map_Q * mask, map_U * mask])
            pseudo_cl_tt = pseudo_cl_hp_tot[0, :] / correction_factor
            pseudo_cl_ee = pseudo_cl_hp_tot[1, :] / correction_factor
            pseudo_cl_te = pseudo_cl_hp_tot[3, :] / correction_factor

        else:
            raise ValueError('which_cls must be namaster or healpy')

        pseudo_cl_tt_list.append(pseudo_cl_tt)
        pseudo_cl_te_list.append(pseudo_cl_te)
        pseudo_cl_ee_list.append(pseudo_cl_ee)

    sim_cls_dict = {
        'pseudo_cl_tt': np.array(pseudo_cl_tt_list),
        'pseudo_cl_te': np.array(pseudo_cl_te_list),
        'pseudo_cl_ee': np.array(pseudo_cl_ee_list),
    }
    print('...done')

    return sim_cls_dict


def sample_cov_nmt(zi, probe):

    print("Sample covariance from nmt documentation")
    zi = 0
    sample_cov = np.zeros([nbl_eff, nbl_eff])
    sample_mean = np.zeros(nbl_eff)

    for _ in tqdm(np.arange(nreal)):

        f0, f2 = get_sample_field_bu(cl_TT=cl_GG_unbinned[:, zi, zi],
                                     cl_EE=cl_LL_unbinned[:, zi, zi],
                                     cl_BB=cl_BB_unbinned[:, zi, zi],
                                     cl_TE=cl_GL_unbinned[:, zi, zi],
                                     nside=nside)

        if probe == 'GG':
            cl_sim = compute_master(f0, f0, w00)[0]
        elif probe == 'LL':
            cl_sim = compute_master(f2, f2, w22)[0]

        sample_cov += cl_sim[None, :] * cl_sim[:, None]
        sample_mean += cl_sim

    sample_mean /= nreal
    sample_cov = sample_cov / nreal
    sample_cov -= sample_mean[None, :] * sample_mean[:, None]

    # excellent agreement for bias=True
    # cl_gg_list_test = np.array(cl_gg_list_test)
    # cov_gg_test = np.cov(cl_gg_list_test, rowvar=False, bias=True)
    # np.testing.assert_allclose(cov_gg_test, sample_cov, atol=0, rtol=1e-2)

    return sample_cov


def get_lmid(ells, k):
    return 0.5 * (ells[k:] + ells[:-k])


cov_blocks_names_all = ('LLLL', 'LLGL', 'LLGG', 'GLLL', 'GLGL', 'GLGG', 'GGLL', 'GGGL', 'GGGG')
cov_nmt_block_names = ['LLLL', 'GLLL', 'GGLL', 'GLGL', 'GGGL', 'GGGG']

# ! settings
# import the yaml config file
# cfg = yaml.load(sys.stdin, Loader=yaml.FullLoader)
# if you want to execute without passing the path
with open(f'{ROOT}/Spaceborne_covg/config/example_config_namaster.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

survey_area_deg2 = cfg['survey_area_deg2']  # deg^2
fsky = survey_area_deg2 / utils.DEG2_IN_SPHERE

zbins = cfg['zbins']
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nbl = cfg['ell_bins']

sigma_eps = cfg['sigma_eps_i'] * np.sqrt(2)
sigma_eps2 = sigma_eps ** 2

EP_or_ED = cfg['EP_or_ED']
GL_or_LG = 'GL'
triu_tril = cfg['triu_tril']
row_col_major = cfg['row_col_major']
covariance_ordering_2D = cfg['covariance_ordering_2D']

part_sky = cfg['part_sky']
workspace_path = cfg['workspace_path']
mask_path = cfg['mask_path']

output_folder = cfg['output_folder']
n_probes = 2
# ! end settings


# sanity checks
assert EP_or_ED in ('EP', 'ED'), 'EP_or_ED must be either EP or ED'
assert GL_or_LG in ('GL', 'LG'), 'GL_or_LG must be either GL or LG'
assert triu_tril in ('triu', 'tril'), 'triu_tril must be either "triu" or "tril"'
assert row_col_major in ('row-major', 'col-major'), 'row_col_major must be either "row-major" or "col-major"'
assert isinstance(zbins, int), 'zbins must be an integer'
assert isinstance(nbl, int), 'nbl must be an integer'

if EP_or_ED == 'EP':
    n_gal_shear = cfg['n_gal_shear']
    n_gal_clustering = cfg['n_gal_clustering']
    assert np.isscalar(n_gal_shear), 'n_gal_shear must be a scalar'
    assert np.isscalar(n_gal_clustering), 'n_gal_clustering must be a scalar'
elif EP_or_ED == 'ED':
    n_gal_shear = np.genfromtxt(cfg['n_gal_path_shear'])
    n_gal_clustering = np.genfromtxt(cfg['n_gal_path_clustering'])
    assert len(n_gal_shear) == zbins, 'n_gal_shear must be a vector of length zbins'
    assert len(n_gal_clustering) == zbins, 'n_gal_clustering must be a vector of length zbins'
else:
    raise ValueError('EP_or_ED must be either EP or ED')

# covariance and datavector ordering
probe_ordering = [['L', 'L'], [GL_or_LG[0], GL_or_LG[1]], ['G', 'G']]
ind = utils.build_full_ind(triu_tril, row_col_major, zbins)
zpairs_auto, zpairs_cross, zpairs_3x2pt = utils.get_zpairs(zbins)
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]

# ! ell binning
if cfg['ell_path'] is None:
    assert cfg['delta_ell_path'] is None, 'if ell_path is None, delta_ell_path must be None'
if cfg['delta_ell_path'] is None:
    assert cfg['ell_path'] is None, 'if delta_ell_path is None, ell_path must be None'

if cfg['ell_path'] is None and cfg['delta_ell_path'] is None:
    ell_values, delta_values, ell_bin_edges = utils.compute_ells(nbl, ell_min, ell_max, recipe='ISTF',
                                                                 output_ell_bin_edges=True)
    ell_bin_lower_edges = ell_bin_edges[:-1]
    ell_bin_upper_edges = ell_bin_edges[1:]

    # save to file for good measure
    ell_grid_header = f'ell_min = {ell_min}\tell_max = {ell_max}\tell_bins = {nbl}\n' \
        f'ell_bin_lower_edge\tell_bin_upper_edge\tell_bin_center\tdelta_ell'
    ell_grid = np.column_stack((ell_bin_lower_edges, ell_bin_upper_edges, ell_values, delta_values))
    np.savetxt(f'{output_folder}/ell_grid.txt', ell_grid, header=ell_grid_header)

else:
    print('Loading \ell and \Delta \ell values from file')

    ell_values = np.genfromtxt(cfg['ell_path'])
    delta_values = np.genfromtxt(cfg['delta_ell_path'])
    nbl = len(ell_values)

    assert len(ell_values) == len(delta_values), 'ell values must have a number of entries as delta ell'
    assert np.all(delta_values > 0), 'delta ell values must have strictly positive entries'
    assert np.all(np.diff(ell_values) > 0), 'ell values must have strictly increasing entries'
    assert ell_values.ndim == 1, 'ell values must be a 1D array'
    assert delta_values.ndim == 1, 'delta ell values must be a 1D array'

# ! import cls
cl_LL_unbinned = np.load(f'{cfg["cl_LL_3D_path"].format(ROOT=ROOT)}')
cl_GL_unbinned = np.load(f'{cfg["cl_GL_3D_path"].format(ROOT=ROOT)}')
cl_GG_unbinned = np.load(f'{cfg["cl_GG_3D_path"].format(ROOT=ROOT)}')

# TODO check that the ell loaded or computed above matches the ell of the loaded Cl's
# For now I just construct the 5D 3x2 Cl's from the nbl of the loaded Cl's
nbl = cl_GG_unbinned.shape[0]

cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
cl_3x2pt_5D[0, 0, :, :, :] = cl_LL_unbinned
cl_3x2pt_5D[1, 0, :, :, :] = cl_GL_unbinned
cl_3x2pt_5D[0, 1, :, :, :] = np.transpose(cl_GL_unbinned, (0, 2, 1))
cl_3x2pt_5D[1, 1, :, :, :] = cl_GG_unbinned

# ! Compute covariance
# create a noise with dummy axis for ell, to have the same shape as cl_3x2pt_5D
noise_3x2pt_4D = utils.build_noise(zbins, n_probes, sigma_eps2=sigma_eps2,
                                   ng_shear=n_gal_shear,
                                   ng_clust=n_gal_clustering,
                                   EP_or_ED=EP_or_ED)
noise_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
for probe_A in (0, 1):
    for probe_B in (0, 1):
        for ell_idx in range(nbl):
            noise_3x2pt_5D[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

# compute 3x2pt cov
start = time.perf_counter()
if part_sky:
    print('Computing the partial-sky covariance with NaMaster')

    # ! =============================================== IMPLEMENTATION BY DAVIDE =======================================
    # TODO check implementation by R. Upham: https://github.com/robinupham/shear_pcl_cov/blob/main/shear_pcl_cov/gaussian_cov.py
    import healpy as hp
    import pymaster as nmt

    ells_unbinned = np.arange(cl_LL_unbinned.shape[0])
    ells_per_band = cfg['ells_per_band']
    nside = cfg['nside']
    nreal = cfg['nreal']
    zbins_use = cfg['zbins_use']
    coupled = cfg['coupled']
    use_INKA = cfg['use_INKA']

    coupled_label = 'coupled' if coupled else 'uncoupled'

    # read or generate mask
    if cfg['read_mask']:
        if mask_path.endswith('.fits'):
            mask = hp.read_map(mask_path)
            dr1_date = 9191.0
            mask = np.where(np.logical_and(mask <= dr1_date, mask >= 0.), 1., 0,)
        elif mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        mask = hp.ud_grade(mask, nside_out=nside)

    else:
        mask = utils.generate_polar_cap(area_deg2=survey_area_deg2, nside=cfg['nside'])

    fsky = np.mean(mask**2)
    survey_area_deg2 = fsky * utils.DEG2_IN_SPHERE

    # TODO check np.all(mask == 1)

    # apodize
    hp.mollview(mask, title='before apodization', cmap='inferno_r')
    if cfg['apodize_mask'] and int(survey_area_deg2) != 41252:
        mask = nmt.mask_apodization(mask, aposize=cfg['aposize'], apotype="Smooth")
        hp.mollview(mask, title='after apodization', cmap='inferno_r')

    # recompute after apodizing
    fsky = np.mean(mask**2)
    survey_area_deg2 = fsky * utils.DEG2_IN_SPHERE

    npix = hp.nside2npix(nside)
    pix_area = 4 * np.pi

    # check fsky and nside
    nside_from_mask = hp.get_nside(mask)
    assert nside_from_mask == cfg['nside'], 'nside from mask is not consistent with the desired nside in the cfg file'

    # set different possible values for lmax
    lmax_mask = int(np.pi / hp.pixelfunc.nside2resol(nside))
    lmax_healpy = 3 * nside
    # to be safe, following https://heracles.readthedocs.io/stable/examples/example.html
    lmax_healpy_safe = int(1.5 * nside)  # TODO test this
    lmax = lmax_healpy

    # get lmin: quick estimate
    survey_area_rad = np.sum(mask) * hp.nside2pixarea(nside)
    lmin_mask = int(np.ceil(np.pi / np.sqrt(survey_area_rad)))

    # ! Define the set of bandpowers used in the computation of the pseudo-Cl
    # Initialize binning scheme with bandpowers of constant width (ells_per_band multipoles per bin)
    # TODO use lmax_mask instead of nside? Decide which binning scheme is the best
    # ell_values, delta_values, ell_bin_edges = utils.compute_ells(nbl, 0, lmax, recipe='ISTF', output_ell_bin_edges=True)
    # bin_obj = nmt.NmtBin.from_edges(ell_bin_edges[:-1].astype(int), ell_bin_edges[1:].astype(int), is_Dell=False, f_ell=None)
    bin_obj = nmt.NmtBin.from_nside_linear(nside, ells_per_band)
    # bin_obj = nmt.NmtBin.from_edges(
    # ell_bin_lower_edges.astype(int),
    # ell_bin_upper_edges.astype(int), is_Dell=False, f_ell=None)
    # bin_obj = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=ells_per_band, is_Dell=False, f_ell=None) # TODO test this

    ells_eff = bin_obj.get_effective_ells()  # get effective ells per bandpower
    ells_tot = np.arange(lmax)
    nbl_eff = len(ells_eff)
    nbl_tot = len(ells_tot)
    ells_eff_edges = np.array([bin_obj.get_ell_list(i)[0] for i in range(nbl_eff)])
    ells_eff_edges = np.append(ells_eff_edges, bin_obj.get_ell_list(nbl_eff - 1)[-1] + 1)  # careful f the +1!
    lmin_eff = ells_eff_edges[0]
    lmax_eff = ells_eff_edges[-1]
    ells_bpw = ells_tot[lmin_eff:lmax_eff]
    delta_ells_bpw = np.diff(np.array([bin_obj.get_ell_list(i)[0] for i in range(nbl_eff)]))
    assert np.all(delta_ells_bpw == ells_per_band), 'delta_ell from bpw does not match ells_per_band'

    # ! create nmt field from the mask (there will be no maps associated to the fields)
    # TODO maks=None (as in the example) or maps=[mask]? I think None
    start_time = time.perf_counter()
    print('computing coupling coefficients...')
    f0_mask = nmt.NmtField(mask=mask, maps=None, spin=0, lite=True)
    f2_mask = nmt.NmtField(mask=mask, maps=None, spin=2, lite=True)
    w00 = nmt.NmtWorkspace()
    w02 = nmt.NmtWorkspace()
    w22 = nmt.NmtWorkspace()
    w00.compute_coupling_matrix(f0_mask, f0_mask, bin_obj)
    w02.compute_coupling_matrix(f0_mask, f2_mask, bin_obj)
    w22.compute_coupling_matrix(f2_mask, f2_mask, bin_obj)
    print(f'...done in {(time.perf_counter() - start_time):.2f}s')

    # ! Plot bpowers
    # TODO: better understand difference between bpw_00, 02, 22, if any
    # TODO: better understand lmin estimate (I could do it direcly from bin_obj...)

    # Get bandpower window functions. Convolve the theory power spectra with these as an alternative to the combination
    # of function calls w.decouple_cell(w.couple_cell(cls_theory))
    bpw_00 = w00.get_bandpower_windows()
    bpw_02 = w02.get_bandpower_windows()
    bpw_22 = w22.get_bandpower_windows()

    if cfg['which_ell_weights'] == 'get_weight_list()':
        ell_weights = np.array([bin_obj.get_weight_list(ell_idx)
                                for ell_idx in range(nbl_eff)]).flatten()  # get effective ells per bandpower
    elif cfg['which_ell_weights'] == 'get_bandpower_windows()':
        warnings.warn('Using bpw_00 as ell_weights')
        ell_weights = bpw_00[0, :, 0]

        # interpolate on ells_bpw
        # ell_weights = np.zeros((nbl_eff, len(ells_bpw)))
        # for ell_idx in range(nbl_eff):
        # ell_weights[ell_idx, :] = np.interp(ells_bpw, ells_tot, _ell_weights[ell_idx, :])
    else:
        raise ValueError(f"Invalid value for 'which_ell_weights': {cfg['which_ell_weights']}")

    assert bpw_00.shape[1] == bpw_02.shape[1] == bpw_22.shape[1], \
        "The number of bandpower windows must be the same for all fields"

    # Plotting bandpower windows and ell_min
    lmin_bpw = find_ellmin_from_bpw(bpw_00, ells=ells_tot, threshold=0.95)

    clr = cm.rainbow(np.linspace(0, 1, bpw_00.shape[1]))
    plt.figure(figsize=(10, 6))
    for i in range(nbl_eff):
        plt.plot(ells_tot, bpw_00[0, i, 0, :], c=clr[i], label='bpw_00' if i == 0 else '')
        plt.plot(ells_tot, bpw_02[0, i, 0, :], c=clr[i], ls=':', label='bpw_02' if i == 0 else '')
        plt.plot(ells_tot, bpw_22[0, i, 0, :], c=clr[i], ls='--', label='bpw_22' if i == 0 else '')

    # ell edges
    for i in range(nbl_eff + 1):
        plt.axvline(ells_eff_edges[i], c='k', ls='--')

    plt.axvline(lmin_bpw, color='r', linestyle='--', label='Estimated ell_min')
    plt.xlabel(r'$\ell$')
    plt.ylabel('Window function')
    plt.title('Bandpower Window Functions')
    plt.legend()
    plt.show()
    # TODO finish checking lmin
    # ! end get lmin: better estimate

    print('lmin_mask:', lmin_mask)
    print('lmin_from bpw:', lmin_bpw)
    print('lmax_mask:', lmax_mask)
    print('lmax_healpy:', lmax_healpy)
    print('nside:', nside)
    print('fsky after apodization:', fsky)

    # cut and bin the theory
    cl_GG_unbinned = cl_GG_unbinned[:lmax, :zbins_use, :zbins_use]
    cl_GL_unbinned = cl_GL_unbinned[:lmax, :zbins_use, :zbins_use]
    cl_LL_unbinned = cl_LL_unbinned[:lmax, :zbins_use, :zbins_use]
    cl_BB_unbinned = np.zeros_like(cl_LL_unbinned)
    cl_GB_unbinned = np.zeros_like(cl_LL_unbinned)
    cl_LB_unbinned = np.zeros_like(cl_LL_unbinned)

    cl_GG_bpw = np.zeros((nbl_eff, zbins_use, zbins_use))
    cl_GL_bpw = np.zeros((nbl_eff, zbins_use, zbins_use))
    cl_LL_bpw = np.zeros((nbl_eff, zbins_use, zbins_use))
    cl_GG_bpw_dav = np.zeros((nbl_eff, zbins_use, zbins_use))
    cl_GL_bpw_dav = np.zeros((nbl_eff, zbins_use, zbins_use))
    cl_LL_bpw_dav = np.zeros((nbl_eff, zbins_use, zbins_use))
    for zi in range(zbins_use):
        for zj in range(zbins_use):
            cl_GG_bpw[:, zi, zj] = bin_obj.bin_cell(cl_GG_unbinned[:, zi, zj])
            cl_GL_bpw[:, zi, zj] = bin_obj.bin_cell(cl_GL_unbinned[:, zi, zj])
            cl_LL_bpw[:, zi, zj] = bin_obj.bin_cell(cl_LL_unbinned[:, zi, zj])
            # these are for a consistency check against bin_obj.bin_cell()
            cl_GG_bpw_dav[:, zi, zj] = utils.bin_cell(ells_in=ells_bpw, ells_out=ells_eff, ells_out_edges=ells_eff_edges,
                                                      cls_in=cl_GG_unbinned[ells_bpw, zi, zj], weights=None,
                                                      ells_eff=ells_eff, which_binning='mean')
            cl_GL_bpw_dav[:, zi, zj] = utils.bin_cell(ells_in=ells_bpw, ells_out=ells_eff, ells_out_edges=ells_eff_edges,
                                                      cls_in=cl_GL_unbinned[ells_bpw, zi, zj], weights=None,
                                                      ells_eff=ells_eff, which_binning='mean')
            cl_LL_bpw_dav[:, zi, zj] = utils.bin_cell(ells_in=ells_bpw, ells_out=ells_eff, ells_out_edges=ells_eff_edges,
                                                      cls_in=cl_LL_unbinned[ells_bpw, zi, zj], weights=None,
                                                      ells_eff=ells_eff, which_binning='mean')

    # generate sample fields
    # TODO how about the cross-redshifts?
    f0 = np.empty(zbins_use, dtype=object)
    f2 = np.empty(zbins_use, dtype=object)
    for zi in range(zbins_use):
        map_T, map_Q, map_U = cls_to_maps(cl_TT=cl_GG_unbinned[:, zi, zi],
                                          cl_EE=cl_LL_unbinned[:, zi, zi],
                                          cl_BB=cl_BB_unbinned[:, zi, zi],
                                          cl_TE=cl_GL_unbinned[:, zi, zi],
                                          nside=nside)
        f0[zi], f2[zi] = masked_maps_to_nmtFields(map_T, map_Q, map_U, mask)

    # Create a map(s) from cl(s) to visualize the simulated - masked - maps, just for fun
    zi = 0
    map_t, map_q, map_u = cls_to_maps(cl_TT=cl_GG_unbinned[:, zi, zi],
                                      cl_EE=cl_LL_unbinned[:, zi, zi],
                                      cl_BB=cl_BB_unbinned[:, zi, zi],
                                      cl_TE=cl_GL_unbinned[:, zi, zi],
                                      nside=nside)
    hp.mollview(map_t * mask, title=f'masked map T, zi={zi}', cmap='inferno_r')
    hp.mollview(map_q * mask, title=f'masked map Q, zi={zi}', cmap='inferno_r')
    hp.mollview(map_u * mask, title=f'masked map U, zi={zi}', cmap='inferno_r')

    # ! COMPUTE AND COMPARE DIFFERENT VERSIONS OF THE Cls
    # with healpy
    _map_t = hp.remove_monopole(map_t)
    _map_q = hp.remove_monopole(map_q)
    _map_u = hp.remove_monopole(map_u)
    hp_pcl_tot = hp.anafast([_map_t * mask, _map_q * mask, _map_u * mask])
    hp_pcl_GG = hp_pcl_tot[0, :]
    hp_pcl_LL = hp_pcl_tot[1, :]
    hp_pcl_GL = hp_pcl_tot[3, :]

    # ! updated way, taking into account cross-bins

    cl_ring_big_list = build_cl_tomo_TEB_ring_ord(
        cl_TT=cl_GG_unbinned,
        cl_EE=cl_LL_unbinned,
        cl_BB=cl_BB_unbinned,
        cl_TE=cl_GL_unbinned,
        cl_EB=cl_LB_unbinned,
        cl_TB=cl_GB_unbinned,
        zbins=zbins_use, spectra_types=['T', 'E', 'B'])

    corr_alms_tot = hp.synalm(cl_ring_big_list, lmax=3 * nside - 1, new=True)
    assert len(corr_alms_tot) == zbins_use * 3, 'wrong number of alms'

    # extract alm for TT, EE, BB
    corr_alms = corr_alms_tot[::3]
    corr_Elms_Blms = list(zip(corr_alms_tot[1::3], corr_alms_tot[2::3]))

    # compute correlated maps for each bin
    corr_maps_gg = [hp.alm2map(alm, nside) for alm in corr_alms]
    corr_maps_ll = [hp.alm2map_spin([Elm, Blm], nside, 2, 3 * nside - 1) for (Elm, Blm) in corr_Elms_Blms]

    # # plot for a quick check (these are lots of plots!)
    # for i in range(len(corr_maps_gg)):
    #     hp.mollview(corr_maps_gg[i], title='T')
    # for i in range(len(corr_maps_ll)):
    #     hp.mollview(corr_maps_ll[i][0], title='Q')
    #     hp.mollview(corr_maps_ll[i][1], title='U')

    # now instantiate the fields
    f0 = np.array([nmt.NmtField(mask, [map_T], n_iter=3, lite=True) for map_T in corr_maps_gg])
    f2 = np.array([nmt.NmtField(mask, [map_Q, map_U], n_iter=3, lite=True) for (map_Q, map_U) in corr_maps_ll])

    # TODO add noise?
    cl_GG_master = np.zeros((nbl_eff, zbins_use, zbins_use))
    cl_GL_master = np.zeros((nbl_eff, zbins_use, zbins_use))
    cl_LL_master = np.zeros((nbl_eff, zbins_use, zbins_use))
    pcl_GG_nmt = np.zeros((nbl_tot, zbins_use, zbins_use))
    pcl_GL_nmt = np.zeros((nbl_tot, zbins_use, zbins_use))
    pcl_LL_nmt = np.zeros((nbl_tot, zbins_use, zbins_use))
    bpw_pcl_GG_nmt = np.zeros((nbl_eff, zbins_use, zbins_use))
    bpw_pcl_GL_nmt = np.zeros((nbl_eff, zbins_use, zbins_use))
    bpw_pcl_LL_nmt = np.zeros((nbl_eff, zbins_use, zbins_use))
    hp_pcl_GG = np.zeros((nbl_tot, zbins_use, zbins_use))
    hp_pcl_GL = np.zeros((nbl_tot, zbins_use, zbins_use))
    hp_pcl_LL = np.zeros((nbl_tot, zbins_use, zbins_use))
    print('Computing pseudo-cls...')
    for zi in tqdm(range(zbins_use)):
        for zj in range(zbins_use):
            # MASTER estimator:
            cl_GG_master[:, zi, zj] = compute_master(f0[zi], f0[zj], w00)[0, :]
            cl_GL_master[:, zi, zj] = compute_master(f0[zi], f2[zj], w02)[0, :]
            cl_LL_master[:, zi, zj] = compute_master(f2[zi], f2[zj], w22)[0, :]
            # Effectively, this is equivalent to calling the usual HEALPix anafast routine on the masked and contaminant-cleaned maps.
            pcl_GG_nmt[:, zi, zj] = nmt.compute_coupled_cell(f0[zi], f0[zj])[0, :]
            pcl_GL_nmt[:, zi, zj] = nmt.compute_coupled_cell(f0[zi], f2[zj])[0, :]
            pcl_LL_nmt[:, zi, zj] = nmt.compute_coupled_cell(f2[zi], f2[zj])[0, :]
            # * "bandpowers" = binned (pseudo)-C_l
            bpw_pcl_GG_nmt[:, zi, zj] = bin_obj.bin_cell(pcl_GG_nmt[:, zi, zj])
            bpw_pcl_GL_nmt[:, zi, zj] = bin_obj.bin_cell(pcl_GL_nmt[:, zi, zj])
            bpw_pcl_LL_nmt[:, zi, zj] = bin_obj.bin_cell(pcl_LL_nmt[:, zi, zj])
            # * with healpy:
            # 1. build a flat list of the 3 maps for the i and j zbins
            _corr_maps_zi = list(itertools.chain([corr_maps_gg[zi]], corr_maps_ll[zi]))
            _corr_maps_zj = list(itertools.chain([corr_maps_gg[zj]], corr_maps_ll[zj]))
            # 2. remove monopole
            _corr_maps_zi = [hp.remove_monopole(_corr_maps_zi[spec_ix]) for spec_ix in range(3)]
            _corr_maps_zj = [hp.remove_monopole(_corr_maps_zj[spec_ix]) for spec_ix in range(3)]
            # 3. compute cls for each bin
            hp_pcl_tot = hp.anafast(map1=[_corr_maps_zi[0] * mask, _corr_maps_zi[1] * mask, _corr_maps_zi[2] * mask],
                                    map2=[_corr_maps_zj[0] * mask, _corr_maps_zj[1] * mask, _corr_maps_zj[2] * mask])
            # output is TT, EE, BB, TE, EB, TB
            hp_pcl_GG[:, zi, zj] = hp_pcl_tot[0, :]
            hp_pcl_LL[:, zi, zj] = hp_pcl_tot[1, :]
            hp_pcl_GL[:, zi, zj] = hp_pcl_tot[3, :]

    # ! compare results
    block = 'GLGL'
    zi, zj = 0, 1

    for block in ['GGGG', 'LLLL', 'GLGL']:

        if block == 'GGGG':
            hp_pcl = hp_pcl_GG
            nmt_pcl = pcl_GG_nmt
            master_cl = cl_GG_master
            cl_th_bpw = cl_GG_bpw
            cl_th_unbinned = cl_GG_unbinned
            cl_th_bpw_dav = cl_GG_bpw_dav
            noise_idx = 0
            mm_gg = w00.get_coupling_matrix()
            pseudo_cl_dav = np.einsum('ij,jkl->ikl', mm_gg, cl_GG_unbinned)
        elif block == 'LLLL':
            hp_pcl = hp_pcl_LL
            nmt_pcl = pcl_LL_nmt
            master_cl = cl_LL_master
            cl_th_bpw = cl_LL_bpw
            cl_th_bpw_dav = cl_LL_bpw_dav
            cl_th_unbinned = cl_LL_unbinned
            noise_idx = 1
            mm_ll = w22.get_coupling_matrix()
            pseudo_cl_dav = np.einsum('ij,jkl->ikl', mm_ll[:nbl_tot, :nbl_tot], cl_LL_unbinned)
        elif block == 'GLGL':
            hp_pcl = hp_pcl_GL
            nmt_pcl = pcl_GL_nmt
            master_cl = cl_GL_master
            cl_th_bpw = cl_GL_bpw
            cl_th_bpw_dav = cl_GL_bpw_dav
            cl_th_unbinned = cl_GL_unbinned
            noise_idx = 1
            mm_gl = w02.get_coupling_matrix()
            pseudo_cl_dav = np.einsum('ij,jkl->ikl', mm_gl[:nbl_tot, :nbl_tot],
                                      cl_GL_unbinned)  # TODO test this better!

        assert np.allclose(cl_th_bpw, cl_th_bpw_dav, atol=0, rtol=1e-4)

        plt.figure()
        clr = cm.rainbow(np.linspace(0, 1, zbins_use))

        plt.plot(ells_tot, hp_pcl[:, zi, zj], label=f'hp pseudo-cl', alpha=.7)
        plt.plot(ells_tot, nmt_pcl[:, zi, zj], label=f'nmt pseudo-cl', alpha=.7, ls='--')
        plt.plot(ells_eff, master_cl[:, zi, zj], label=f'MASTER-cl', alpha=.7, marker='.')
        plt.plot(ells_tot, pseudo_cl_dav[:, zi, zj], label=f'dav pseudo-cl', alpha=.7)

        plt.scatter(ells_eff, cl_th_bpw[:, zi, zj] * fsky, marker='.', label=f'bpw th cls*fsky')
        plt.plot(ells_tot, cl_th_unbinned[:, zi, zj], label=f'unbinned th cls')
        plt.plot(ells_tot, cl_th_unbinned[:, zi, zj] * fsky, label=f'unbinned th cls*fsky')

        plt.xlabel(r'$\ell$')
        plt.axvline(lmax_healpy_safe, color='k', ls='--', label='1.5 * nside', alpha=.7)
        plt.yscale('log')
        plt.legend()
        plt.ylabel(r'$C_\ell$')
        plt.title(f'{block}, nside={nside}, fsky={fsky:.2f}, zi={zi}, zj={zj}')
        plt.xscale('log')
        plt.tight_layout()

        # plt.savefig(f'{block}.png')
        # plt.show()

    # ! Let's now compute the Gaussian estimate of the covariance!
    start_time = time.perf_counter()
    # First we generate a NmtCovarianceWorkspace object to precompute
    # and store the necessary coupling coefficients
    cw = nmt.NmtCovarianceWorkspace()
    # This is the time-consuming operation
    # Note that you only need to do this once, regardless of spin
    print("Computing cov workspace coupling coefficients...")
    # cw.compute_coupling_coefficients(f0[0], f0[0], f0[0], f0[0])
    cw.compute_coupling_coefficients(f0_mask, f0_mask, f0_mask, f0_mask)
    print(f"Coupling coefficients computed in {(time.perf_counter() - start_time):.2f} s...")

    # TODO generalize to all zbin cross-correlations; z=0 for the moment
    # shape: (n_cls, n_bpws, n_cls, lmax+1)
    # n_cls is the number of power spectra (1, 2 or 4 for spin 0-0, spin 0-2 and spin 2-2 correlations)

    # if coupled:
    #     raise ValueError('coupled case not fully implemented yet')
    #     print('Inputting pseudo-Cls/fsky to use INKA...')
    #     nbl_4covnmt = nbl_tot
    #     cl_GG_4covnmt = pcl_GG_nmt[:, zi, zj] / fsky
    #     cl_GL_4covnmt = pcl_GL_nmt[:, zi, zj] / fsky
    #     cl_LL_4covnmt = pcl_LL_nmt[:, zi, zj] / fsky
    #     cl_GG_4covsb = pcl_GG_nmt  # or bpw_pcl_GG_nmt?
    #     cl_GL_4covsb = pcl_GL_nmt  # or bpw_pcl_GL_nmt?
    #     cl_LL_4covsb = pcl_LL_nmt  # or bpw_pcl_LL_nmt?
    #     ells_4covsb = ells_tot
    #     nbl_4covsb = len(ells_4covsb)
    #     delta_ells_4covsb = np.ones(nbl_4covsb)  # since it's unbinned
    # else:

    nbl_4covnmt = nbl_eff
    ells_4covsb = ells_tot
    nbl_4covsb = len(ells_4covsb)
    delta_ells_4covsb = np.ones(nbl_4covsb)  # since it's unbinned
    cl_GG_4covsb = cl_GG_unbinned[:, :zbins_use, :zbins_use]
    cl_GL_4covsb = cl_GL_unbinned[:, :zbins_use, :zbins_use]
    cl_LL_4covsb = cl_LL_unbinned[:, :zbins_use, :zbins_use]
    if use_INKA:
        cl_GG_4covnmt = pcl_GG_nmt / fsky
        cl_GL_4covnmt = pcl_GL_nmt / fsky
        cl_LL_4covnmt = pcl_LL_nmt / fsky

        # TODO not super sure about this
        # cl_GG_4covsb = pcl_GG_nmt[:, :zbins_use, :zbins_use] / fsky
        # cl_GL_4covsb = pcl_GL_nmt[:, :zbins_use, :zbins_use] / fsky
        # cl_LL_4covsb = pcl_LL_nmt[:, :zbins_use, :zbins_use] / fsky
    else:
        cl_GG_4covnmt = cl_GG_unbinned
        cl_GL_4covnmt = cl_GL_unbinned
        cl_LL_4covnmt = cl_LL_unbinned
        # cl_GG_4covsb = cl_GG_unbinned[:, :zbins_use, :zbins_use]
        # cl_GL_4covsb = cl_GL_unbinned[:, :zbins_use, :zbins_use]
        # cl_LL_4covsb = cl_LL_unbinned[:, :zbins_use, :zbins_use]

    cl_tt = cl_GG_4covnmt
    cl_te = cl_GL_4covnmt
    cl_ee = cl_LL_4covnmt
    cl_tb = np.zeros_like(cl_GG_4covnmt)
    cl_eb = np.zeros_like(cl_GG_4covnmt)
    cl_bb = np.zeros_like(cl_GG_4covnmt)

    cov_nmt_10d = utils.nmt_gaussian_cov(cl_tt, cl_te, cl_ee, cl_tb, cl_eb, cl_bb, zbins_use, nbl_eff,
     coupled, cw, w00, w02, w22, nbl_4covnmt, False)
    # cov_nmt_10d = utils.nmt_gaussian_cov_spin0(cl_tt, cl_te, cl_ee, cl_tb, cl_eb, cl_bb, zbins_use, nbl_eff,
    #                                            coupled, cw, w00, w02, w22, nbl_4covnmt)
    
    probename_dict = {
        'L': 0,
        'G': 1,
    }
    probename_dict_inv = {
        '0': 'L',
        '1': 'G',
    }

    # ! SPACEBORNE full-sky/fsky covariance
    cl_3x2pt_5d = np.zeros((n_probes, n_probes, nbl_4covsb, zbins_use, zbins_use))
    cl_3x2pt_5d[0, 0, :, :, :] = cl_LL_4covsb
    cl_3x2pt_5d[1, 0, :, :, :] = cl_GL_4covsb
    cl_3x2pt_5d[0, 1, :, :, :] = cl_GL_4covsb.transpose(0, 2, 1)
    cl_3x2pt_5d[1, 1, :, :, :] = cl_GG_4covsb
    noise_3x2pt_5d = np.zeros_like(cl_3x2pt_5d)

    cov_sb_10d = utils.covariance_einsum(cl_3x2pt_5d, noise_3x2pt_5d, fsky,
                                         ells_4covsb, delta_ells_4covsb)
    bin_cov_sb_10d = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl_eff,
                               nbl_eff, zbins_use, zbins_use, zbins_use, zbins_use))

    # # ! This is quite ugly, find a way to vectorize, + avoid repeated code to bin the nmt/sb covariances
    z_combinations = list(itertools.product(range(zbins_use), repeat=4))
    for zi, zj, zk, zl in z_combinations:

        for i, block_name in enumerate(cov_blocks_names_all):
            probe_idxs = \
                probename_dict[block_name[0]], probename_dict[block_name[1]], \
                probename_dict[block_name[2]], probename_dict[block_name[3]]

            if cov_sb_10d[probe_idxs][:, :, zi, zj, zk, zl].shape != (nbl_eff, nbl_eff):
                print(f'Binning analytical Spaceborne {block_name} covariance')
                bin_cov_sb_10d[probe_idxs][:, :, zi, zj, zk, zl] = \
                    utils.bin_2d_matrix(cov=cov_sb_10d[probe_idxs][:, :, zi, zj, zk, zl],
                                        ells_in=ells_4covsb, ells_out=ells_eff,
                                        ells_out_edges=ells_eff_edges, weights=None,
                                        which_binning='mean')

        # TODO add the remaining blocks? not really necessary if I symmetrize...
        for i, block_name in enumerate(cov_nmt_block_names):

            probe_idxs = \
                probename_dict[block_name[0]], probename_dict[block_name[1]], \
                probename_dict[block_name[2]], probename_dict[block_name[3]]

            if cov_nmt_10d[probe_idxs][:, :, zi, zj, zk, zl].shape != (nbl_eff, nbl_eff):
                print(f'Binning analytical NaMaster {block_name} covariance')
                cov_nmt_10d[probe_idxs][:, :, zi, zj, zk, zl] = \
                    utils.bin_2d_matrix(cov=cov_nmt_10d[probe_idxs][:, :, zi, zj, zk, zl],
                                        ells_in=ells_tot, ells_out=ells_eff,
                                        ells_out_edges=ells_eff_edges, weights=None,
                                        which_binning='mean')

    zi, zj, zk, zl = 1, 1, 0, 1
    block = 'GLGL'
    # for zi, zj, zk, zl in z_combinations:

    # ! SAMPLE COVARIANCE - FROM NAMASTER DOCS
    if cfg['compute_namaster_sims']:
        probe = block[0] + block[1]
        cov_sims_nmt = sample_cov_nmt(zi, probe)

        # Let's plot the error bars (first and second diagonals)
        l_mid = get_lmid(ells_eff, k=1)
        plt.figure()
        plt.title('GG')
        plt.plot(ells_eff, np.sqrt(np.diag(cov_nmt_plt)), 'r-', label='Analytical, 1st-diag.')
        plt.plot(l_mid, np.sqrt(np.fabs(np.diag(cov_nmt_plt, k=1))), 'r--', label='Analytical, 2nd-diag.')
        plt.plot(ells_eff, np.sqrt(np.diag(cov_sims_nmt)), 'g-', label='Simulated, 1st-diag.')
        plt.plot(l_mid, np.sqrt(np.fabs(np.diag(cov_sims_nmt, k=1))), 'g--', label='Simulated, 2nd-diag.')
        plt.xlabel(r'$\ell$', fontsize=16)
        plt.ylabel(r'$\sigma(C_\ell)$', fontsize=16)
        plt.yscale('log')
        # plt.xscale('log')
        plt.legend(fontsize=12, frameon=False)
        plt.show()

    # ! SAMPLE COVARIANCE - DAVIDE
    if not cfg['load_simulated_cls']:
        print('Producing gaussian simulations...')
        simulated_cls_dict_ij = produce_gaussian_sims(cl_GG_unbinned[:, zi, zj],
                                                      cl_LL_unbinned[:, zi, zj],
                                                      cl_BB_unbinned[:, zi, zj],
                                                      cl_GL_unbinned[:, zi, zj],
                                                      nside=nside, nreal=nreal,
                                                      mask=mask,
                                                      coupled=coupled,
                                                      which_cls=cfg['which_cls'])

        if (zk, zl) == (zi, zj):
            simulated_cls_dict_kl = simulated_cls_dict_ij
        else:
            simulated_cls_dict_kl = produce_gaussian_sims(cl_GG_unbinned[:, zk, zl],
                                                          cl_LL_unbinned[:, zk, zl],
                                                          cl_BB_unbinned[:, zk, zl],
                                                          cl_GL_unbinned[:, zk, zl],
                                                          nside=nside, nreal=nreal,
                                                          mask=mask,
                                                          coupled=coupled,
                                                          which_cls=cfg['which_cls'])

        print('...done in {:.2f}s'.format(time.perf_counter() - start_time))
        np.save(f'../output/simulated_cls_dict_nreal{nreal}_{survey_area_deg2:.1f}deg2'
                f'_nside{nside}_which_cls{cfg["which_cls"]}_coupled{coupled}_INKA{use_INKA}_zi{zi}zj{zj}.npy',
                simulated_cls_dict_ij, allow_pickle=True)
        np.save(f'../output/simulated_cls_dict_nreal{nreal}_{survey_area_deg2:.1f}deg2'
                f'_nside{nside}_which_cls{cfg["which_cls"]}_coupled{coupled}_INKA{use_INKA}_zk{zk}zl{zl}.npy',
                simulated_cls_dict_ij, allow_pickle=True)

    elif cfg['load_simulated_cls']:
        simulated_cls_dict_ij = np.load(f'../output/simulated_cls_dict_nreal{nreal}_{survey_area_deg2:.1f}deg2'
                                        f'_nside{nside}_which_cls{cfg["which_cls"]}_coupled{coupled}_'
                                        f'INKA{use_INKA}_zi{zi}zj{zj}.npy',
                                        allow_pickle=True).item()
        simulated_cls_dict_ij = np.load(f'../output/simulated_cls_dict_nreal{nreal}_{survey_area_deg2:.1f}deg2'
                                        f'_nside{nside}_which_cls{cfg["which_cls"]}_coupled{coupled}_'
                                        f'INKA{use_INKA}_zk{zk}zl{zl}.npy',
                                        allow_pickle=True).item()

    sim_cl_GG_ij = simulated_cls_dict_ij['pseudo_cl_tt'][:, 0, :]
    sim_cl_GL_ij = simulated_cls_dict_ij['pseudo_cl_te'][:, 0, :]
    sim_cl_LL_ij = simulated_cls_dict_ij['pseudo_cl_ee'][:, 0, :]
    sim_cl_GG_kl = simulated_cls_dict_kl['pseudo_cl_tt'][:, 0, :]
    sim_cl_GL_kl = simulated_cls_dict_kl['pseudo_cl_te'][:, 0, :]
    sim_cl_LL_kl = simulated_cls_dict_kl['pseudo_cl_ee'][:, 0, :]
    sim_cl_list_ij = [sim_cl_GG_ij, sim_cl_GL_ij, sim_cl_LL_ij]
    sim_cl_list_kl = [sim_cl_GG_kl, sim_cl_GL_kl, sim_cl_LL_kl]

    # ! bin *before* computing the covariance
    # TODO check whether binning the covariance or the cls gives similar (same?) results
    for idx, sim_cls in enumerate(sim_cl_list_ij):
        if sim_cls.shape[1] != nbl_eff:
            print(f'Binning simulated cls into bandpowers for sim_cls[{idx}]...')
            bpw_sim_cls = np.zeros((nreal, nbl_eff))
            for i in range(nreal):
                bpw_sim_cls[i, :] = bin_obj.bin_cell(sim_cls[i, :])
            sim_cl_list_ij[idx] = bpw_sim_cls  # Update the list with the binned values
    for idx, sim_cls in enumerate(sim_cl_list_kl):
        if sim_cls.shape[1] != nbl_eff:
            print(f'Binning simulated cls into bandpowers for sim_cls[{idx}]...')
            bpw_sim_cls = np.zeros((nreal, nbl_eff))
            for i in range(nreal):
                bpw_sim_cls[i, :] = bin_obj.bin_cell(sim_cls[i, :])
            sim_cl_list_kl[idx] = bpw_sim_cls  # Update the list with the binned values

    # ! compute the sample covariance
    # TODO this is not the most efficient way of doing this, you could also cut the mixed cov terms
    cov_sims_LLLL = np.cov(sim_cl_LL_ij, sim_cl_LL_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    cov_sims_LLGL = np.cov(sim_cl_LL_ij, sim_cl_GL_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    cov_sims_LLGG = np.cov(sim_cl_LL_ij, sim_cl_GG_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    cov_sims_GLLL = np.cov(sim_cl_GL_ij, sim_cl_LL_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    cov_sims_GLGL = np.cov(sim_cl_GL_ij, sim_cl_GL_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    cov_sims_GLGG = np.cov(sim_cl_GL_ij, sim_cl_GG_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    cov_sims_GGLL = np.cov(sim_cl_GG_ij, sim_cl_LL_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    cov_sims_GGGL = np.cov(sim_cl_GG_ij, sim_cl_GL_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    cov_sims_GGGG = np.cov(sim_cl_GG_ij, sim_cl_GG_kl, rowvar=False, bias=False)[:nbl_eff, nbl_eff:]
    row_1 = np.hstack((cov_sims_LLLL, cov_sims_LLGL, cov_sims_LLGG))
    row_2 = np.hstack((cov_sims_GLLL, cov_sims_GLGL, cov_sims_GLGG))
    row_3 = np.hstack((cov_sims_GGLL, cov_sims_GGGL, cov_sims_GGGG))
    cov_sim_tot = np.vstack((row_1, row_2, row_3))

    probe_idxs = \
        probename_dict[block[0]], probename_dict[block[1]], \
        probename_dict[block[2]], probename_dict[block[3]]
    cov_nmt_plt = cov_nmt_10d[probe_idxs][:, :, zi, zj, zk, zl]
    cov_sb_plt = bin_cov_sb_10d[probe_idxs][:, :, zi, zj, zk, zl]

    # ! reshape the total 10d arrays to 4d and 2d
    ind_use = utils.build_full_ind(triu_tril, row_col_major, zbins_use)
    zpairs_auto_use, zpairs_cross_use, zpairs_3x2pt_use = utils.get_zpairs(zbins_use)
    ind_auto_use = ind_use[:zpairs_auto_use, :].copy()
    ind_cross_use = ind_use[zpairs_auto_use:zpairs_cross_use + zpairs_auto_use, :].copy()
    elem_auto_use = zpairs_auto_use * nbl_eff
    elem_autpluscross_use = (zpairs_auto_use + zpairs_cross_use) * nbl_eff

    cov_nmt_4d = utils.cov_3x2pt_10D_to_4D(cov_nmt_10d, probe_ordering,
                                           nbl_eff, zbins_use, ind_use.copy(), GL_or_LG)
    cov_sb_4d = utils.cov_3x2pt_10D_to_4D(bin_cov_sb_10d, probe_ordering,
                                          nbl_eff, zbins_use, ind_use.copy(), GL_or_LG)

    # ell-probe-zpair ordering
    # cov_nmt_2d = utils.cov_4D_to_2D(cov_nmt_4d, block_index='ij', optimize=True)
    # cov_sb_2d = utils.cov_4D_to_2D(cov_sb_4d, block_index='vincenzo', optimize=True)
    # probe-ell-zpair ordering
    cov_nmt_2d = utils.cov_4D_to_2DCLOE_3x2pt(cov_nmt_4d, zbins_use, block_index='ell')
    cov_nmt_2d = utils.symmetrize_2d_array(cov_nmt_2d)
    cov_sb_2d = utils.cov_4D_to_2DCLOE_3x2pt(cov_sb_4d, zbins_use, block_index='ell')
    # ! important note: the 2d covariance has all the blocks after symmetrixing, the 4D and 6D ones do not    

    # plot full covs and diag
    utils.matshow(cov_nmt_2d, 'dav', log=True, abs_val=True)
    utils.matshow(utils.percent_diff(cov_nmt_2d, cov_sb_2d), 'dav/sb [%]', log=True, abs_val=True, threshold=1)

    k_diag = 0
    diag_sb = np.diag(cov_sb_2d, k=k_diag)
    diag_nmt = np.diag(cov_nmt_2d, k=k_diag)
    fig, ax = plt.subplots(2, 1, sharex=True,
                           gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [2, 1]})
    ax[0].semilogy(diag_sb, label='sb')
    ax[0].semilogy(diag_nmt, ls='--', label='nmt')
    ax[1].plot(utils.percent_diff(diag_sb, diag_nmt), c='tab:green')
    ax[1].set_xlabel('total cov idx')
    ax[0].set_ylabel(f'diag total cov, k={k_diag}')
    ax[1].set_ylabel('nmt/sb [%]')
    ax[0].legend()
    fig.suptitle('Total cov diag: LL - GL - GG')

    # ! check different zij x zjk blocks
    if fsky == 1:
        for ell_idx in range(nbl_eff):
            np.testing.assert_allclose(cov_sb_4d[ell_idx, ell_idx, :, :], cov_nmt_4d[ell_idx, ell_idx, :, :],
                                       atol=0, rtol=1e-3)

    # check symmetry of different blocks in ell1, ell2
    print('checking symmetry of ell1xell1 covariance sub-blocks')
    for a, b, c, d in tqdm(z_combinations):
        for zi, zj, zk, zl in z_combinations:

            cov_block = cov_nmt_10d[a, b, c, d, :, :, zi, zj, zk, zl]
            try:
                np.testing.assert_allclose(cov_block, cov_block.T, atol=0, rtol=1e-2)
            except AssertionError:
                print(f'NMT cov_block', a, b, c, d, ' - ', zi, zj, zk, zl, ' not symmetric ❌')
                # utils.matshow(utils.percent_diff(cov_block, cov_block.T), log=False, abs_val=True, threshold=1)

            cov_block = cov_sb_10d[a, b, c, d, :, :, zi, zj, zk, zl]
            try:
                np.testing.assert_allclose(cov_block, cov_block.T, atol=0, rtol=1e-2)
            except AssertionError:
                print(f'SB cov_block', a, b, c, d, ' - ', zi, zj, zk, zl, ' not symmetric ❌')

    cov_LLLL_nmt_2d = cov_nmt_2d[:elem_auto_use, :elem_auto_use]
    cov_GLLL_nmt_2d = cov_nmt_2d[elem_auto_use:elem_autpluscross_use, :elem_auto_use]
    cov_GLGL_nmt_2d = cov_nmt_2d[elem_auto_use:elem_autpluscross_use, elem_auto_use:elem_autpluscross_use]
    cov_GLGG_nmt_2d = cov_nmt_2d[elem_auto_use:elem_autpluscross_use, elem_autpluscross_use:]
    cov_GGLL_nmt_2d = cov_nmt_2d[elem_autpluscross_use:, :elem_auto_use]
    cov_GGGL_nmt_2d = cov_nmt_2d[elem_autpluscross_use:, elem_auto_use:elem_autpluscross_use]
    cov_GGGG_nmt_2d = cov_nmt_2d[elem_autpluscross_use:, elem_autpluscross_use:]

    cov_LLLL_sb_2d = cov_sb_2d[:elem_auto_use, :elem_auto_use]
    cov_GLLL_sb_2d = cov_sb_2d[elem_auto_use:elem_autpluscross_use, :elem_auto_use]
    cov_GLGL_sb_2d = cov_sb_2d[elem_auto_use:elem_autpluscross_use, elem_auto_use:elem_autpluscross_use]
    cov_GLGG_sb_2d = cov_sb_2d[elem_auto_use:elem_autpluscross_use, elem_autpluscross_use:]
    cov_GGLL_sb_2d = cov_sb_2d[elem_autpluscross_use:, :elem_auto_use]
    cov_GGGL_sb_2d = cov_sb_2d[elem_autpluscross_use:, elem_auto_use:elem_autpluscross_use]
    cov_GGGG_sb_2d = cov_sb_2d[elem_autpluscross_use:, elem_autpluscross_use:]

    # check symemtry of square (on-diagonal) blocks
    utils.compare_arrays(cov_LLLL_nmt_2d, cov_LLLL_nmt_2d.T, 'cov_LLLL_nmt_2d',
                         'cov_LLLL_nmt_2d.T', abs_val=True, log_array=True, log_diff=True)
    utils.compare_arrays(cov_GLGL_nmt_2d, cov_GLGL_nmt_2d.T, 'cov_GLGL_nmt_2d',
                         'cov_GLGL_nmt_2d.T', abs_val=True, log_array=True, log_diff=True)
    utils.compare_arrays(cov_GGGG_nmt_2d, cov_GGGG_nmt_2d.T, 'cov_GGGG_nmt_2d',
                         'cov_GGGG_nmt_2d.T', abs_val=True, log_array=True, log_diff=True)

    # And the diagonals of the diagonal blocks
    np.testing.assert_allclose(np.diag(cov_LLLL_nmt_2d), np.diag(cov_LLLL_sb_2d), atol=0, rtol=1e-3)
    np.testing.assert_allclose(np.diag(cov_GLGL_nmt_2d), np.diag(cov_GLGL_sb_2d), atol=0, rtol=1e-3)
    np.testing.assert_allclose(np.diag(cov_GGGG_nmt_2d), np.diag(cov_GGGG_sb_2d), atol=0, rtol=1e-3)

    # check symmetry in AB, CD (I manually computed the GLGG block for this purpose)
    np.testing.assert_allclose(cov_GLGG_nmt_2d, cov_GGGL_nmt_2d.T, atol=0, rtol=1e-3)

    # check all blocks
    kw = dict(abs_val=True, log_array=True, log_diff=False, plot_diff_threshold=1)
    utils.compare_arrays(cov_LLLL_nmt_2d, cov_LLLL_sb_2d, 'cov_LLLL_nmt_2d', 'cov_LLLL_sb_2d', **kw)
    utils.compare_arrays(cov_GGGG_nmt_2d, cov_GGGG_sb_2d, 'cov_GGGG_nmt_2d', 'cov_GGGG_sb_2d', **kw)
    utils.compare_arrays(cov_GLGL_nmt_2d, cov_GLGL_sb_2d, 'cov_GLGL_nmt_2d', 'cov_GLGL_sb_2d', **kw)
    utils.compare_arrays(cov_GGLL_nmt_2d, cov_GGLL_sb_2d, 'cov_GGLL_nmt_2d', 'cov_GGLL_sb_2d', **kw)
    utils.compare_arrays(cov_GGGL_nmt_2d, cov_GGGL_sb_2d, 'cov_GGGL_nmt_2d', 'cov_GGGL_sb_2d', **kw)
    utils.compare_arrays(cov_GLLL_nmt_2d, cov_GLLL_sb_2d, 'cov_GLLL_nmt_2d', 'cov_GLLL_sb_2d', **kw)

    # ! plot main diagonal of full 2d covariance
    k_diag = 0
    diag_sb = np.diag(cov_nmt_2d, k=k_diag)
    diag_nmt = np.diag(cov_sb_2d, k=k_diag)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                           gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [2, 1]})
    ax[0].semilogy(diag_nmt, label='nmt')
    ax[0].semilogy(diag_sb, ls='--', label='sb')
    ax[1].plot(utils.percent_diff(diag_nmt, diag_sb))
    ax[1].set_xlabel('total cov idx')
    ax[0].set_ylabel(f'diag total cov, k={k_diag}')
    ax[1].set_ylabel('nmt/sb [%]')
    ax[0].legend()
    fig.suptitle('Total cov diag: LL - GL - GG')

    # construct ell1xell2 multiprobe cov (single zbin combination!!)
    row_1 = np.hstack((bin_cov_sb_10d[0, 0, 0, 0, :, :, zi, zj, zk, zl],
                       bin_cov_sb_10d[0, 0, 1, 0, :, :, zi, zj, zk, zl],
                       bin_cov_sb_10d[0, 0, 1, 1, :, :, zi, zj, zk, zl]))
    row_2 = np.hstack((bin_cov_sb_10d[1, 0, 0, 0, :, :, zi, zj, zk, zl],
                       bin_cov_sb_10d[1, 0, 1, 0, :, :, zi, zj, zk, zl],
                       bin_cov_sb_10d[1, 0, 1, 1, :, :, zi, zj, zk, zl]))
    row_3 = np.hstack((bin_cov_sb_10d[1, 1, 0, 0, :, :, zi, zj, zk, zl],
                       bin_cov_sb_10d[1, 1, 1, 0, :, :, zi, zj, zk, zl],
                       bin_cov_sb_10d[1, 1, 1, 1, :, :, zi, zj, zk, zl]))
    cov_sb_tot = np.vstack((row_1, row_2, row_3))

    zeros_block = np.zeros((nbl_eff, nbl_eff))
    row_1 = np.hstack((cov_nmt_10d[0, 0, 0, 0, :, :, zi, zj, zk, zl],
                       zeros_block,
                       zeros_block))
    row_2 = np.hstack((cov_nmt_10d[1, 0, 0, 0, :, :, zi, zj, zk, zl],
                       cov_nmt_10d[1, 0, 1, 0, :, :, zi, zj, zk, zl],
                       zeros_block))
    row_3 = np.hstack((cov_nmt_10d[1, 1, 0, 0, :, :, zi, zj, zk, zl],
                       cov_nmt_10d[1, 1, 1, 0, :, :, zi, zj, zk, zl],
                       cov_nmt_10d[1, 1, 1, 1, :, :, zi, zj, zk, zl]))
    cov_nmt_tot = np.vstack((row_1, row_2, row_3))
    cov_nmt_tot = utils.symmetrize_2d_array(cov_nmt_tot)

    utils.matshow(cov_sim_tot, log=True, abs_val=True, title=f'sim tot covariance matrix, z = {zi}, {zj}, {zk}, {zl}')
    utils.matshow(cov_sb_tot, log=True, abs_val=True, title=f'SB tot covariance matrix, z = {zi}, {zj}, {zk}, {zl}')
    utils.matshow(cov_nmt_tot, log=True, abs_val=True, title=f'nmt tot covariance matrix, z = {zi}, {zj}, {zk}, {zl}')

    # check inversion of different blocks and "total" cov (still missing zbins combinations!!)
    print('Testing inversion of the covariance blocks...')
    for probe_idx in probe_idxs:

        block_name = \
            probename_dict_inv[str(probe_idxs[0])], probename_dict_inv[str(probe_idxs[1])], \
            probename_dict_inv[str(probe_idxs[2])], probename_dict_inv[str(probe_idxs[3])]
        block_name = ''.join(block_name)
        try:
            covar_inv = np.linalg.inv(cov_nmt_10d[probe_idxs])
            np.linalg.cholesky(cov_nmt_10d[probe_idxs])

            print(f'nmt block {block_name} is invertible!')
        except np.linalg.LinAlgError as err:
            print(f'nmt block {block_name} is not invertible: {err}')

    for cov, cov_name in zip((cov_sb_tot, cov_nmt_tot, cov_sim_tot), ('sb', 'nmt', 'sim')):
        try:
            covar_inv = np.linalg.inv(cov)
            np.linalg.cholesky(cov)
            print(f'Cov {cov_name} is invertible!')
        except np.linalg.LinAlgError as err:
            print(f'Cov {cov_name} is not invertible: {err}')

    for cov, cov_name in zip((cov_sb_2d, cov_nmt_2d), ('sb 2d full', 'nmt 2d full')):
        try:
            covar_inv = np.linalg.inv(cov)
            print(f'Numpy inversion performed for {cov_name}')
        except np.linalg.LinAlgError as err:
            print(f'Numpy inversion failed for {cov_name}: {err}')

        try:
            np.linalg.cholesky(cov)
            print(f'Cholesky decomposition performed for {cov_name}')
        except np.linalg.LinAlgError as err:
            print(f'Cholesky decomposition failed for {cov_name}: {err}')

    # ! PLOT SIMS for a quick check against theoy cls
    # TODO add kl?
    if block == 'GGGG':
        cl_plt = cl_GG_unbinned[:, zi, zj]
        sim_cls_plt = sim_cl_GG_ij
        cov_sims_plt = cov_sims_GGGG
    elif block == 'GLGL':
        cl_plt = cl_GL_unbinned[:, zi, zj]
        sim_cls_plt = sim_cl_GL_ij
        cov_sims_plt = cov_sims_GLGL
    elif block == 'LLLL':
        cl_plt = cl_LL_unbinned[:, zi, zj]
        sim_cls_plt = sim_cl_LL_ij
        cov_sims_plt = cov_sims_LLLL

    plt.figure()
    count = 0
    for i in range(nreal)[:100:5]:
        plt.semilogy(ells_eff, sim_cls_plt[i, :], label=f'simulated {coupled_label} cls' if count == 0 else '',
                     marker='.')
        count += 1
    plt.loglog(cl_plt, label='theory cls', c='tab:orange')
    plt.loglog(cl_plt * fsky, label='theory cls*fsky', c='k', ls='--')
    plt.axvline(lmax_healpy_safe, c='k', ls='--', label='1.5 * nside')
    plt.legend()
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell$')
    plt.tight_layout()

    # ! PLOT DIAGONAL,
    # no delta_ell if you're using the pseudo-cls in the gaussian_simulations func!!

    clr = cm.plasma(np.linspace(0, 1, 5))
    label = r'cov_{code:s}, $\ell^\prime=\ell+{off_diag:d}$'
    diag_label = '$\\ell^\prime=\\ell$'
    title = f'cov {block}\nsurvey_area = {survey_area_deg2} deg2\nlinear binning,'\
        f' $\Delta\ell={delta_ells_bpw[0]:.1f}$, use_INKA {use_INKA}' \
        f'\nzi={zi}, zj={zj}, zk={zk}, zl={zl}'
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                           gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [2, 1]})
    ax[0].set_title(title)
    ax[0].loglog(ells_eff, np.diag(cov_sb_plt), label=f'cov_sb/fsky, {diag_label}', marker='.', c='tab:orange')
    ax[0].loglog(ells_eff, np.diag(cov_nmt_plt), label=f'cov_nmt, {diag_label}', marker='.', c=clr[0])
    ax[0].loglog(ells_eff, np.diag(cov_sims_plt), label=f'cov_sims, {diag_label}', marker='.', c=clr[1])
    # ax[0].loglog(ells_eff, np.diag(cov_sims_nmt), label=f'cov_sims_nmt, {diag_label}', marker='.', c=clr[0], ls=':')

    for k in range(1, 2):
        diag_nmt = np.diag(cov_nmt_plt, k=k)
        diag_sim = np.diag(cov_sims_plt, k=k)
        l_mid = get_lmid(ells_eff, k)
        l_mid_tot = get_lmid(ells_tot, k)
        # ls_nmt = '--' if np.all(diag_nmt < 0) else '-'
        # ls_sim = '--' if np.all(diag_sim < 0) else '-'
        ls_nmt = '-'
        ls_sim = '--'
        # diag_nmt = np.fabs(diag_nmt) if np.all(diag_nmt < 0) else diag_nmt
        # diag_sim = np.fabs(diag_sim) if np.all(diag_sim < 0) else diag_sim
        diag_nmt = np.fabs(diag_nmt)
        diag_sim = np.fabs(diag_sim)
        ax[0].loglog(l_mid, diag_nmt, label='abs ' + label.format(code='nmt',
                     off_diag=k), ls='--', c=clr[0], marker='.')
        ax[0].loglog(l_mid, diag_sim, label='abs ' + label.format(code='sim',
                     off_diag=k), ls='--', c=clr[1], marker='.')

    ax[1].plot(ells_eff, utils.percent_diff(np.diag(cov_sb_plt), np.diag(cov_nmt_plt)),
               marker='.', label='sb/nmt', c='tab:orange')
    # ax[1].plot(ells_eff, utils.percent_diff(np.diag(cov_sims_plt), np.diag(cov_nmt_plt)),
    #    marker='.', label='sim/nmt', c=clr[0], ls='--')
    # ax[1].plot(ells_eff, utils.percent_diff(np.diag(cov_sims_nmt), np.diag(cov_nmt)),
    #    marker='.', label='sims_nmt/nmt', c=clr[0], ls=':')

    ax[1].set_ylabel('% diff cov fsky/part_sky')
    ax[1].set_xlabel(r'$\ell$')
    ax[1].fill_between(ells_eff, -10, 10, color='k', alpha=0.1)
    ax[1].axhline(y=0, color='k', alpha=0.5, ls='--')
    ax[0].axvline(lmax_healpy_safe, color='k', alpha=0.5, ls='--', label='1.5 * nside')
    ax[1].axvline(lmax_healpy_safe, color='k', alpha=0.5, ls='--')
    ax[0].axvline(2 * nside, color='k', alpha=0.5, ls=':', label='2 * nside')
    ax[1].axvline(2 * nside, color='k', alpha=0.5, ls=':')
    ax[1].legend()
    ax[0].set_ylabel('diag cov')
    ax[0].legend()
    # ax[0].set_ylim(1e-25, 1e-10)
    # ax[0].set_xlim(10, 1700)

    # ! plot whole covmat, for zi = zj = zk = zl = 0
    corr_nmt = utils.cov2corr(cov_nmt_plt)
    corr_sb = utils.cov2corr(cov_sb_plt)
    corr_sims = utils.cov2corr(cov_sims_plt)

    threshold = 10  # percent
    cov_abs_diff_sims = np.fabs(utils.percent_diff(cov_sims_plt, cov_nmt_plt))
    cor_abs_diff_sims = np.fabs(utils.percent_diff(corr_sims, corr_nmt))
    mask_cov_abs_diff_sims = np.where(cov_abs_diff_sims < threshold, np.nan, cov_abs_diff_sims)
    mask_corr_abs_diff_sims = np.where(cor_abs_diff_sims < threshold, np.nan, cor_abs_diff_sims)

    fig, ax = plt.subplots(4, 2, figsize=(10, 14))
    # covariance
    cax0 = ax[0, 0].matshow(np.log10(np.fabs(cov_sb_plt)))
    cax2 = ax[1, 0].matshow(np.log10(np.fabs(cov_nmt_plt)))
    cax3 = ax[2, 0].matshow(np.log10(np.fabs(cov_sims_plt)))
    cax4 = ax[3, 0].matshow(np.log10(mask_cov_abs_diff_sims))
    ax[0, 0].set_title(f'log10 abs \nfull_sky/fsky cov')
    ax[1, 0].set_title(f'log10 abs \nNaMaster cov')
    ax[2, 0].set_title(f'log10 abs \nsim cov')
    ax[3, 0].set_title(f'log10 abs \nsim/nmt [%]\n{threshold}% threshold')
    fig.colorbar(cax0, ax=ax[0, 0])
    fig.colorbar(cax2, ax=ax[1, 0])
    fig.colorbar(cax3, ax=ax[2, 0])
    fig.colorbar(cax4, ax=ax[3, 0])

    # correlation (common colorbar)
    cbar_corr_1 = ax[0, 1].matshow(corr_sb, vmin=-1, vmax=1, cmap='RdBu_r')
    cbar_corr_2 = ax[1, 1].matshow(corr_nmt, vmin=-1, vmax=1, cmap='RdBu_r')  # Apply same cmap and limits
    cbar_corr_3 = ax[2, 1].matshow(corr_sims, vmin=-1, vmax=1, cmap='RdBu_r')  # Apply same cmap and limits
    cbar_corr_4 = ax[3, 1].matshow(np.log10(mask_corr_abs_diff_sims), cmap='RdBu_r')  # Apply same cmap and limits
    ax[0, 1].set_title(f'full_sky/fsky corr')
    ax[1, 1].set_title(f'NaMaster corr')
    ax[2, 1].set_title(f'sim corr')
    ax[3, 1].set_title(f'log10 abs \nsim/nmt [%]\n{threshold}% threshold')
    fig.colorbar(cbar_corr_1, ax=ax[0, 1])
    fig.colorbar(cbar_corr_2, ax=ax[1, 1])
    fig.colorbar(cbar_corr_3, ax=ax[2, 1])
    fig.colorbar(cbar_corr_4, ax=ax[3, 1])

    # Adjust layout to make room for colorbars
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    assert False, 'stop here to check partial-sky cov'

    # Bandpower info:
    print("Bandpower info:")
    print(" %d bandpowers" % (bin_obj.get_n_bands()))
    print("The columns in the following table are:")
    print("[1]=band index, [2]=list of multipoles,"
          "[3]=list of weights, [4]=effective multipole")
    for i in range(bin_obj.get_n_bands()):
        print(i, bin_obj.get_ell_list(i), bin_obj.get_weight_list(i), ells_eff[i])
    print("")

    # Bin a power spectrum into bandpowers. This is carried out as a weighted
    # average over the multipoles in each bandpower.
    cl_GG_3D_binned = np.array([[bin_obj.bin_cell(np.array([cl_GG_unbinned[:lmax, zi, zj]]))[0]
                               for zi in range(zbins)]
                                for zj in range(zbins)]).transpose((2, 0, 1))

    # Un-bins a set of bandpowers into a power spectrum. This is simply done by assigning a
    # constant value for every multipole in each bandpower.
    cl_GG_3D_binned_unbinned = np.array([[bin_obj.unbin_cell(cl_GG_3D_binned[:lmax, zi, zj])
                                        for zi in range(zbins)]
                                         for zj in range(zbins)]).transpose((2, 0, 1))

    # print('computing MASTER estimator for spin-0 x spin-0...')
    # start_time = time.perf_counter()
    # Computes the full MASTER estimate of the power spectrum of two fields (f1 and f2).
    # It represents the measured power spectrum after correcting for the mask and other observational effects.
    # cl_GG_3D_measured = np.array([[nmt.compute_full_master(f0[zi], f0[zj], bin_obj)[0]
    #                                for zi in range(zbins)]
    #                               for zj in range(zbins)]).transpose((2, 0, 1))
    # print('done in {:.2f} s'.format(time.perf_counter() - start_time))

    # Compute predictions
    # this is a general workspace that can be used for any spin combination, as it only depends on the survey geometry;
    # it is typically used for general coupling matrix computations and can be applied to
    # decouple power spectra for any spin combination
    # -
    # w_mask.decouple_cell decouples a set of pseudo-C_\ell power spectra into a set of bandpowers by inverting the binned
    # coupling matrix (se Eq. 16 of the NaMaster paper).
    # this is a bandpower cls as well, but after correcting for the mask
    bpow_GG_3D = np.array([[w00.decouple_cell(w00.couple_cell([cl_GG_unbinned[:lmax + 1, zi, zj]]))[0]
                          for zi in range(zbins)]
                           for zj in range(zbins)]).transpose((2, 0, 1))

    # These represent the pseudo-power spectra, which are the raw power spectra measured
    # on the masked sky without any corrections. These are computed on the masked sky, so the power is lower!!!
    # Convolves the true Cl with the coupling matrix due to the mask (pseudo-spectrum).
    pseudoCl_GG_3d_1 = np.array([[w00.couple_cell(cl_GG_unbinned[:lmax + 1, zi, zj][None, :])[0]
                                for zi in range(zbins)]
                                 for zj in range(zbins)]).transpose((2, 0, 1))
    # directly computes the pseudo-Cl from the field maps.
    pseudoCl_GG_3d_2 = np.array([[nmt.compute_coupled_cell(f0[zi], f0[zj])[0]
                                for zi in range(zbins)]
                                 for zj in range(zbins)]).transpose((2, 0, 1))

    # Plot results
    plt.figure()

    plt.plot(ells_unbinned[:lmax], cl_GG_unbinned[:, zi, zj], label=r'Original $C_\ell$')
    plt.plot(ells_eff, cl_GG_3D_binned[:, zi, zj], ls='', c='C1', label=r'Binned $C_\ell$', marker='o', alpha=0.6)
    # plt.plot(ells_unbinned[:lmax], cl_GG_3D_binned_unbinned[:, zi, zj],
    #  label=r'Binned-unbinned $C_\ell$', alpha=0.6)

    # plt.scatter(ell_eff, cl_GG_3D_measured[:, zi, zj], label=r'Reconstructed $C_\ell$', marker='.', alpha=0.6)
    # plt.plot(ells_eff, bpow_GG_3D[:, zi, zj], label=r'Bandpower $C_\ell$', alpha=0.6)
    # plt.plot(ells_unbinned[:lmax], pseudoCl_GG_3d_1[:, zi, zj]/fsky, label=r'pseudo $C_\ell$/fsky', alpha=0.6)
    # plt.plot(ells_unbinned[:lmax], pseudoCl_GG_3d_2[:, zi, zj]/fsky, label=r'pseudo $C_\ell$/fsky', alpha=0.6)

    plt.axvline(x=lmax, ls='--', c='k', label=r'$\ell_{max}$ healpy')
    plt.axvline(x=lmax, ls='--', c='k', label=r'$\ell_{max}$ mask')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$C_\ell$')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(f'zi, zj = ({zi}, {zj})')
    plt.show()

    assert False, 'stop here to check part sky'

    # ! end, dav

    cov_3x2pt_2D = utils.covariance_nmt(cl_3x2pt_5D, noise_3x2pt_5D, workspace_path, mask_path)
    print(f'covariance computation took {time.perf_counter() - start:.2f} seconds')

# else:
#     print('Computing the full-sky covariance divided by f_sky')
#     cov_3x2pt_10D_arr = utils.covariance_einsum(cl_3x2pt_5D, noise_3x2pt_5D, fsky, ell_values, delta_values)
#     print(f'covariance computation took {time.perf_counter() - start:.2f} seconds')

# # reshape to 4D
# cov_3x2pt_10D_dict = utils.cov_10D_array_to_dict(cov_3x2pt_10D_arr)
# cov_3x2pt_4D = utils.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_10D_dict, probe_ordering, nbl_eff, zbins, ind.copy(),
#                                               GL_or_LG)
# del cov_3x2pt_10D_dict, cov_3x2pt_10D_arr
# gc.collect()

# # reshape to 2D
# # if not cfg['use_2DCLOE']:
# #     cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)
# # elif cfg['use_2DCLOE']:
# #     cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index='ell')
# # else:
# #     raise ValueError('use_2DCLOE must be a true or false')

# if covariance_ordering_2D == 'probe_ell_zpair':
#     use_2DCLOE = True
#     block_index = 'ell'
#     cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index=block_index)

# elif covariance_ordering_2D == 'probe_zpair_ell':
#     use_2DCLOE = True
#     block_index = 'ij'
#     cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index=block_index)

# elif covariance_ordering_2D == 'ell_probe_zpair':
#     use_2DCLOE = False
#     block_index = 'ell'
#     cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)

# elif covariance_ordering_2D == 'zpair_probe_ell':
#     use_2DCLOE = False
#     block_index = 'ij'
#     cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)

# else:
#     raise ValueError('covariance_ordering_2D must be a one of the following: probe_ell_zpair, probe_zpair_ell,'
#                      'ell_probe_zpair, zpair_probe_ell')

# if cfg['plot_covariance_2D']:
#     plt.matshow(np.log10(cov_3x2pt_2D))
#     plt.colorbar()
#     plt.title(f'log10(cov_3x2pt_2D)\nordering: {covariance_ordering_2D}')

# other_quantities_tosave = {
#     'n_gal_shear [arcmin^{-2}]': n_gal_shear,
#     'n_gal_clustering [arcmin^{-2}]': n_gal_clustering,
#     'survey_area [deg^2]': survey_area_deg2,
#     'sigma_eps': sigma_eps,
# }

# np.save(f'{output_folder}/cov_Gauss_3x2pt_2D_{covariance_ordering_2D}.npy', cov_3x2pt_2D)

# with open(f'{output_folder}/other_specs.txt', 'w') as file:
#     file.write(json.dumps(other_quantities_tosave))

# print(f'Done')
# print(f'Covariance files saved in {output_folder}')

# # ! Plot covariance
