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


def masked_maps_to_nmtFields(map_T, map_Q, map_U, mask, lmax, n_iter=0, lite=True):
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
    f0 = nmt.NmtField(mask, [map_T], n_iter=n_iter, lite=lite, lmax=lmax)
    f2 = nmt.NmtField(mask, [map_Q, map_U], spin=2, n_iter=n_iter, lite=lite, lmax=lmax)
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

def produce_gaussian_sims(cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB, zi, zj, nreal, nside, mask, coupled, which_cls, lmax):

    # both healpy anafast and nmt.compute_coupled_cell return the coupled cls. Dividing by fsky gives a rough
    # approximation of the true Cls
    correction_factor = 1. if coupled else fsky

    pseudo_cl_tt_list = []
    pseudo_cl_te_list = []
    pseudo_cl_ee_list = []

    print(f'Generating {nreal} maps for nside {nside}...')

    cl_ring_big_list = build_cl_tomo_TEB_ring_ord(
        cl_TT=cl_TT,
        cl_EE=cl_EE,
        cl_BB=cl_BB,
        cl_TE=cl_TE,
        cl_EB=cl_EB,
        cl_TB=cl_TB,
        zbins=zbins_use,
        spectra_types=['T', 'E', 'B'])

    corr_maps_gg_list = []
    corr_maps_ll_list = []

    for _ in tqdm(range(nreal)):

        corr_alms_tot = hp.synalm(cl_ring_big_list, lmax=3 * nside - 1, new=True)
        assert len(corr_alms_tot) == zbins_use * 3, 'wrong number of alms'

        # extract alm for TT, EE, BB
        corr_alms = corr_alms_tot[::3]
        corr_Elms_Blms = list(zip(corr_alms_tot[1::3], corr_alms_tot[2::3]))

        # compute correlated maps for each bin
        corr_maps_gg = [hp.alm2map(alm, nside) for alm in corr_alms]
        corr_maps_ll = [hp.alm2map_spin([Elm, Blm], nside, 2, 3 * nside - 1) for (Elm, Blm) in corr_Elms_Blms]

        corr_maps_gg_list.append(corr_maps_gg)
        corr_maps_ll_list.append(corr_maps_ll)

    return corr_maps_gg_list, corr_maps_ll_list


def pcls_from_maps(corr_maps_gg, corr_maps_ll, zi, zj, mask, coupled_cls, which_cls):

    # both healpy anafast and nmt.compute_coupled_cell return the coupled cls. Dividing by fsky gives a rough
    # approximation of the true Cls
    correction_factor = 1. if coupled_cls else fsky

    if which_cls == 'namaster':

        f0 = np.array([nmt.NmtField(mask, [map_T], n_iter=3, lite=True)
                       for map_T in corr_maps_gg])
        f2 = np.array([nmt.NmtField(mask, [map_Q, map_U], n_iter=3, lite=True)
                       for (map_Q, map_U) in corr_maps_ll])

        if coupled_cls:  # ! TODO fix this!!
            # pseudo-Cls. Becomes an ok estimator for the true Cls if divided by fsky
            pseudo_cl_tt = nmt.compute_coupled_cell(f0[zi], f0[zj])[0] / correction_factor
            pseudo_cl_te = nmt.compute_coupled_cell(f0[zi], f2[zj])[0] / correction_factor
            pseudo_cl_ee = nmt.compute_coupled_cell(f2[zi], f2[zj])[0] / correction_factor
        else:
            # best estimator for the true Cls
            pseudo_cl_tt = compute_master(f0[zi], f0[zj], w00)[0, :]
            pseudo_cl_te = compute_master(f0[zi], f2[zj], w02)[0, :]
            pseudo_cl_ee = compute_master(f2[zi], f2[zj], w22)[0, :]

    elif which_cls == 'healpy':

        _corr_maps_zi = list(itertools.chain([corr_maps_gg[zi]], corr_maps_ll[zi]))
        _corr_maps_zj = list(itertools.chain([corr_maps_gg[zj]], corr_maps_ll[zj]))
        # 2. remove monopole
        _corr_maps_zi = [hp.remove_monopole(_corr_maps_zi[spec_ix]) for spec_ix in range(3)]
        _corr_maps_zj = [hp.remove_monopole(_corr_maps_zj[spec_ix]) for spec_ix in range(3)]
        # 3. compute cls for each bin
        hp_pcl_tot = hp.anafast(map1=[_corr_maps_zi[0] * mask, _corr_maps_zi[1] * mask, _corr_maps_zi[2] * mask],
                                map2=[_corr_maps_zj[0] * mask, _corr_maps_zj[1] * mask, _corr_maps_zj[2] * mask])
        # output is TT, EE, BB, TE, EB, TB
        # hp_pcl_GG[:, zi, zj] = hp_pcl_tot[0, :]
        # hp_pcl_LL[:, zi, zj] = hp_pcl_tot[1, :]
        # hp_pcl_GL[:, zi, zj] = hp_pcl_tot[3, :]

        # pseudo-Cls. Becomes an ok estimator for the true Cls if divided by fsky
        pseudo_cl_tt = hp_pcl_tot[0, :]
        pseudo_cl_ee = hp_pcl_tot[1, :]
        pseudo_cl_bb = hp_pcl_tot[2, :]
        pseudo_cl_te = hp_pcl_tot[3, :]
        pseudo_cl_eb = hp_pcl_tot[4, :]
        pseudo_cl_tb = hp_pcl_tot[5, :]
        pseudo_cl_be = pseudo_cl_eb  # ! warning!!
        if not coupled_cls:

            pseudo_cl_tt = w00.decouple_cell(pseudo_cl_tt[None, :])[0, :]
            pseudo_cl_ee = w22.decouple_cell(np.vstack((pseudo_cl_ee, pseudo_cl_eb, pseudo_cl_be, pseudo_cl_bb)))[0, :]
            pseudo_cl_te = w02.decouple_cell(np.vstack((pseudo_cl_te, pseudo_cl_tb)))[0, :]

    else:
        raise ValueError('which_cls must be namaster or healpy')

    return np.array(pseudo_cl_tt), np.array(pseudo_cl_te), np.array(pseudo_cl_ee)


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

print('Computing the partial-sky covariance with NaMaster')

# ! =============================================== IMPLEMENTATION BY DAVIDE =======================================
# TODO check implementation by R. Upham: https://github.com/robinupham/shear_pcl_cov/blob/main/shear_pcl_cov/gaussian_cov.py
import healpy as hp
import pymaster as nmt

ells_unbinned = np.arange(cl_LL_unbinned.shape[0])
ells_per_band = cfg['ells_per_band']
nside = cfg['nside']
zbins_use = cfg['zbins_use']
use_INKA = cfg['use_INKA']
which_cls = cfg['which_cls']

if use_INKA and cfg['coupled_nmt_cov'] :
    raise ValueError('Cannot do iNKA for coupled Cls covariance.')


# read or generate mask
if cfg['read_mask']:
    if mask_path.endswith('.fits'):
        mask = hp.read_map(mask_path)
        mask = np.where(np.logical_and(mask <= utils.DR1_DATE, mask >= 0.), 1., 0,)
    elif mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    mask = hp.ud_grade(mask, nside_out=nside)

else:
    # mask = utils.generate_polar_cap(area_deg2=survey_area_deg2, nside=cfg['nside'])
    mask = utils.generate_survey_mask(area_deg2=survey_area_deg2,
                                        nside=cfg['nside'],
                                        shape=cfg['mask_shape'])

fsky = np.mean(mask**2)
survey_area_deg2 = fsky * utils.DEG2_IN_SPHERE

# TODO check np.all(mask == 1)

# apodize
# hp.mollview(mask, title='before apodization', cmap='inferno_r')
if cfg['apodize_mask'] and int(survey_area_deg2) != int(utils.DEG2_IN_SPHERE):
    mask = nmt.mask_apodization(mask, aposize=cfg['aposize'], apotype="Smooth")
    # hp.mollview(mask, title='after apodization', cmap='inferno_r')

# recompute after apodizing
fsky = np.mean(mask**2)
survey_area_deg2 = fsky * utils.DEG2_IN_SPHERE

npix = hp.nside2npix(nside)
pix_area = 4 * np.pi

# check fsky and nside
nside_from_mask = hp.get_nside(mask)
assert nside_from_mask == cfg['nside'], 'nside from mask is not consistent with the desired nside in the cfg file'


# get lmin: quick estimate
survey_area_rad = np.sum(mask) * hp.nside2pixarea(nside)
lmin_mask = int(np.ceil(np.pi / np.sqrt(survey_area_rad)))

# ! Define the set of bandpowers used in the computation of the pseudo-Cl
# Initialize binning scheme with bandpowers of constant width (ells_per_band multipoles per bin)
# TODO use lmax_mask instead of nside? Decide which binning scheme is the best
# ell_values, delta_values, ell_bin_edges = utils.compute_ells(nbl, 0, lmax, recipe='ISTF', output_ell_bin_edges=True)
# bin_obj = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=ells_per_band, is_Dell=False, f_ell=None) # TODO test this
# bin_obj = nmt.NmtBin.from_nside_linear(nside, ells_per_band)
bin_obj = nmt.NmtBin.from_edges(
ell_bin_lower_edges.astype(int),
ell_bin_upper_edges.astype(int), is_Dell=False, f_ell=None)

# set different possible values for lmax
lmax_mask = int(np.pi / hp.pixelfunc.nside2resol(nside))
lmax = bin_obj.lmax + 1

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
# assert np.all(delta_ells_bpw == ells_per_band), 'delta_ell from bpw does not match ells_per_band'

# ! create nmt field from the mask (there will be no maps associated to the fields)
# TODO maks=None (as in the example) or maps=[mask]? I think None
start_time = time.perf_counter()
print('computing coupling coefficients...')
f0_mask = nmt.NmtField(mask=mask, maps=None, spin=0, lite=True, lmax=bin_obj.lmax)
f2_mask = nmt.NmtField(mask=mask, maps=None, spin=2, lite=True, lmax=bin_obj.lmax)
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

print('lmin_mask:', lmin_mask)
print('lmax_mask:', lmax_mask)
print('lmax_bin_obj:',  bin_obj.lmax)
print('nside:', nside)
print('fsky after apodization:', fsky)

# cut and bin the theory
cl_GG_unbinned = cl_GG_unbinned[:lmax, :zbins_use, :zbins_use]
cl_GL_unbinned = cl_GL_unbinned[:lmax, :zbins_use, :zbins_use]
cl_LL_unbinned = cl_LL_unbinned[:lmax, :zbins_use, :zbins_use]
cl_BB_unbinned = np.zeros_like(cl_LL_unbinned)
cl_TB_unbinned = np.zeros_like(cl_LL_unbinned)
cl_EB_unbinned = np.zeros_like(cl_LL_unbinned)

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

if use_INKA:
    cl_GG_4covnmt = np.zeros_like(cl_GG_unbinned)
    cl_GL_4covnmt = np.zeros_like(cl_GL_unbinned)
    cl_LL_4covnmt = np.zeros_like(cl_LL_unbinned)
    for zi in range(zbins_use):
        for zj in range(zbins_use):
            # setting B-mode related spcectra to 0
            cl_GG_4covnmt[:, zi, zj] = w00.couple_cell([cl_GG_unbinned[:, zi, zj]])[0] / fsky
            cl_GL_4covnmt[:, zi, zj] = w02.couple_cell([cl_GL_unbinned[:, zi, zj],
                                                        np.zeros_like(cl_GL_unbinned[:, zi, zj])])[0] / fsky
            cl_LL_4covnmt[:, zi, zj] = w22.couple_cell([cl_LL_unbinned[:, zi, zj],
                                                        np.zeros_like(cl_LL_unbinned[:, zi, zj]),
                                                        np.zeros_like(cl_LL_unbinned[:, zi, zj]),
                                                        np.zeros_like(cl_LL_unbinned[:, zi, zj])])[0] / fsky

else:
    cl_GG_4covnmt = cl_GG_unbinned
    cl_GL_4covnmt = cl_GL_unbinned
    cl_LL_4covnmt = cl_LL_unbinned

cl_tt = cl_GG_4covnmt
cl_te = cl_GL_4covnmt
cl_ee = cl_LL_4covnmt
cl_tb = np.zeros_like(cl_GG_4covnmt)
cl_eb = np.zeros_like(cl_GG_4covnmt)
cl_bb = np.zeros_like(cl_GG_4covnmt)

# ! NAMASTER covariance
if cfg['spin0']:
    if cfg['coupled_nmt_cov']:
        cov_nmt_10d = utils.nmt_gaussian_cov_spin0_coupled(cl_tt=cl_tt,
                                                           cl_te=cl_te,
                                                           cl_ee=cl_ee,
                                                           zbins=zbins_use,
                                                           nbl=nbl_eff,
                                                           cw=cw, w00=w00,
                                                           ells_in=ells_tot,
                                                           ells_out=ells_eff,
                                                           ells_out_edges=ells_eff_edges,
                                                           weights=None,
                                                           which_binning='mean')
    else:
        cov_nmt_10d = utils.nmt_gaussian_cov_spin0(cl_tt=cl_tt,
                                                   cl_te=cl_te,
                                                   cl_ee=cl_ee,
                                                   zbins=zbins_use,
                                                   nbl=nbl_eff,
                                                   cw=cw, w00=w00)


else:
    if cfg['coupled_nmt_cov']:
        cov_nmt_10d = utils.nmt_gaussian_cov_coupled(cl_tt=cl_tt, cl_te=cl_te,
                                                     cl_ee=cl_ee, cl_tb=cl_tb,
                                                     cl_eb=cl_eb, cl_bb=cl_bb,
                                                     zbins=zbins_use,
                                                     nbl=nbl_eff,
                                                     cw=cw, w00=w00, w02=w02, w22=w22,
                                                     compute_all_blocks=cfg['compute_all_blocks'],
                                                     ells_in=ells_tot,
                                                     ells_out=ells_eff,
                                                     ells_out_edges=ells_eff_edges,
                                                     weights=None,
                                                     which_binning='mean')

    else:
        cov_nmt_10d = utils.nmt_gaussian_cov(cl_tt=cl_tt, cl_te=cl_te, cl_ee=cl_ee,
                                            cl_tb=cl_tb, cl_eb=cl_eb, cl_bb=cl_bb,
                                            zbins=zbins_use,
                                            nbl=nbl_eff,
                                            cw=cw, w00=w00, w02=w02, w22=w22,
                                            compute_all_blocks=cfg['compute_all_blocks'])

probename_dict = {
    'L': 0,
    'G': 1,
}
probename_dict_inv = {
    '0': 'L',
    '1': 'G',
}

# # # ! BIN COVARIANCE MATRICES IF NEEDED
# # # ! This is quite ugly, find a way to vectorize, + avoid repeated code to bin the nmt/sb covariances
# z_combinations = list(itertools.product(range(zbins_use), repeat=4))
# cov_nmt_10d_binned = np.zeros((2, 2, 2, 2, nbl_eff, nbl_eff, zbins_use, zbins_use, zbins_use, zbins_use))
# for zi, zj, zk, zl in z_combinations:
#     for i, block_name in enumerate(cov_blocks_names_all):
#         probe_idxs = \
#             probename_dict[block_name[0]], probename_dict[block_name[1]], \
#             probename_dict[block_name[2]], probename_dict[block_name[3]]

#         if cov_nmt_10d[probe_idxs][:, :, zi, zj, zk, zl].shape != (nbl_eff, nbl_eff):
#             print(f'Binning NaMaster {block_name} covariance')
#             cov_nmt_10d_binned[probe_idxs][:, :, zi, zj, zk, zl] = \
#                 utils.bin_2d_matrix(cov=cov_nmt_10d[probe_idxs][:, :, zi, zj, zk, zl],
#                                     ells_in=ells_tot, ells_out=ells_eff,
#                                     ells_out_edges=ells_eff_edges, weights=None,
#                                     which_binning='mean')
# cov_nmt_10d = cov_nmt_10d_binned


# ! reshape the total 10d arrays to 4d
ind_use = utils.build_full_ind(triu_tril, row_col_major, zbins_use)
zpairs_auto_use, zpairs_cross_use, zpairs_3x2pt_use = utils.get_zpairs(zbins_use)
ind_auto_use = ind_use[:zpairs_auto_use, :].copy()
ind_cross_use = ind_use[zpairs_auto_use:zpairs_cross_use + zpairs_auto_use, :].copy()
elem_auto_use = zpairs_auto_use * nbl_eff
elem_autpluscross_use = (zpairs_auto_use + zpairs_cross_use) * nbl_eff

cov_nmt_4d = utils.cov_3x2pt_10D_to_4D(cov_nmt_10d, probe_ordering,
                                        nbl_eff, zbins_use, ind_use.copy(), GL_or_LG)

if covariance_ordering_2D == 'probe-ell-zpair':
    use_2DCLOE = True
    block_index = 'ell'
    cov_nmt_2d = utils.cov_4D_to_2DCLOE_3x2pt(cov_nmt_4d, zbins_use, block_index=block_index)

elif covariance_ordering_2D == 'probe-zpair-ell':
    use_2DCLOE = True
    block_index = 'ij'
    cov_nmt_2d = utils.cov_4D_to_2DCLOE_3x2pt(cov_nmt_4d, zbins_use, block_index=block_index)

elif covariance_ordering_2D == 'ell-probe-zpair':
    use_2DCLOE = False
    block_index = 'ell'
    cov_nmt_2d = utils.cov_4D_to_2D(cov_nmt_4d, block_index=block_index, optimize=True)

elif covariance_ordering_2D == 'zpair-probe-ell':
    use_2DCLOE = False
    block_index = 'ij'
    cov_nmt_2d = utils.cov_4D_to_2D(cov_nmt_4d, block_index=block_index, optimize=True)

spin_dic = {True:'spin0', False: 'spin2'}
coupled_dic = {True:'coupled', False: 'decoupled'}

fname = f"{output_folder}/cov_Gauss_3x2pt_2D_{covariance_ordering_2D}_{zbins_use}bins_{spin_dic[cfg['spin0']]}_iNKA{use_INKA}_{coupled_dic[cfg['coupled_nmt_cov']]}.npy"

np.save(fname, cov_nmt_2d)