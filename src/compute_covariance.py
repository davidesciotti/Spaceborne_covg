import gc
import json
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import yaml
import scipy

import utils

# ! settings
# import the yaml config file
# cfg = yaml.load(sys.stdin, Loader=yaml.FullLoader)

# if you want to execute without passing the path
with open('../config/example_config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

survey_area = cfg['survey_area']  # deg^2
deg2_in_sphere = utils.DEG2_IN_SPHERE
fsky = survey_area / deg2_in_sphere

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
assert isinstance(cfg['n_gal_shear'], (int, float, list, str)), 'n_gal_shear must be an integer, float, list or string'
assert isinstance(cfg['n_gal_clustering'], (int, float, list, str)
                  ), 'n_gal_clust must be an integer, float, list or string'

# get and check the number of galaxies in each redshift bin
n_gal_shear = utils.get_ngal(cfg['n_gal_shear'], EP_or_ED, zbins, cfg['EP_check_tol'])
n_gal_clustering = utils.get_ngal(cfg['n_gal_clustering'], EP_or_ED, zbins, cfg['EP_check_tol'])

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
    fmt = '%12.4f'
    header_lines = [
        f'ell_min = {ell_min}',
        f'ell_max = {ell_max}',
        f'ell_bins = {nbl}',
        'ell_bin_lower_edge    ell_bin_upper_edge    ell_bin_center    delta_ell'
    ]
    ell_grid_header = '\n'.join(header_lines)
    ell_grid = np.column_stack((ell_bin_lower_edges, ell_bin_upper_edges, ell_values, delta_values))
    np.savetxt(f'{output_folder}/ell_grid.txt', ell_grid,  fmt=fmt, header=ell_grid_header, comments='# ')
    
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
cl_LL_2D = np.genfromtxt(
    '/home/davide/Documenti/Lavoro/Programmi/Spaceborne_covg/input/data_DR1/Cls_zNLA_ShearShear_DR1_CLOE.dat')
cl_GL_2D = np.genfromtxt(
    '/home/davide/Documenti/Lavoro/Programmi/Spaceborne_covg/input/data_DR1/Cls_zNLA_PosShear_DR1_CLOE.dat')
cl_GG_2D = np.genfromtxt(
    '/home/davide/Documenti/Lavoro/Programmi/Spaceborne_covg/input/data_DR1/Cls_zNLA_PosPos_DR1_CLOE.dat')
assert np.allclose(cl_LL_2D[:, 0], ell_values, atol=0, rtol=1e-5)
assert np.allclose(cl_GL_2D[:, 0], ell_values, atol=0, rtol=1e-5)
assert np.allclose(cl_GG_2D[:, 0], ell_values, atol=0, rtol=1e-5)
cl_LL_2D = cl_LL_2D[:, 1:]
cl_GL_2D = cl_GL_2D[:, 1:]
cl_GG_2D = cl_GG_2D[:, 1:]

cl_LL_3D = utils.cl_2D_to_3D_symmetric(cl_LL_2D, nbl, zpairs_auto, zbins)
cl_GL_3D = utils.cl_2D_to_3D_asymmetric(cl_GL_2D, nbl, zbins, order='C')
cl_GG_3D = utils.cl_2D_to_3D_symmetric(cl_GG_2D, nbl, zpairs_auto, zbins)
# cl_LL_3D = np.load(f'{cfg["cl_LL_3D_path"]}')
# cl_GL_3D = np.load(f'{cfg["cl_GL_3D_path"]}')
# cl_GG_3D = np.load(f'{cfg["cl_GG_3D_path"]}')


cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
cl_3x2pt_5D[0, 0, :, :, :] = cl_LL_3D
cl_3x2pt_5D[1, 1, :, :, :] = cl_GG_3D
cl_3x2pt_5D[1, 0, :, :, :] = cl_GL_3D
cl_3x2pt_5D[0, 1, :, :, :] = np.transpose(cl_GL_3D, (0, 2, 1))

# ! Compute covariance
# create a noise with dummy axis for ell, to have the same shape as cl_3x2pt_5D
noise_3x2pt_4D = utils.build_noise(zbins, n_probes, sigma_eps2=sigma_eps2,
                                   ng_shear=n_gal_shear,
                                   ng_clust=n_gal_clustering,
                                   EP_or_ED=EP_or_ED,
                                   which_shape_noise=cfg['which_shape_noise'])
noise_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
for probe_A in (0, 1):
    for probe_B in (0, 1):
        for ell_idx in range(nbl):
            noise_3x2pt_5D[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

# compute 3x2pt cov
start = time.perf_counter()
cov_3x2pt_10D_arr = utils.covariance_einsum(cl_3x2pt_5D, noise_3x2pt_5D, fsky, ell_values, delta_values)
print(f'covariance computation took {time.perf_counter() - start:.2f} seconds')

# reshape to 4D
cov_3x2pt_10D_dict = utils.cov_10D_array_to_dict(cov_3x2pt_10D_arr)
cov_3x2pt_4D = utils.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_10D_dict, probe_ordering, nbl, zbins, ind.copy(),
                                              GL_or_LG)
del cov_3x2pt_10D_dict, cov_3x2pt_10D_arr
gc.collect()

# reshape to 2D
# if not cfg['use_2DCLOE']:
#     cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)
# elif cfg['use_2DCLOE']:
#     cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index='ell')
# else:
#     raise ValueError('use_2DCLOE must be a true or false')

if covariance_ordering_2D == 'probe_ell_zpair':
    use_2DCLOE = True
    block_index = 'ell'
    cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index=block_index)

elif covariance_ordering_2D == 'probe_zpair_ell':
    use_2DCLOE = True
    block_index = 'ij'
    cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index=block_index)

elif covariance_ordering_2D == 'ell_probe_zpair':
    use_2DCLOE = False
    block_index = 'ell'
    cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)

elif covariance_ordering_2D == 'zpair_probe_ell':
    use_2DCLOE = False
    block_index = 'ij'
    cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)

else:
    raise ValueError('covariance_ordering_2D must be a one of the following: probe_ell_zpair, probe_zpair_ell,'
                     'ell_probe_zpair, zpair_probe_ell')

if cfg['plot_covariance_2D']:
    plt.matshow(np.log10(cov_3x2pt_2D))
    plt.colorbar()
    plt.title(f'log10(cov_3x2pt_2D)\nordering: {covariance_ordering_2D}')

other_quantities_tosave = {
    'n_gal_shear [arcmin^{-2}]': list(n_gal_shear),
    'n_gal_clustering [arcmin^{-2}]': list(n_gal_clustering),
    'survey_area [deg^2]': survey_area,
    'sigma_eps': sigma_eps,
}

np.save(f'{output_folder}/cov_Gauss_3x2pt_2D_{covariance_ordering_2D}_{survey_area:d}deg2.npy', cov_3x2pt_2D)

with open(f'{output_folder}/other_specs.txt', 'w') as file:
    file.write(json.dumps(other_quantities_tosave, indent=4))

print(f'Done')
print(f'Covariance files saved in {output_folder}')


cov_bench = np.load(
    '/home/davide/Documenti/Lavoro/Programmi/Spaceborne_covg/input/data_DR1/cov_Gauss_3x2pt_CLOEDR1_2500sqdeg.npy')
diff = (cov_bench / cov_3x2pt_2D - 1) * 100

plt.figure()
plt.plot(np.diag(diff))

plt.figure()
plt.matshow(np.log10(np.abs(diff)))
plt.colorbar()
