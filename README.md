# ⚠️ This repository is no longer maintained. Please refer to the [Spaceborne](https://github.com/davidesciotti/Spaceborne) repository instead ⚠️

# Spaceborne - Gaussian covariance

 A simple script to compute the Gaussian 3x2pt covariance in harmonic space, using the following formula (Eq.(138) of the _Euclid_ Preparation: VII. Forecast Validation [paper](https://arxiv.org/abs/1910.09273)).

$$
{\rm Cov}\left[C_{i j}^{A B}(\ell), C_{k l}^{A^{\prime} B^{\prime}}\left(\ell^{\prime}\right)\right]=\frac{\left[C_{i k}^{A A^{\prime}}(\ell)+N_{i k}^{A A^{\prime}}(\ell)\right]\left[C_{j l}^{B B^{\prime}}\left(\ell^{\prime}\right)+N_{j l}^{B B^{\prime}}\left(\ell^{\prime}\right)\right]+\left[C_{i l}^{A B^{\prime}}(\ell)+N_{i l}^{A B^{\prime}}(\ell)\right]\left[C_{j k}^{B A^{\prime}}\left(\ell^{\prime}\right)+N_{j k}^{B A^{\prime}}\left(\ell^{\prime}\right)\right]}{(2 \ell+1) f_{\mathrm{sky}} \Delta \ell} \delta_{\ell \ell^{\prime}}^{\mathrm{K}} \text {.}
$$

where $A$, $B$, $A'$ and $B'$ run over the photometric observables. See the `config/example_config.yaml` file for a detailed description of the inputs required. You can run the script with 
 
     python compute_covariance.py < ../config/example_config.yaml
 
