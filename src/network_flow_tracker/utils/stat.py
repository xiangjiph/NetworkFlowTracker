from collections import defaultdict

import numpy as np
import scipy as sp
import pandas as pd
import scipy.interpolate as spi
import sklearn.mixture as sklm
import matplotlib.pyplot as plt

def compute_basic_statistics(data, bins=None):
    # bins:
    #   If scalar, should be an integer that specifies the number of bins used for the histogram calculation.
    #   If a list or numpy array, the PDF, CDF calculation will be based on the bin edge. Data outside the edge will be ignored.
    # Takes 5.7 seconds for data.shape = (150, 1280, 1024)
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    try: 
        data = data[np.isfinite(data)] # data might contain nan
    except: 
        print(data)
        raise
    data = data.flatten()
    stat = {'num_data': data.size, 'mean': np.nan, 'std': np.nan, 'cv': np.nan, 'max': np.nan, 'min': np.nan, 'median': np.nan, 
            'num_bins': np.array([]), 'hist_count': np.array([]), 'bin_val': np.array([]), 
            'bin_width': np.array([]), 'probability': np.array([]), 'pdf': np.array([]),
             'cdf': np.array([]),'prctile_th': np.array([]), 'prctile_th': np.array([]), 
             'eff_ptrl_std': np.nan, 'eff_ptrl_cv': np.nan}

    if bins is None:
        stat['num_bins'] = 10
    elif np.isscalar(bins):
        stat['num_bins'] = bins
    else:
        if not isinstance(bins, np.ndarray):
            bins = np.array(bins)
    if data.size > 0: 
        stat['mean'] = np.mean(data)
        mean_x2 = np.mean(data ** 2)
        stat['std'] = np.sqrt(mean_x2 - stat['mean'] ** 2)
        stat['cv'] = stat['std'] / np.abs(stat['mean'])

        if isinstance(bins, np.ndarray):
            stat['num_bins'] = bins.size - 1
            stat['hist_count'], stat['bin_edge'] = np.histogram(data, bins=bins)
        else:
            stat['hist_count'], stat['bin_edge'] = np.histogram(
                data, bins=stat['num_bins'])

        stat['bin_val'] = (stat['bin_edge'][:-1] + stat['bin_edge'][1:]) / 2
        stat['bin_width'] = stat['bin_edge'][1:] - stat['bin_edge'][:-1]
        stat['probability'] = stat['hist_count'] / np.sum(stat['hist_count'])
        stat['pdf'] = stat['probability'] / stat['bin_width']
        stat['cdf'] = np.cumsum(stat['probability'])

        stat['prctile_th'] = np.array(
            [0, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100])
        stat['prctile_val'] = np.percentile(data, q=stat['prctile_th'])
        stat['median'] = stat['prctile_val'][6]
        stat['max'] = stat['prctile_val'][-1]
        stat['min'] = stat['prctile_val'][0]
        stat['eff_ptrl_std'] = (stat['prctile_val'][7] - stat['prctile_val'][5]) / 1.349
        stat['eff_ptrl_cv'] = np.abs(stat['eff_ptrl_std'] / stat['median'])
        
    return stat

def compute_percentile_outlier_threshold(x, ipr = 1.5):
    # ipr = 1.5 -> 2 sigma
    #       2.2 -> 3 sigma
    p25, p75 = np.nanpercentile(x, [25, 75])
    width = ipr * (p75 - p25)
    return [p25 - width, p75 + width]

def fill_internal_nan(data, method='linear'):
    mask = np.isnan(data)
    if np.any(mask):
        itp_hdl = spi.interp1d(np.flatnonzero(~mask), data[~mask], kind=method)
        data[mask] = itp_hdl(np.flatnonzero(mask))
    return data

def fill_edge_nan_1d(data, method='nearest'):
    if method == 'nearest':
        ind = np.flatnonzero(~np.isnan(data))
        data[:ind[0]] = data[ind[0]]
        data[ind[-1]:] = data[ind[-1]]
    return data 

def estimate_background_by_fraction(data, bg_frac=1, sample_step=None, bg_mask=None):
    if bg_mask is not None: 
        data = data[bg_mask]
    data = data.flatten()
    if sample_step is not None: 
        data = data[::sample_step]
    data = np.sort(data)
    if bg_frac < 1:
        data = data[:int(bg_frac * data.size)]
    result = {}
    result['mean'] = np.mean(data)
    result['std'] = np.std(data)
    result['p25'] = data[int(0.25 * data.size)]
    result['p50'] = data[int(0.50 * data.size)]
    result['p75'] = data[int(0.75 * data.size)]
    return result

def estimate_percentile_from_hist_count(count, ptrl, bin_val=None):

    count = np.asarray(count)
    ptrl = np.asarray(ptrl)
    if bin_val is None: 
        bin_val = np.arange(count.size) + 0.5
    cdf = np.cumsum(count) / np.sum(count)
    cdf_itp = spi.interp1d(cdf, bin_val, kind='linear', fill_value='extrapolate')
    return cdf_itp(ptrl / 100)
    
def analyze_hist_count_peaks_with_gaussian_mixture(bin_val, bin_count, num_comp_list, vis_Q=False):
    data = np.repeat(bin_val, bin_count.astype(np.int64))[:, None]
    result = analyze_1d_data_with_gaussian_mixture(data, num_comp_list=num_comp_list, vis_Q=vis_Q)
    return result

def analyze_1d_data_with_gaussian_mixture(data, num_comp_list, vis_Q=False):
    data = np.asarray(data).flatten()[:, None]
    num_sample = data.size
    if num_sample < 2:
        result = {'num_sample': num_sample, 'num_component': 0, 'weight': [], 
                  'mean': [], 'variance': [], 'std': [], 'cv': [], 'peak': [], 'peak_idx': [], 
                'peak_mean': [], 'peak_std': [], 'bic': [], 'num_comp_list': num_comp_list}
        return result

    lowest_bic = np.inf
    best_gm = None
    bics = []
    for n_comp in num_comp_list:
        if data.size >= n_comp:
            
            gm = sklm.GaussianMixture(n_components=n_comp, covariance_type='full').fit(data)

            bic = gm.bic(data)
            bics.append(bic)
            if bic < lowest_bic:
                best_gm = gm
                lowest_bic = bic
    
    means = best_gm.means_.flatten()
    covariances = best_gm.covariances_.flatten()  # For 1D, covariances are just variances
    stds = np.sqrt(covariances)
    weights = best_gm.weights_
    with np.errstate(divide='ignore'):
        cvs = stds / means
        peak_prob = weights / (np.sqrt(2 * np.pi * covariances))

    result = {'num_sample': num_sample, 'num_component': best_gm.n_components, 'weight': weights,
               'mean': means, 'variance': covariances, 'std': stds, 'cv': cvs, 'peak': peak_prob, 
              'bic': bics, 'num_comp_list': num_comp_list}
    result['peak_idx'] = np.argmax(result['peak']) 
    result['peak_mean'] = result['mean'][result['peak_idx']]
    result['peak_std'] = result['std'][result['peak_idx']]
    

    if vis_Q:
        print(f"Optimal number of Gaussians: {best_gm.n_components}")
        print("Component means:", means)
        print("Component stds:", stds)
        print("Component weights:", best_gm.weights_)
        print("Component peaks:", peak_prob)

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(data, bins=50, alpha=0.5, density=True, label='Data')

        x_values = np.linspace(np.min(data), np.max(data), 500)
        pdf = np.zeros_like(x_values)
        for peak, mean, var in zip(peak_prob, means, covariances):
            component_pdf = peak * np.exp(-(x_values-mean)**2/(2*var))
            pdf += component_pdf
            ax1.plot(x_values, component_pdf, '--', label='Component')

        ax1.plot(x_values, pdf, 'k-', lw=2, label='GMM fit')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Density')
        ax1.set_title('GMM Fit and Components')
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(num_comp_list, bics, marker='o')
        ax2.set_xlabel('Number of components')
        ax2.set_ylabel('BIC Score')
        ax2.set_title('BIC vs. Number of Components')
        ax2.grid(True)

    return result


def relative_difference(x, y):
    x = np.asarray(x).astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    if np.all(x >= 0) and np.all(y >= 0):
        return np.abs(x - y) / (x + y)
    else: 
        raise NotImplementedError
    
def moving_outlier_detection_by_z_score_1d(trace, wd_sz):
    trace = np.asarray(trace)
    pad_width = int(wd_sz) // 2
    trace = np.pad(trace, pad_width=pad_width, mode='reflect')
    ker = np.ones(wd_sz) / wd_sz
    mov_m = sp.signal.convolve(trace, ker, mode='valid')
    mov_m2 = sp.signal.convolve(trace ** 2, ker, mode='valid')
    std = np.sqrt(mov_m2 - mov_m ** 2) # prevent overflow? 
    z_score = mov_m / std
    return z_score

# def max_bidir_rel_diff(t_trace):
#     t_trace = np.asarray(t_trace).astype(np.float32)
#     t_diff = np.diff(t_trace)
#     forward_dn = np.concatenate(([0], np.abs(t_diff / t_trace[:-1]))) 
#     backward_dn = np.concatenate((np.abs(t_diff / t_trace[1:]), [0]))
#     return np.maximum(forward_dn, backward_dn)


def rolling_outlier_detection(data, window_size, quantile_range=(0.25, 0.75), k=3, min_lower=None, max_high=None):
    """
    Detect outliers in a time series using a rolling window approach with quantile-based thresholds.

    Parameters
    ----------
    data : array-like
        1D time series data.
    window_size : int
        Size of the rolling window.
    quantile_range : tuple of float
        The lower and upper quantiles to compute (e.g., (0.25, 0.75) for Q1 and Q3).
    k : float
        Multiplier for the IQR to define the outlier range.
    
    Returns
    -------
    outliers : pd.Series (dtype=bool)
        Boolean Series indicating True for outliers and False for inliers.
    """
    series = pd.Series(data)
    
    # Compute rolling quantiles
    q_low = series.rolling(window=window_size, center=True).quantile(quantile_range[0])
    q_high = series.rolling(window=window_size, center=True).quantile(quantile_range[1])
    
    # Compute rolling IQR and thresholds
    iqr = q_high - q_low
    lower_bound = q_low - k * iqr
    upper_bound = q_high + k * iqr
    if min_lower is not None: 
        lower_bound = np.maximum(min_lower, lower_bound)
    if max_high is not None:
        upper_bound = np.minimum(max_high, upper_bound)
    # Identify outliers
    outliers = (series < lower_bound) | (series > upper_bound)
    
    return outliers.values

def compute_rolling_inliner_range(data, wd_sz, k=3):

    data = pd.Series(data)
    data = pd.concat((data[:wd_sz-1][::-1], data, data[-(wd_sz-1):][::-1]))
    q_low = data.rolling(window=wd_sz, center=True).quantile(0.25)[wd_sz-1 : -wd_sz + 1]
    q_high = data.rolling(window=wd_sz, center=True).quantile(0.75)[wd_sz-1 : -wd_sz + 1]
    iqr = q_high - q_low
    upper_bound = q_high + k * iqr
    lower_bound = q_low - k * iqr

    return lower_bound.values, upper_bound.values

def analyze_matrix_diagonals(mat):

    assert mat.shape[0] == mat.shape[1], 'mat is not a square matrix'
    l = mat.shape[0]
    diag_offset = np.arange(-l + 1, l)
    stat = defaultdict(lambda : np.full(diag_offset.shape, np.nan))
    stat['offset'] = diag_offset
    for i, j in enumerate(diag_offset): 
        tmp_terms = np.diagonal(mat, offset=j)
        stat['num'][i] = tmp_terms.size
        stat['sum'][i] = np.sum(tmp_terms)
        stat['std'][i] = np.std(tmp_terms)
    
    stat['mean'] = stat['sum'] / stat['num']

    return stat

def analyze_twp_traces_corr(x, y):
    valid_Q = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[valid_Q]
    y = y[valid_Q]
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    x -= x_avg
    y -= y_avg
    xy = np.sum(x * y)
    result = {}
    result['corr'], result['p'] = sp.stats.pearsonr(x, y)
    result['k_y2x'] = xy / np.sum(x ** 2) # y = k_x * x
    result['k_x2y'] = xy / np.sum(y ** 2) # x = k_y * y
    result['k_y2x_n'] = result['k_y2x'] * x_avg / y_avg 
    result['k_x2y_n'] = result['k_x2y'] * y_avg / x_avg
    return result

def analyze_traces_corr_mat(X, skip_vec=None): 
    """
        X: (num_traces, T)
    
    """
    num_traces, num_T = X.shape
    if skip_vec is None: 
        skip_vec = np.ones((num_traces, ), bool)

    tree_corr_mat = np.full((num_traces, num_traces), np.nan)
    tree_corr_p_mat = np.full((num_traces, num_traces), 1.0)
    tree_slope_mat = np.full((num_traces, num_traces), np.nan)
    tree_slope_n_mat = np.full((num_traces, num_traces), np.nan)
    for i in range(num_traces):
        tmp_trace_1 = X[i]
        if skip_vec[i]:
            continue
        for j in range(i+1, num_traces): 
            tmp_trace_2 = X[j]
            if skip_vec[j]: 
                continue
            tmp_valid_Q = np.logical_and(np.isfinite(tmp_trace_1), np.isfinite(tmp_trace_2))
            if (np.count_nonzero(tmp_valid_Q) > 10):  
                tmp_corr = analyze_twp_traces_corr(tmp_trace_1, tmp_trace_2)
                tree_corr_mat[i, j] = tmp_corr['corr']
                tree_corr_mat[j, i] = tmp_corr['corr']
                tree_corr_p_mat[i, j] = tmp_corr['p']
                tree_corr_p_mat[j, i] = tmp_corr['p']
                # f_j = k_y2x * f_i, i.e. the linear response of edge j in response to the change in edge i
                tree_slope_mat[i, j] = tmp_corr['k_y2x'] 
                tree_slope_mat[j, i] = tmp_corr['k_x2y']
                tree_slope_n_mat[i, j] = tmp_corr['k_y2x_n']
                tree_slope_n_mat[j, i] = tmp_corr['k_x2y_n']
    
    return {'corr': tree_corr_mat, 'p': tree_corr_p_mat, 'k': tree_slope_mat, 'k_n': tree_slope_n_mat}

def compute_ratio_uncertainty(x, y, x_std, y_std): 
    s = x + y
    f_x = x / s
    f_y = y / s
    f_x_sigma = np.sqrt( (y * x_std/ s ** 2) ** 2 + (x * y_std / s ** 2) ** 2 )
    return f_x, f_y, f_x_sigma, f_x_sigma