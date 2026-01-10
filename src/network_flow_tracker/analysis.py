from collections import defaultdict
import warnings

import pandas as pd 
import numpy as np
import scipy as sp 
from scipy.stats import binomtest, beta
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from sklearn.linear_model import HuberRegressor

from .utils import neighbors as nb
from .utils import stat
from .utils import util

from network_flow_tracker import linking as NFTLinking


#region Edge features
def cart2sph(x, y, z):
    """
    Convert Cartesian to spherical coordinates.
    Returns azimuth, elevation, radius in radians.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    # Prevent division by zero for degenerate vectors
    r[r == 0] = 1e-16 
    az = np.arctan2(y, x)       # range: [-pi, pi]
    el = np.arcsin(z / r)       # range: [-pi/2, pi/2]
    return az, el, r

def pca_first_component(X):
    """
    Return the first principal component (largest singular vector)
    of the centered point cloud X (N x 3).
    """
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Rows of Vt are the principal directions; the first row is the largest principal component.
    return Vt[0]  # shape: (3,)

def pca_full(X):
    """
    Return all principal components and corresponding eigenvalues
    of the covariance matrix of X (N x 3).
    """
    X_centered = X - np.mean(X, axis=0)
    # Compute covariance
    cov_mat = X_centered.T @ X_centered / (X.shape[0] - 1)
    # SVD of covariance => eigen decomposition
    U, S, _ = np.linalg.svd(cov_mat)
    # U columns are eigenvectors; S are eigenvalues
    return U, S  # U: (3,3), S: (3,)

def sub_to_length(sub_points):
    """
    Sum of Euclidean distances between consecutive points in sub_points (N x 3).
    """
    if len(sub_points) < 2:
        return 0.0
    diffs = np.diff(sub_points, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(seg_lengths)

def fun_analysis_get_link_features(vessel_graph, link_feature=None):
    """
    Translated Python version of the MATLAB function 
    'fun_analysis_get_link_features' using NumPy (and optionally pandas).
    """
    if link_feature is None:
        link_feature = ['geometry', 'dt']
    
    compute_geometryQ = ('geometry' in link_feature)
    compute_dtQ = ('dt' in link_feature)
    
    # Collect needed fields from vessel_graph
    link_cc = vessel_graph['link']['cc_ind']  # list of lists of voxel indices
    image_size = vessel_graph['num']['mask_size']  # (dim_x, dim_y, dim_z)
    
    # 'radius' is where the DT values are stored
    vessel_mask_dt = vessel_graph.get('radius', None)
    
    num_l = len(link_cc)
    
    # Initialize output dictionary 'lf'
    lf = {}
    
    # Geometry placeholders
    if compute_geometryQ:
        lf['length'] = np.full(num_l, np.nan)
        lf['link_com'] = np.full((num_l, 3), np.nan)
        lf['ep1_sub'] = np.full((num_l, 3), np.nan)
        lf['ep2_sub'] = np.full((num_l, 3), np.nan)
        lf['ep2ep_dist'] = np.full(num_l, np.nan)
        lf['ep1_to_ep2_direction_vec'] = np.full((num_l, 3), np.nan)
        lf['ep1_direction_vec'] = np.full((num_l, 3), np.nan)
        lf['ep2_direction_vec'] = np.full((num_l, 3), np.nan)
        lf['cc_sub_pca1_vec'] = np.full((num_l, 3), np.nan)
        lf['cc_sub_pca2_vec'] = np.full((num_l, 3), np.nan)
        lf['cc_sub_pca3_vec'] = np.full((num_l, 3), np.nan)
        lf['cc_sub_cov_eig_val'] = np.full((num_l, 3), np.nan)
    
    # DT placeholders
    if compute_dtQ and (vessel_mask_dt is not None):
        lf['dt_ep1'] = np.full(num_l, np.nan)
        lf['dt_ep2'] = np.full(num_l, np.nan)
        lf['dt_max'] = np.full(num_l, np.nan)
        lf['dt_min'] = np.full(num_l, np.nan)
        lf['dt_mean'] = np.full(num_l, np.nan)
        lf['dt_median'] = np.full(num_l, np.nan)
        lf['dt_std'] = np.full(num_l, np.nan)
        lf['dt_diff_ep2ep'] = np.full(num_l, np.nan)
    
    # Iterate over each link
    for i_link in range(num_l):
        tmp_ind = link_cc[i_link]  # voxel indices for this link
        # Convert linear indices to 3D subscripts
        tmp_sub = np.column_stack(np.unravel_index(tmp_ind, image_size))  # shape: (N, 3)
        
        if compute_geometryQ and tmp_sub.size > 0:
            tmp_num_voxel = len(tmp_sub)
            # Center of mass (mean of coordinates)
            tmp_sub_mean = np.mean(tmp_sub, axis=0)
            lf['link_com'][i_link, :] = tmp_sub_mean
            
            # Compute length along the link
            lf['length'][i_link] = sub_to_length(tmp_sub)
            
            if tmp_num_voxel >= 1:
                lf['ep1_sub'][i_link, :] = tmp_sub[0, :]
                lf['ep2_sub'][i_link, :] = tmp_sub[-1, :]
            
            if tmp_num_voxel > 1:
                # End-to-end distance
                ep1_to_ep2_vec = tmp_sub[-1, :] - tmp_sub[0, :]
                ep_dist = np.linalg.norm(ep1_to_ep2_vec)
                lf['ep2ep_dist'][i_link] = ep_dist
                if ep_dist > 0:
                    lf['ep1_to_ep2_direction_vec'][i_link, :] = ep1_to_ep2_vec / ep_dist
            
            # If at least 3 points, do PCA-based direction calculations
            if tmp_num_voxel > 2:
                pca_max_num_voxel = 10
                # Region near ep1
                tmp_1_sub = tmp_sub[:min(pca_max_num_voxel, tmp_num_voxel), :]
                ep_1_vec = pca_first_component(tmp_1_sub)
                
                # Check alignment with ep1->ep2 direction
                if ep_dist > 0 and np.dot(ep1_to_ep2_vec, ep_1_vec) > 0:
                    lf['ep1_direction_vec'][i_link, :] = -ep_1_vec
                else:
                    lf['ep1_direction_vec'][i_link, :] = ep_1_vec
                
                # Region near ep2
                tmp_2_sub = tmp_sub[max(0, tmp_num_voxel - min(pca_max_num_voxel, tmp_num_voxel)):, :]
                ep_2_vec = pca_first_component(tmp_2_sub)
                
                if ep_dist > 0 and np.dot(ep1_to_ep2_vec, ep_2_vec) < 0:
                    lf['ep2_direction_vec'][i_link, :] = -ep_2_vec
                else:
                    lf['ep2_direction_vec'][i_link, :] = ep_2_vec
                
                # PCA of entire link
                U, S = pca_full(tmp_sub)
                # Columns of U are principal directions
                lf['cc_sub_pca1_vec'][i_link, :] = U[:, 0]
                lf['cc_sub_pca2_vec'][i_link, :] = U[:, 1]
                lf['cc_sub_pca3_vec'][i_link, :] = U[:, 2]
                lf['cc_sub_cov_eig_val'][i_link, :] = S  # 3 eigenvalues
        
        # Distance transform features
        if compute_dtQ and (vessel_mask_dt is not None) and len(tmp_ind) > 0:
            tmp_dt = vessel_mask_dt[tmp_ind]  # gather DT values
            # Direct endpoints
            lf['dt_ep1'][i_link] = tmp_dt[0]
            lf['dt_ep2'][i_link] = tmp_dt[-1]
            lf['dt_diff_ep2ep'][i_link] = abs(tmp_dt[-1] - tmp_dt[0])
            
            # Filter out zero or invalid DT if needed
            tmp_dt_valid = tmp_dt[tmp_dt > 0]
            if tmp_dt_valid.size > 0:
                lf['dt_max'][i_link] = np.max(tmp_dt_valid)
                lf['dt_min'][i_link] = np.min(tmp_dt_valid)
                lf['dt_mean'][i_link] = np.mean(tmp_dt_valid)
                lf['dt_median'][i_link] = np.median(tmp_dt_valid)
                lf['dt_std'][i_link] = np.std(tmp_dt_valid)
    
    # Post-processing geometry
    if compute_geometryQ:
        # straightness
        lf['straightness'] = lf['ep2ep_dist'] / lf['length']
        # Convert end-to-end direction vector to spherical angles
        tmp_vec = lf['ep1_to_ep2_direction_vec'].copy()
        # Flip negative z to ensure z >= 0
        negative_z = tmp_vec[:, 2] < 0
        tmp_vec[negative_z, :] = -tmp_vec[negative_z, :]
        az, el, _ = cart2sph(tmp_vec[:, 0], tmp_vec[:, 1], tmp_vec[:, 2])
        
        # Store angles in degrees
        lf['ep2ep_angle_azimuth_deg'] = np.degrees(az)
        lf['ep2ep_angle_elevation_deg'] = np.degrees(el)
    
    # Post-processing dt
    if compute_dtQ and (vessel_mask_dt is not None):
        dt_std_n = np.divide(lf['dt_std'], lf['dt_mean'], out=np.full(num_l, np.nan), where=lf['dt_mean']!=0)
        lf['dt_std_n'] = dt_std_n
        
        ep2ep_dist = lf['ep2ep_dist'] if compute_geometryQ else np.ones(num_l)
        lf['dt_e2e_2_ep_dist'] = lf['dt_diff_ep2ep'] / ep2ep_dist
        
        lf['dt_ep1_plus_ep2'] = lf['dt_ep1'] + lf['dt_ep2']
        
        if compute_geometryQ:
            length_ = lf['length']
            # Avoid division by zero
            nonzero_len = length_ != 0
            dt_diff = lf['dt_diff_ep2ep']
            lf['dt_e2e_2_length'] = np.where(nonzero_len, dt_diff / length_, np.nan)
            lf['dt_max_2_length'] = np.where(nonzero_len, lf['dt_max'] / length_, np.nan)
            lf['dt_mmxx_2_length'] = np.where(nonzero_len, (lf['dt_max'] - lf['dt_min']) / length_, np.nan)
            lf['dt_ep_sum_2_length'] = np.where(nonzero_len, lf['dt_ep1_plus_ep2'] / length_, np.nan)
            
            # Example: approximate surface area / volume if you interpret dt as radius
            lf['surface_area'] = 2 * np.pi * length_ * lf['dt_median']
            lf['volume'] = np.pi * length_ * (lf['dt_median']**2)
    
    # Convert dictionary of arrays to a pandas DataFrame (optional)
    lf_df = pd.DataFrame(lf)
    
    return lf_df
#endregion

#region Edge traces
def compute_edges_spatiotemporal_traces(fg, trace_result, el_to_idx, mv_avg_wd, mv_min_num=3, 
                                        voxel_size_um=1, frame_rate_Hz=1, cell_labeled_fraction=1):

    std_para = {'vxl_size_um': voxel_size_um, 
            'frame_rate_Hz': frame_rate_Hz, 
            'labeled_fraction': cell_labeled_fraction}
    
    edges_traces = {}
    for test_el in range(fg.edge.num_cc):
        if (test_el in el_to_idx): 
            tmp_edge_table = trace_result.iloc[el_to_idx[test_el]]
            test_ef = fg.get_edgeflow_object(test_el, detection_in_edge=tmp_edge_table)
        else: 
            test_ef = None

        if test_ef is not None: 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                tmp_stat = test_ef.analyze_edge_spatiotemporal_dynamics(mv_avg_wd, mv_min_num, **std_para)
                edges_traces[test_el] = tmp_stat
        print(f"\rFinish computing dynamic traces in edge {test_el}. ", end="", flush=True)
    return edges_traces

#region Phase separation

def collect_downstream_first_appearance_frame_for_cells_in_an_edge(edge_label:int, trace_result:pd.DataFrame, 
                                                                   pid_to_idx:dict, el_to_idx:dict):
    detect_in_edge = trace_result.iloc[el_to_idx[edge_label]]
    p_id_in_edge = np.unique(detect_in_edge.particle.values)
    ds_edge_entr_frame = defaultdict(list) # first detection frame in the downstream edge
    for p_id in p_id_in_edge: 
        tmp_p_idx = pid_to_idx[p_id]
        tmp_p_el = trace_result.edge_label.values[tmp_p_idx]
        tmp_p_el_u, tmp_p_el_idx = np.unique(tmp_p_el, return_index=True)
        tmp_p_idx = tmp_p_idx[tmp_p_el_idx]
        tmp_p_etr_frame = trace_result.frame.values[tmp_p_idx]
        for e, f in zip(tmp_p_el_u, tmp_p_etr_frame): 
            if e != edge_label: 
                ds_edge_entr_frame[e].append(f)
    
    return ds_edge_entr_frame

def collect_downstream_first_exit_edge_appearance_frame_for_cells_in_an_edge(edge_label:int, ds_edge_label, trace_result:pd.DataFrame, 
                                                                   pid_to_idx:dict, el_to_idx:dict):
    detect_idx = el_to_idx[edge_label] if edge_label in el_to_idx.keys() else []
    detect_in_edge = trace_result.iloc[detect_idx]
    p_id_in_edge = np.unique(detect_in_edge.particle.values)
    ds_edge_entr_frame = defaultdict(list) # first detection frame in the downstream edge
    for p_id in p_id_in_edge: 
        tmp_p_idx = pid_to_idx[p_id]
        tmp_p_el = trace_result.edge_label.values[tmp_p_idx]
        # find the last frame stay in the edge: 
        tmp_lf_idx = np.nonzero(tmp_p_el == edge_label)[0]
        tmp_lf_idx = tmp_lf_idx[-1]
        tmp_dse, tmp_first_dse_ind, _ = np.intersect1d(tmp_p_el, ds_edge_label, return_indices=True)
        if np.any(tmp_first_dse_ind < tmp_lf_idx): 
            # print(f"The particle {p_id} reached the downstream edge before entering the parent edge {edge_label}. Skip this trajectory.")
            continue

        if tmp_dse.size == 0: 
            tmp_first_exit_frame_idx = tmp_lf_idx + 1
            # particle is too fast - might pass through the entire edge
            while tmp_first_exit_frame_idx <= (tmp_p_el.size - 1): 
                if tmp_p_el[tmp_first_exit_frame_idx] >= 0: 
                    tmp_first_exit_frame = trace_result.frame.values[tmp_p_idx[tmp_first_exit_frame_idx]]
                    tmp_dse = tmp_p_el[tmp_first_exit_frame_idx]         
                    ds_edge_entr_frame[int(tmp_dse)].append(tmp_first_exit_frame)           
                    break
                else: 
                    tmp_first_exit_frame_idx += 1
        elif tmp_dse.size == 1: 
            assert tmp_first_dse_ind > tmp_lf_idx, ValueError(f"The particle {p_id} reached the downstream edge before entering the parent edge ???")
            tmp_first_exit_frame = trace_result.frame.values[tmp_p_idx[tmp_first_dse_ind]]
            ds_edge_entr_frame[int(tmp_dse)].append(tmp_first_exit_frame)
        else: 
            # this could happen when the cell somehow get assigned to one edge near the entrance 
            # but actually get into the other edge ...
            tmp_idx = np.argmax(tmp_first_dse_ind)
            tmp_dse = tmp_dse[tmp_idx]
            tmp_first_dse_ind = tmp_first_dse_ind[tmp_idx]
            tmp_first_exit_frame = trace_result.frame.values[tmp_p_idx[tmp_first_dse_ind]]
            ds_edge_entr_frame[int(tmp_dse)].append(tmp_first_exit_frame)
            # raise ValueError(f"The particle {p_id} reached both downstream edges ???")

    return ds_edge_entr_frame

def compute_edge_downstream_cell_partition_trace(fg, ptrace_result, pid_to_idx, el_to_idx, edge_label, ds_edge_label=None,
                                             mv_wd_sz=None, mv_wd_min_n=3, num_t_pts=None):
    edge_label = int(edge_label)    
    if num_t_pts is None: 
        num_t_pts = ptrace_result.frame.max() + 1
    
    ds_e_count = {}
    tmp_parent_count = np.zeros(num_t_pts)
    # If downstream edges are not given, use the 1-st order downstream edges
    if ds_edge_label is None: 
        ds_edge_label = fg.get_nearest_downstream_edge_labels(edge_label)
    # classify downstream edges
    dsds_edge_label = {}
    tmp_ds_e_etr_f = collect_downstream_first_exit_edge_appearance_frame_for_cells_in_an_edge(\
                    edge_label, ds_edge_label, ptrace_result, pid_to_idx, el_to_idx)
    for dse in ds_edge_label: 
        dsds_e = []
        tmp_ds_e = fg.get_downstream_edges(dse, cutoff_order=np.inf, include_self_Q=True) # for backward compactiblity - to be test and remove
        for k in tmp_ds_e_etr_f.keys():
            if k in tmp_ds_e: 
                dsds_e.append(k)
        
        dsds_edge_label[dse] = np.asarray(np.unique(dsds_e))
    dsds_el = np.concatenate(list(dsds_edge_label.values()))
    # bi-loop
    # assert dsds_el.size == np.unique(dsds_el).size, ValueError("Nearest downstream edges share the same next downstream edge")

    # If a cell cross the entire edge and get into the downstream of an edge, 
    # add it to the corrresponding nearesting downstream
    for dse in ds_edge_label: 
        tmp_count = np.zeros(num_t_pts)
        for el in dsds_edge_label[dse]:
            for i in tmp_ds_e_etr_f[el]:
                tmp_count[i] += 1
                tmp_parent_count[i] += 1        
        # should not set the 0 to nan here. 0 means not observed here. 
        ds_e_count[dse] = tmp_count

    result = __analyze_1_in_2_out_trace_stat(tmp_parent_count, ds_e_count, mv_wd_sz, mv_wd_min_n)
    return result

def compute_edge_downstream_cell_partition_trace_by_passenger(fg, node_label, mv_avg_wd, num_t_pts, mv_wd_min_n=3): 
    n_fw_info = analyze_node_cell_forward_stat(node_label, fg, fg.edge_v_pxl, return_time_Q=True)
    parent_trace, child_traces = __fw_etr_t_dict_to_trace(n_fw_info['fw_etr_t'], num_t_pts)
    ps_trace_stat = __analyze_1_in_2_out_trace_stat(parent_trace, child_traces, mv_avg_wd, mv_wd_min_n)
    return n_fw_info, ps_trace_stat


def analyze_edge_downstream_phase_seperation(ds_e_count, edges_traces, p_el, c_el, x_name, 
                                             fill_nan_method=None, both_valid_stat_Q=True, vis_Q=False, 
                                             vis_csv_flx_Q=False, flux_corr_method=None):
    x_d = {}
    x_sum = None
    for e in c_el: 
        tmp_x = np.abs(edges_traces[e][x_name])
        if fill_nan_method is not None: 
            if fill_nan_method == 'mean': 
                tmp_x[np.isnan(tmp_x)] = np.nanmean(tmp_x)
            elif fill_nan_method == 'median': 
                tmp_x[np.isnan(tmp_x)] = np.nanmedian(tmp_x)
        x_d[e] = tmp_x

        if x_sum is None: 
            x_sum = tmp_x.copy()
        else: 
            x_sum += tmp_x  

    if flux_corr_method == 'Pries': 
        if len(c_el) == 2:
            x_sum, x_d[c_el[0]], x_d[c_el[1]] = correct_bifurcation_flow_conservation_Pries(\
                np.abs(edges_traces[p_el][x_name]), x_d[c_el[0]], x_d[c_el[1]])
        else: 
            raise NotImplementedError
    elif flux_corr_method == 'LSE': 
        tmp_Qp = np.abs(edges_traces[p_el][x_name].copy())
        tmp_Qcs = np.vstack([np.abs(x_d[e].copy()) for e in c_el]) 
        x_sum, tmp_Qcs_c = correct_node_flow_conservation_LS(tmp_Qp, tmp_Qcs)
        for i, e in enumerate(c_el): 
            x_d[e] = tmp_Qcs_c[i]

    x_d = {k: v / x_sum for k, v in x_d.items()}
    y_d = {e : ds_e_count[e]['p'] for e in c_el}

    x_stat_d = {}
    y_stat_d = {}
    for e in c_el: 
        tmp_x = x_d[e]
        tmp_y = y_d[e]
        # only take the datapoints with both dimensions being finite? 
        if both_valid_stat_Q:
            tmp_valid_Q = np.logical_and(np.isfinite(tmp_x), np.isfinite(tmp_y))
            tmp_x = tmp_x[tmp_valid_Q]
            tmp_y = tmp_y[tmp_valid_Q]

        x_stat_d[e] = stat.compute_basic_statistics(tmp_x)
        y_stat_d[e] = stat.compute_basic_statistics(tmp_y)

    if vis_Q: 
        f = plt.figure(figsize=(10, 4))
        a1 = f.add_subplot(1, 2, 1)
        for e in c_el: 
            sc = a1.scatter(x_d[e], y_d[e], label=f"{e}", alpha=0.1)
            a1.scatter(x_stat_d[e]['median'], y_stat_d[e]['median'], alpha=1, color='k')
            a1.errorbar(x_stat_d[e]['mean'], y_stat_d[e]['mean'], y_stat_d[e]['std'], x_stat_d[e]['std'], 
                        color=sc.get_facecolors(), alpha=1, linewidth=3)

        a1.plot([0, 1], [0, 1], 'k')
        a1.set_xlabel("RBC Flux fraction")
        a1.set_ylabel("Entrance probability")
        a1.set_title(f"{x_name} fnan {fill_nan_method}")
        a1.grid()
        a1.legend()

        # TODO: Check for the conservation of flux? 
        if vis_csv_flx_Q: 
            a2 = f.add_subplot(1, 2, 2)
            a2.scatter(np.abs(edges_traces[p_el][x_name]), x_sum)
            vis_max_flux = np.nanmax(np.abs(edges_traces[p_el][x_name]))
            a2.plot([0, vis_max_flux], [0, vis_max_flux], 'k', alpha=0.5)
            a2.grid()
            a2.set_xlabel(f'Parent {x_name}')
            a2.set_ylabel(f'Sum of children {x_name}')

    return {'y': y_d, 'x': x_d, 'y_stat': y_stat_d, 'x_stat': x_stat_d}

def correct_bifurcation_flow_conservation_Pries(Qp, Q1, Q2): 
    """
        Procedure introduced by Pries et al. 
        Only works for degree-3 1-in-2-out node 
        Need to read the paper to figure out the derivation later. 
    
        Inputs: 
            
    """
    Wp = Qp / (Q1 + Q2)
    W1 = Q1 / (Qp + Q2)
    W2 = Q2 / (Qp + Q1)
    W_sum = Wp + W1 + W2
    Qp_c = Qp *  (1 + W1 + W2) / W_sum
    Q1_c = Q1 * (Wp + (Qp - Q2) / (Qp + Q2) + W2) / W_sum
    Q2_c = Q2 * (Wp + (Qp - Q1) / (Qp + Q1) + W1) / W_sum

    return Qp_c, Q1_c, Q2_c

def correct_node_flow_conservation_LS(Qp, Qcs): 
    """
        Force noisy flow measurement at the node to respect the conservation of flow using
        constrained least square estimation: 
            L = \sum_i^D (Q_i - \hat{Q_i})^2 + \lambda * (Qp - \sum_i^{D-1} Q_i)

        Inputs: 
            Qp: scalar or (T, ) np.array, where T is the number of time points
            Qcs: (D-1, T) np.array, where D is the node degree

    """
    node_degree = Qcs.shape[0] + 1
    child_sum = np.sum(Qcs, axis=0)
    Qp_c = (child_sum + (node_degree - 1) * Qp) / node_degree
    Qcs_c = []
    for c in range(node_degree - 1): 
        tmp_c = Qcs[c]
        tmp_q = (Qp + node_degree * tmp_c - child_sum) / node_degree
        Qcs_c.append(tmp_q)
    Qcs_c = np.vstack(Qcs_c)
    return Qp_c, Qcs_c

def analyze_node_flow_config_1_in_2_out(node_label, fg, edge_features, edges_traces, ptrace_result, p_pid_to_idx, p_el_to_idx, 
                                        mv_avg_wd, num_t_pts=None): 
    # n_fbf = {}
    # nb_el, nb_d_dir = fg.get_node_connected_edge_flow_dir(node_label, fg.edge_v_pxl)
    # assert np.count_nonzero(nb_d_dir == 0) == 0, "Exist edge with unknown flow direction"
    # n_fbf['p_el'] = int(nb_el[nb_d_dir == -1])
    # n_fbf['c_el'] = nb_el[nb_d_dir == 1]
    # ds_e_count = compute_edge_downstream_cell_partition_trace(fg, ptrace_result, p_pid_to_idx, p_el_to_idx, 
    #                                                                     n_fbf['p_el'], ds_edge_label=n_fbf['c_el'],
    #                                                                     mv_wd_sz=mv_avg_wd, num_t_pts=num_t_pts)
    if num_t_pts is None: 
        num_t_pts = ptrace_result.frame.max() + 1
    n_fbf, ds_e_count = compute_edge_downstream_cell_partition_trace_by_passenger(fg, node_label, mv_avg_wd, num_t_pts, mv_wd_min_n=3)
    for k in ['fw_etr_t', 'fw_count']: 
        n_fbf.pop(k, None)
    if len(ds_e_count) == 0: 
        n_fbf['p_num_cell'] = 0
        return n_fbf
    nb_el = n_fbf['nb_el']
    nb_d_dir = n_fbf['nb_d_dir']

    n_fbf['p_e_length'] = fg.abs_g.length[n_fbf['p_el']]
    n_fbf['c_e_length'] = fg.abs_g.length[n_fbf['c_el']]
    n_fbf['c_e_num_ep'] = 2 - fg.edge.num_connected_node[n_fbf['c_el']]

    n_f_dir = compute_node_flow_dir_angle(node_label, nb_el, nb_d_dir, fg.edge.connected_node_label, 
                                          edge_features['ep1_dir_vec'], edge_features['ep2_dir_vec'])
    n_fbf |= n_f_dir

    # child edges phase seperation probability mean and MSE
    n_fbf['num_data'] = np.asarray([ds_e_count[e]['tot_count'] for e in n_fbf['c_el']])
    n_fbf['p_num_cell'] = np.sum(n_fbf['num_data'])
    n_fbf['c_ps_p_m'] = np.asarray([ds_e_count[e]['avg_p'] for e in n_fbf['c_el']]) 
    n_fbf['c_ps_p_mse'] = np.asarray([ds_e_count[e]['avg_p_se'] for e in n_fbf['c_el']]) 

    e_list = [n_fbf['p_el']] + list(n_fbf['c_el'])
    # TODO: add uncertainty for the velocity and flux
    p_v_stat = {e: stat.compute_basic_statistics(edges_traces[e]['p_t_v_mean']) for e in e_list}
    e_avg_v = np.abs(np.asarray([p_v_stat[e]['median'] for e in e_list]))
    e_v_eff_std = np.abs(np.asarray([p_v_stat[e]['eff_ptrl_std'] for e in e_list]))
    
    # e_avg_v = np.abs(np.asarray([edges_traces[e]['stat_e_v']['mean'] if e in edges_traces else np.nan for e in e_list]))

    # e_v_std = np.abs(np.asarray([edges_traces[e]['stat_e_v']['std'] if e in edges_traces else np.nan for e in e_list]))
    e_est_r = np.asarray([edges_traces[e]['e_t_radius_t_avg'] if e in edges_traces else np.nan for e in e_list])
    e_est_vf = e_avg_v * (np.pi * e_est_r ** 2)

    n_fbf['e_avg_v'] = e_avg_v
    n_fbf['e_v_std'] = e_v_eff_std
    n_fbf['e_est_r'] = e_est_r
    tmp_sum, tmp_1, tmp_2 = correct_bifurcation_flow_conservation_Pries(e_est_vf[0], e_est_vf[1], e_est_vf[2])
    n_fbf['c_avg_vf_r'] = np.asarray([tmp_1 / tmp_sum, tmp_2 / tmp_sum])
    n_fbf['c_avg_vf_r_raw'] = e_est_vf[1:3] / np.sum(e_est_vf[1:3])

    tmp_sum, tmp_1, tmp_2 = correct_bifurcation_flow_conservation_Pries(e_avg_v[0], e_avg_v[1], e_avg_v[2])
    n_fbf['c_avg_v_r'] = np.asarray([tmp_1 / tmp_sum, tmp_2 / tmp_sum])
    tmp_r1, tmp_r2, tmp_r1_s, tmp_r2_s = stat.compute_ratio_uncertainty(e_avg_v[1], e_avg_v[2], e_v_eff_std[1], e_v_eff_std[2])
    n_fbf['c_avg_v_r_raw'] = np.asarray([tmp_r1, tmp_r2])
    n_fbf['c_avg_v_r_raw_se'] = np.asarray([tmp_r1_s, tmp_r2_s])

    n_fbf['c_avg_v_2_p'] = e_avg_v[1:3] / e_avg_v[0]
    n_fbf['c_avg_v_diff_n'] = 2 * np.abs(e_avg_v[1] - e_avg_v[2]) / (e_avg_v[1] + e_avg_v[2])

    return n_fbf

def compute_node_flow_dir_angle(nl, nb_el, nb_d_dir, edge_connected_node_label, ep1_dir_vec, ep2_dir_vec): 
    """
    Input: 
        nb: node label, integer scalar 
        nb_el: list of directly connected edge labels 
        nb_d_dir: flow direction of the connected edge labels. +1: flow out of the node; -1: flow into the node
        edge_connected_node_label: (N, 2) np.array, connected node label of the edges in FlowGraph
        ep1_dir_vec, ep2_dir_vec: (N, 3) np.array, edge endpoint vector
    """
    result = {}
    assert np.count_nonzero(nb_d_dir == 0) == 0, "Exist edge with unknown flow direction"
    result['p_el'] = int(nb_el[nb_d_dir == -1])
    result['c_el'] = nb_el[nb_d_dir == 1]
    e_ep_vec = {}
    for el, el_dir in zip(nb_el, nb_d_dir): 
        tmp_el_nl = edge_connected_node_label[el]
        tmp_n_idx = np.nonzero(nl == tmp_el_nl)[0]
        tmp_ep_vec = ep2_dir_vec[el] if tmp_n_idx == 1 else ep1_dir_vec[el]
        if el_dir > 0: # outward flowing 
            tmp_ep_vec = - tmp_ep_vec
        e_ep_vec[el] = tmp_ep_vec
    result['pc_fd_cos'] = np.asarray([np.sum(e_ep_vec[result['p_el']] * e_ep_vec[e]) for e in result['c_el']])
    result['pc_fd_agl_deg'] = np.arccos(result['pc_fd_cos']) / np.pi * 180
    return result

def get_edge_passenger_last_appearance(edge_passengers): 
    last_time = pid_u = None
    if len(edge_passengers): 
        edge_passengers = np.vstack(edge_passengers)
        # sort by time 
        tmp_s_idx = np.argsort(edge_passengers[:, 0].flatten())
        edge_passengers = edge_passengers[tmp_s_idx, :]
        pass_time = edge_passengers[:, 0].astype(np.uint16).flatten()
        pid = edge_passengers[:, 2].astype(np.uint64).flatten()
        tmp_p_idx, pid_u = util.bin_data_to_idx_list(pid)
        last_time = np.asarray([pass_time[tmp_idx[-1]] for tmp_idx in tmp_p_idx])
    return last_time, pid_u

def analyze_node_cell_forward_stat(node_label, fg, edge_v=None, return_time_Q=False):
    """
    For most of the edges, this is straightforward, but we need to deal with some corner cases: 
    1. Either $i$ and $j$ is too short and the cell just pass the entire edges directly. 
    2. The nearest neighbor assignment is not completely reliable. If the cell "enters" both of 
       the downstrea edges, we should only count the final edge it enters. This requires having the 
       tracking informaton. 

    So, for each edge, get all the passing cells. For each passing cell, get its trajectory and do 
    the counting. Passing cells were collected when constructing the voxel velocity map. This will 
    get all the cells that are detected in the downstream edge. 

    # fg is necessary here as we need the network structure... 

    Input: 
        node_label: integer scalar
        fg: FlowGraph instance. 

    """
    if edge_v is None: 
        # Use for determine flow direction only
        edge_v = fg.edge_v_xpl

    nb_el, nb_d_dir = fg.get_node_connected_edge_flow_dir(node_label, edge_v)
    # for each passing particle, get the last time the particle is in the edge
    e_passenger_info = {}
    for i, tmp_oe in enumerate(nb_el): 
        tmp_t, tmp_pid = get_edge_passenger_last_appearance(fg.edge_passenger[tmp_oe])
        if tmp_t is not None: 
            e_passenger_info[tmp_oe] = (tmp_t, tmp_pid)

    # For each inflow edge, get the number of particles ended up in each edge
    inflow_edge = nb_el[nb_d_dir == -1]
    nif_edge = nb_el[nb_d_dir != -1]
    n_fw_info = {'p_el': inflow_edge, 'nif_el': nif_edge, 'c_el': nb_el[nb_d_dir == 1], 
                 'nb_el': nb_el, 'nb_d_dir': nb_d_dir, 
                 'fw_count': {}, 'fw_etr_t': {}}
    e_forward_count = {}
    e_forward_etr_time = {}
    for tmp_ie in inflow_edge: 
        if tmp_ie in e_passenger_info: 
            tmp_i_t, tmp_i_pid = e_passenger_info[tmp_ie]
            tmp_i_pid_to_t = {k:v for k, v in zip(tmp_i_pid, tmp_i_t)}

            tmp_i_e_t, tmp_o_i, tmp_o_t, tmp_o_p = ([] for _ in range(4))
            for j, tmp_oe in enumerate(nif_edge): 
                if tmp_oe in e_passenger_info: 
                    # out-flow edge passengers (last_appear_time, pid)
                    tmp_t, tmp_pid = e_passenger_info[tmp_oe]
                    tmp_valid_Q = np.zeros(tmp_t.size, bool)
                    tmp_ie_time = []
                    # Downstream particle should appear no earlier than the upstream particle 
                    for i, v in enumerate(tmp_pid): 
                        if (v in tmp_i_pid_to_t): 
                            tmp_ie_last_time = tmp_i_pid_to_t[v]
                            if tmp_t[i] >= tmp_ie_last_time: 
                                tmp_valid_Q[i] = True
                                if return_time_Q: 
                                    tmp_ie_time.append(tmp_ie_last_time)

                    tmp_t = tmp_t[tmp_valid_Q]
                    tmp_pid = tmp_pid[tmp_valid_Q]
                    tmp_i_l = np.repeat(j, tmp_t.size)
                    tmp_i_e_t.append(np.asarray(tmp_ie_time))
                    tmp_o_i.append(tmp_i_l)
                    tmp_o_t.append(tmp_t)
                    tmp_o_p.append(tmp_pid)
            # If the same particle appears in multiple downstream edges (e.g. when the particle is 
            # close to the node but got assigned to the parallel branch first), count the final 
            # edge it ends up with. 
            if len(tmp_o_t) > 0: 
                
                tmp_o_t = np.concatenate(tmp_o_t)
                tmp_o_i = np.concatenate(tmp_o_i)
                tmp_o_p = np.concatenate(tmp_o_p)

                tmp_s_idx = np.argsort(tmp_o_t)
                tmp_o_i = tmp_o_i[tmp_s_idx]
                tmp_o_p = tmp_o_p[tmp_s_idx]
                
                tmp_idx, tmp_o_p_u = util.bin_data_to_idx_list(tmp_o_p)
                tmp_o_e_count = np.zeros(nif_edge.size, np.uint32)
                tmp_o_e_ie_last_time = {k: [] for k in nif_edge}

                if return_time_Q: 
                    tmp_i_e_t = np.concatenate(tmp_i_e_t)
                    tmp_i_e_t = tmp_i_e_t[tmp_s_idx]

                for idx in tmp_idx: 
                    ds_e_idx = tmp_o_i[idx[-1]]
                    tmp_o_e_count[ds_e_idx] += 1
                    # the last time a particle that enters ds edge was in the upstream edge
                    if return_time_Q: 
                        tmp_o_e_ie_last_time[nif_edge[ds_e_idx]].append(tmp_i_e_t[idx[-1]])

                e_forward_count |= {(tmp_ie, oe) : tmp_o_e_count[i] for i, oe in enumerate(nif_edge)}
                if return_time_Q: 
                    e_forward_etr_time |= {(tmp_ie, oe): tmp_o_e_ie_last_time[oe] for oe in nif_edge}
    
    n_fw_info['fw_count'] = e_forward_count
    n_fw_info['fw_etr_t'] = e_forward_etr_time
    return n_fw_info

def compute_forward_transition_matrix(fg, edge_v=None): 
    num_edges = fg.edge.num_cc
    fw_c_mat = np.zeros((num_edges, num_edges))
    for tmp_nl in range(fg.node.num_cc):
        n_fw_info = analyze_node_cell_forward_stat(tmp_nl, fg, edge_v=edge_v)
        tmp_e_fw_c = n_fw_info['fw_count']

        for ep, c in tmp_e_fw_c.items():
            fw_c_mat[ep[0], ep[1]] += c

    fw_prob_mat = fw_c_mat / np.sum(fw_c_mat, axis=1, keepdims=True)
    fw_prob_mat[np.isnan(fw_prob_mat)] = 0

    total_fw_prob = np.zeros(fw_prob_mat.shape)
    tmp_mat = np.eye(num_edges)
    tmp_count = 0
    while np.sum(tmp_mat) > 1e-6:
        tmp_mat = tmp_mat @ fw_prob_mat
        total_fw_prob += tmp_mat
        tmp_count += 1

    result = {'fw_prob_mat': fw_prob_mat, 'fw_count_mat': fw_c_mat, 'fw_eql_dist_mat': total_fw_prob}
    return result

def __fw_etr_t_dict_to_trace(fw_ext_t_dict:dict, num_pts): 
    etr_trace = defaultdict(list)
    p_el_l = []
    c_el_l = []

    for k, v in fw_ext_t_dict.items():
        tmp_p_el, tmp_c_el = k
        etr_trace[tmp_p_el].append(v)
        etr_trace[tmp_c_el].append(v)
        p_el_l.append(tmp_p_el)
        c_el_l.append(tmp_c_el)

    for k, v_pts in etr_trace.items():
        if len(v_pts): 
            v_pts = np.concatenate(v_pts).astype(np.uint32)
            v = np.zeros(num_pts)
            for tmp_v in v_pts: 
                v[tmp_v] += 1
        etr_trace[k] = np.asarray(v)
    
    # replace 0 with nan
    # for k, v in etr_trace.items():
    #     v[v == 0] = np.nan
    #     etr_trace[k] = v

    parent_trace_d = {k: etr_trace[k] for k in p_el_l}
    child_trace_d = {k: etr_trace[k] for k in c_el_l}
    
    return parent_trace_d, child_trace_d

def __analyze_1_in_2_out_trace_stat(parent_trace, child_trace_d: dict, mv_wd_sz=None, mv_wd_min_n=0): 
    ds_e_count = {}
    if isinstance(parent_trace, dict): 
        num_parent = len(parent_trace)
        if num_parent == 0: 
            return ds_e_count
        else: 
            assert len(parent_trace) == 1, 'More than 1 parent edge'
            parent_trace = list(parent_trace.values())[0]
    else: 
        assert isinstance(parent_trace, np.ndarray)

    # num_children = len(child_trace_d)
    # assert num_children == 2, f'Number of children is {num_children}'
    
    if mv_wd_sz is None: 
        parent_count_sum = parent_trace
    else:     
        parent_count_sum = pd.Series(parent_trace).rolling(window=mv_wd_sz, min_periods=mv_wd_min_n, center=True).sum().values
    
    parent_tot_count = np.nansum(parent_trace)
    parent_obs_frac = np.mean(parent_trace > 0)
    
    # Wald interval: $z_{\alpha} = 1.96 $ for 95\% interval
    # $$ p = \frac{n_s}{n} \pm \frac{z_{\alpha}}{\sqrt{n}}\sqrt{\frac{n_s}{n}\frac{n - n_s}{n}} $$ 
    # This formula is valid for multi-nominal estimation. 
    # For p_i - p_j, refer to https://en.wikipedia.org/wiki/Multinomial_distribution
    for dse, tmp_count in child_trace_d.items(): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp_total_count = np.nansum(tmp_count)
            if mv_wd_sz is None: 
                tmp_sum = tmp_count
            else: 
                tmp_sum = pd.Series(tmp_count).rolling(window=mv_wd_sz, min_periods=mv_wd_min_n, center=True).sum().values
            tmp_observed_fraction = np.mean(tmp_count > 0)

            tmp_p = tmp_sum / parent_count_sum
            tmp_std = 1 / np.sqrt(parent_count_sum) * np.sqrt((tmp_sum * (parent_count_sum - tmp_sum) / parent_count_sum ** 2))
            tmp_avg_p = tmp_total_count / parent_tot_count

            beta_dist_alpha_post = 1 + tmp_sum
            beta_dist_beta_post = 1 + parent_count_sum - tmp_sum
            ci_lower = beta.ppf(0.025, beta_dist_alpha_post, beta_dist_beta_post)
            ci_higher = beta.ppf(0.975, beta_dist_alpha_post, beta_dist_beta_post)

            # tmp_binomial_p = np.asarray([binomtest(int(k), int(n), p=tmp_avg_p, alternative='two-sided').pvalue for k, n in zip(tmp_sum, parent_count_sum)])

            ds_e_count[dse] = {'count': tmp_sum, 'p': tmp_p, 'p_se': tmp_std, 'p_ci_l': ci_lower, 'p_ci_h': ci_higher, 
                            'tot_count': tmp_total_count, 
                            'observed_fraction': tmp_observed_fraction, 
                            'parent_observed_fraction': parent_obs_frac, 
                            'parent_tot_count': parent_tot_count, 
                            'parent_count': parent_count_sum, 
                            'mv_wd_sz': mv_wd_sz, 
                            'avg_p': tmp_avg_p, 
                            'avg_p_se': 1 / np.sqrt(parent_tot_count) * np.sqrt((tmp_avg_p * (1 - tmp_avg_p))),
                            'p_to_avg_p_se': np.sqrt(tmp_avg_p * (1 - tmp_avg_p) / parent_count_sum)}
            ds_e_count[dse]['p_2_avg_p_z'] = np.abs(ds_e_count[dse]['p'] - ds_e_count[dse]['avg_p']) / ds_e_count[dse]['p_to_avg_p_se']
    return ds_e_count

    
def fit_phase_seperation_function(x, b): 
    return x ** b / (x ** b + (1 - x) ** b)


def fit_phase_seperation_data(x, y, visQ=False, nonlinear_fit_Q=True, y_std=None, nf_loss='linear', nf_f_scale=1.0): 
    if nonlinear_fit_Q: 
        if y_std is None: 
            b, b_e = curve_fit(fit_phase_seperation_function, x, y, p0=[1], method='trf', loss=nf_loss, f_scale=nf_f_scale)
        else: 
            sig = np.maximum(0.001, y_std)
            b, b_e = curve_fit(fit_phase_seperation_function, x, y, p0=[1], sigma=sig, absolute_sigma=True, method='trf', loss=nf_loss, f_scale=nf_f_scale)

        b = b[0]
        b_e = b_e[0][0]
        y_p = fit_phase_seperation_function(x, b)
        y_res = np.abs(y - y_p)
        ss_res = np.sum((y_res) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        result = {'num_data': x.size // 2, 'b': b, 'r2': r2, 'b_e': b_e}
    else: 
        fit_x_data = np.log((1 - x) / x)
        fit_y_data = np.log((1 - y) / y)
        fit_valid_Q = np.logical_and(np.isfinite(fit_x_data), np.isfinite(fit_y_data))
        fit_x_data = fit_x_data[fit_valid_Q].flatten()
        fit_y_data = fit_y_data[fit_valid_Q].flatten()
        # model = HuberRegressor()
        # model.fit(fit_x_data[:, None], fit_y_data)
        # result = {'num_data': fit_y_data.size, 'b': model.coef_[0], ''}
        model = sp.stats.linregress(fit_x_data, fit_y_data)
        result = {'num_data': fit_x_data.size // 2, 'b': model.slope, 'r2': model.rvalue ** 2, 'b_e': model.stderr}
    
    lr_b = result['b']
    if visQ: 
        f, a = plt.subplots(1, 1, figsize=(5, 4))
        # a[0].scatter(fit_x_data, fit_y_data)
        # plt_x = np.linspace(fit_x_data.min(), fit_x_data.max(), 50)
        # plt_y = plt_x * result['b']
        # a[0].plot(plt_x, plt_y)
        # a[0].plot()
        plt_x = np.linspace(0.01, 0.99, 50)
        plt_y = (plt_x ** lr_b / (plt_x ** lr_b + (1 - plt_x) ** lr_b))
        a.plot(plt_x, plt_y, 'k', label=f'b = {lr_b:.2f}')
        a.scatter(x.flatten(), y.flatten(), alpha=0.5)
        a.legend()
        a.grid()
        a.set_xlabel(f"X")
        a.set_ylabel(f"RBC entrance prob.")

    return result
#endregion

#region Flux analysis



#endregion



#region Multi-layer analysis 

def combine_tracked_detections_in_multiple_layers(tk_data_dict, fg):
    c_detections = []
    c_edge_v = defaultdict(list)
    c_edge_std = defaultdict(list)
    c_edge_frac = defaultdict(list)
    detection_name = 'tracking_ns'

    cum_particle_count = 0
    cum_did_count = 0
    cum_frame_count = 0
    z_list = sorted(list(tk_data_dict.keys()))
    whole_mask_size = fg.num['mask_size']
    for z_idx in z_list: 
        sv_tracking_data = tk_data_dict[z_idx]
        sv_detections = sv_tracking_data[detection_name].copy()
        detect_vol_shape = sv_tracking_data['sv_data']['mask_size']
        # Update coordinates
        sv_disp_vec = sv_tracking_data['sv_data']['disp_vec']

        sv_detections = NFTLinking.register_detections([sv_detections], disp_vec=sv_disp_vec, 
                                                       vol_shape=whole_mask_size, inplace_Q=False, 
                                                       verboseQ=False, detection_vol_shape=detect_vol_shape)[0]
        sv_detections['z_idx'] = z_idx
        sv_detections['frame'] = sv_detections['frame'] + cum_frame_count
        cum_frame_count = sv_detections.frame.max() + 2
        sv_detections['did'] = sv_detections['did'] + cum_did_count
        cum_did_count = sv_detections.did.max() + 1
        sv_detections['particle'] = sv_detections['particle'] + cum_particle_count
        cum_particle_count = sv_detections.particle.max() + 1
        
        sv_detections['skl_ind'] = fg.nearest_map.ind_to_nearest_ind(sv_detections.ind.values)
        sv_detections['edge_label'] = fg.edge.ind_to_label(sv_detections['skl_ind'].values)
        sv_detections['node_label'] = fg.node.ind_to_label(sv_detections['skl_ind'].values)
        c_detections.append(sv_detections)

        # register edge velocity 
        sv_e_ep_ind = sv_tracking_data['e_ep_ind'].flatten()
        sv_e_ep_sub = np.unravel_index(sv_e_ep_ind, detect_vol_shape)
        sv_e_ep_sub = [sub + sv_disp_vec[i] for i, sub in enumerate(sv_e_ep_sub)]
        sv_e_ep_ind = np.ravel_multi_index(sv_e_ep_sub, whole_mask_size)
        sv_e_ep_el = fg.edge.ind_to_label(sv_e_ep_ind)
        sv_e_ep_el = sv_e_ep_el.reshape(-1, 2)
        sv_e_ep_ind = sv_e_ep_ind.reshape(-1, 2)
        
        sv_e_v_pxl = sv_tracking_data['edge_v_pxl'].copy()
        sv_e_v_std_pxl = sv_tracking_data['edge_v_std_pxl']
        sv_e_track_frac = sv_tracking_data['edge_track_frac']

        for i, e_ep_els in enumerate(sv_e_ep_el): 
            tmp_v = sv_e_v_pxl[i]
            for j, tmp_el in enumerate(e_ep_els):
                if tmp_el >= 0: 
                    tmp_e_ind = fg.edge.cc_ind[tmp_el]
                    idx1 = np.nonzero(tmp_e_ind == sv_e_ep_ind[i, 0])[0]
                    idx2 = np.nonzero(tmp_e_ind == sv_e_ep_ind[i, 1])[0]
                    if idx1.size == 1 and idx2.size == 1: 
                        tmp_v = - tmp_v if idx2 < idx1 else tmp_v
                        break
                    elif idx1.size == 1: 
                        tmp_v = - tmp_v if idx1 == tmp_e_ind.size else tmp_v
                        break

            if tmp_el >= 0: 
                c_edge_v[tmp_el].append(tmp_v)
                c_edge_std[tmp_el].append(sv_e_v_std_pxl[i])
                c_edge_frac[tmp_el].append(sv_e_track_frac[i])
    c_detections = pd.concat(c_detections)
    return c_detections, c_edge_v, c_edge_std, c_edge_frac


def combine_edge_v_across_layers(c_detections, fg, c_edge_v, dm_select_para, verboseQ=False): 
    c_el_to_idx = util.get_table_value_to_idx_dict(c_detections, key='edge_label', filter=lambda x: x>=0)
    c_edge_v_pxl = np.full(fg.edge.num_cc, np.nan)
    c_edge_v_cflt_el = [] # conflicting edge label
    for tmp_el, v in c_edge_v.items():
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        if len(v) == 1: 
            c_edge_v_pxl[tmp_el] = v[0]
        elif len(v) > 1: 
            if np.all(v >= 0) or np.all(v <= 0):
                # same direction
                c_edge_v_pxl[tmp_el] = np.mean(v)
            else: 
                # opposite direction 
                test_ef = fg.get_edgeflow_object(tmp_el, c_detections.iloc[c_el_to_idx[tmp_el]])
                test_dm = test_ef.iterative_est_avg_velocity_from_detection_map(test_ef.detect_map, vis_Q=False)
                
                test_dm_v, test_dm_v_std = NFTLinking.select_edge_velocity_from_correlation_analysis(test_dm, **dm_select_para)
                if np.isfinite(test_dm_v): 
                    if verboseQ: 
                        print(f"Edge {tmp_el}: ori v est: {v}. new v est {test_dm_v}. Average over consistent v.")
                    tmp_selected_Q = (v * test_dm_v) > 0
                    c_edge_v_pxl[tmp_el] = np.mean(v[tmp_selected_Q])
                else: 
                    if verboseQ: 
                        print(f"Edge {tmp_el}: Unable to determine the velocity direction. {v}, {test_dm_v}")
                    c_edge_v_cflt_el.append(tmp_el)
    
    return c_edge_v_pxl, c_edge_v_cflt_el

def get_cc_label_map_in_subvolume_graphs(fg, bbox_mm_zyx, bbox_xx_zyx, layer_stat, cc_name): 
    if cc_name == 'edge': 
        unique_map_Q = False
    elif cc_name == 'node': 
        unique_map_Q = True
    else: 
        raise ValueError(F"Invalid cc_name {cc_name}")
    
    g_cc_obj = getattr(fg, cc_name)
    bbox_ll = bbox_xx_zyx - bbox_mm_zyx
    map_gel_to_svel = defaultdict(list)
    for tmp_el in range(g_cc_obj.num_cc): 
        tmp_el_ind = g_cc_obj.cc_ind[tmp_el]
        tmp_el_sub = np.column_stack(fg.ind2sub(tmp_el_ind))  
        tmp_vxl_in_bboxs_Q = np.logical_and(np.all(tmp_el_sub[:, None, :] >= bbox_mm_zyx[None, :, :], axis=2), 
                                    np.all(tmp_el_sub[:, None, :] < bbox_xx_zyx[None, :, :], axis=2)) 
        tmp_found_Q = False
        for tmp_sv_idx, tmp_sv_mask_Q in enumerate(tmp_vxl_in_bboxs_Q.T): 
            if np.any(tmp_sv_mask_Q):
                tmp_sv_bbox_ll = bbox_ll[tmp_sv_idx]
                tmp_sv_mm = bbox_mm_zyx[tmp_sv_idx].copy()
                tmp_sv_mm[1:] = 0 # This is pretty tricky... the subgraphs themself were cropped from the stitched graph so there's no need to shift along the xy direction 
                
                tmp_sv_sub = np.atleast_2d(tmp_el_sub)
                tmp_sv_sub = tmp_sv_sub[tmp_sv_mask_Q, :]
                tmp_sv_sub = tmp_sv_sub - tmp_sv_mm
                tmp_in_bbox_Q = np.logical_and(np.all(tmp_sv_sub >= 0, axis=1), np.all(tmp_sv_sub < tmp_sv_bbox_ll, axis=1))
                tmp_sv_sub = tmp_sv_sub[tmp_in_bbox_Q]
                
                tmp_sv_ind = np.ravel_multi_index([tmp_sv_sub[:, i] for i in range(tmp_sv_sub.shape[1])], tmp_sv_bbox_ll)
                # This can be simplified to only pass the graph - need to construct a new dictionary
                tmp_sv_cc_obj = getattr(layer_stat[tmp_sv_idx]['fg'], cc_name)

                tmp_sv_cc_l = np.asarray(tmp_sv_cc_obj.ind_to_label(tmp_sv_ind)) 
                tmp_idx, tmp_sv_cc_l = util.bin_data_to_idx_list(tmp_sv_cc_l)
                tmp_valid_Q = tmp_sv_cc_l >= 0
                tmp_sv_cc_l = tmp_sv_cc_l[tmp_valid_Q]
                
                if tmp_sv_cc_l.size: 
                    if unique_map_Q: 
                        assert tmp_sv_cc_l.size == 1, "Node voxel in the stitched graph correspond to more than 1 node in one sub-volume"
                        for tmp in list(tmp_sv_cc_l): 
                            map_gel_to_svel[tmp_el].append((tmp_sv_idx, int(tmp)))
                    else: 
                        tmp_idx = tmp_idx[tmp_valid_Q]
                        for i, tmp in enumerate(list(tmp_sv_cc_l)): 
                            # determine segment dir 
                            tmp_seg_ind = tmp_sv_ind[tmp_idx[i]]
                            tmp_cc_ind = tmp_sv_cc_obj.cc_ind[tmp]
                            idx1 = np.nonzero(tmp_cc_ind == tmp_seg_ind[0])[0]
                            idx2 = np.nonzero(tmp_cc_ind == tmp_seg_ind[-1])[0]
                            tmp_seg_dir = 0
                            if idx1.size == 1 and idx2.size == 1: 
                                tmp_seg_dir = -1 if idx2 < idx1 else 1
                            else: 
                                raise ValueError(f"Segment {tmp_sv_ind} is not in sv {tmp_sv_idx} edge {tmp} cc {tmp_cc_ind}")

                            map_gel_to_svel[tmp_el].append((tmp_sv_idx, int(tmp), tmp_seg_dir))

                    tmp_found_Q = True
        
        if not tmp_found_Q: 
            print(f"Node {tmp_el} does not exist in any of the sub-volumes.\n")
        else: 
            print(f"\rFinish matching {cc_name} {tmp_el}.", end='', flush=True)
    
    return map_gel_to_svel


#endregion 