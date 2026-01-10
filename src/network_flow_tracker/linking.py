import numpy as np 
import warnings
from collections import defaultdict
import pandas as pd
from scipy.spatial import KDTree
import skimage.morphology as skim

from .utils import neighbors as nb
from .utils import util as util

import network_flow_tracker.LFBFP as LFBFP
import trackpy as tp
from trackpy import linking as tpl
import network_flow_tracker.particle as particle
import network_flow_tracker.FlowGraph as FG

#region Tracking function 
def tracking(detections: pd.DataFrame, search_range, pos_col=['z', 'y', 'x'], 
             feature_col=None, dist_func=None, output_type='table'):

    t_to_idx = util.get_table_value_to_idx_dict(detections, key='frame')
    frame_l = sorted(list(t_to_idx.keys()))
    if (frame_l[-1] - frame_l[0] + 1) != len(frame_l): 
        print(f"The data are not continuous in time")

    assert 'did' in detections, 'Missing did (detection ID) column in detections'
    num_detect = detections.shape[0]
    result_dict = {'particle': np.full(num_detect, -1, dtype=np.int32), 
                   'cost': np.full(num_detect, np.nan, dtype=np.float32), 
                   'alt_cost': np.full(num_detect, np.nan, dtype=np.float32), 
                   'num_cands': np.zeros(num_detect, dtype=np.uint8), 
                   'exit_ind': np.full(num_detect, -1, dtype=np.int64), 
                   'dist_to_ep': np.full(num_detect, np.nan, dtype=np.float32), 
                   'exit_v': np.full(num_detect, np.nan, dtype=np.float32)
                   }

    for t in range(frame_l[0], frame_l[-1] + 1): 
        if t in t_to_idx: 
            t_idx = t_to_idx[t]
        else: 
            raise NotImplementedError
        t_table = detections.iloc[t_idx]
        t_coords = t_table[pos_col].to_numpy()
        if feature_col is not None: 
            t_features = {k: t_table[k].values for k in feature_col}
        else: 
            t_features = None

        if t == 0: 
            linker = tpl.Linker(search_range, adaptive_stop=1, adaptive_step=0.9, 
                                graph_dist_func=dist_func)
            linker.init_level(t_coords, t, extra_data=t_features)        
        else: 
            prev_hash = linker.update_hash(t_coords, t, extra_data=t_features)

            linker.subnets = tpl.subnet.Subnets(prev_hash, linker.hash, linker.search_range,
                                linker.MAX_NEIGHBORS, graph_dist_func=linker.graph_dist_func)

            spl, dpl = linker.assign_links()
            linker.apply_links(spl, dpl)
            # Store the cost
            # prev_did = np.asarray([p.extra_data['did'] for p in prev_hash.points])

            t_result = {k: v[t_to_idx[t - 1]] for k, v in result_dict.items() if k != 'particle'}

            # num_pts = len(prev_hash.points)
            # t_result = {'num_cands': np.zeros(num_pts, dtype=np.uint8), 
            #             'cost': np.full(num_pts, np.nan, dtype=np.float32), 
            #             'alt_cost': np.full(num_pts, np.nan, dtype=np.float32), 
            #             'exit_ind': np.full(num_pts, -1, dtype=np.int64), 
            #             'exit_v': np.full(num_pts, np.nan, dtype=np.float32)
            #             }
            for i, p in enumerate(prev_hash.points): 
                t_result['num_cands'][i] = p.num_forward_cands
                t_result['cost'][i] = p.track_cost

                tmp_alternative_cost = np.asarray(p.forward_cost)
                if tmp_alternative_cost.size > 0: 
                    tmp_non_track_cost_Q = tmp_alternative_cost != p.track_cost
                    tmp_alternative_cost = tmp_alternative_cost[tmp_non_track_cost_Q]
                    if tmp_alternative_cost.size > 0: 
                        t_result['alt_cost'][i] = np.min(tmp_alternative_cost)
                    elif np.count_nonzero(tmp_non_track_cost_Q) > 1: 
                        t_result['alt_cost'][i] = p.track_cost
                
                if 'exit_info' in p.extra_data: 
                    tmp_ep_ind, t_result['dist_to_ep'][i], t_result['exit_v'][i] = p.extra_data['exit_info']
                    if 'exitQ' in p.extra_data: 
                        t_result['exit_ind'][i] = tmp_ep_ind

            for k, v in t_result.items():
                result_dict[k][t_to_idx[t - 1]] = v
            # num_cands[t_to_idx[t - 1]] = np.asarray([p.num_forward_cands for p in prev_hash.points])
            # cost[t_to_idx[t - 1]] = np.asarray([p.track_cost for p in prev_hash.points])

        # curr_did = np.asarray([p.extra_data['did'] for p in linker.hash.points])
        
        assert np.all(result_dict['particle'][t_idx] == -1), 'Particle id has been labeled.'
        result_dict['particle'][t_idx] = linker.particle_ids
        print(f"\rFinish processing frame {t}. ", end='', flush=True)
    
    if output_type == 'table': 
        result = detections.copy()
        for k, v in result_dict.items(): 
            result[k] = v
    elif output_type == 'dict':
        result = {'t_to_idx': t_to_idx, 'frames': frame_l} | result_dict 
    print(f"Finish tracking cells. ")
    return result

def track_stationary_particles(detections: pd.DataFrame, mask_size, min_vxl_detect=70, search_r=1, max_speed_pxl=1, \
    min_trace_length=5, min_hccc_max_length=10, max_time_gap=5, min_final_trace_length=35):
    # min_num_frame = int(1 * data_info['frame_rate_Hz'])

    # Find the 3D connected component where the voxel detection count is higher than a threshold
    list_idx, detect_ind = util.bin_data_to_idx_list(detections.ind.values)
    num_count = np.asarray(list(map(lambda x: x.size, list_idx)))
    high_count_Q = num_count > min_vxl_detect
    high_count_ind = detect_ind[high_count_Q]
    high_count_mask = np.zeros(mask_size, bool)
    high_count_mask.flat[high_count_ind] = True
    high_count_mask = skim.dilation(high_count_mask, skim.ball(search_r))

    # Find the detections within these connected components, track them within these connected components 
    need_tracking_Q = high_count_mask.flat[detections.ind.values]
    active_detections = detections[need_tracking_Q].sort_values(by='did')
    if len(active_detections) == 0:
        print(f"No detections found in high count regions.")
        return []
    
    trace_result = tracking(active_detections, max_speed_pxl, pos_col=['z', 'y', 'x'], output_type='dict')
    selected_trace_ind = get_particle_trajectory_from_tracking(trace_result['particle'], min_trace_length, sortedQ=0)
    print(f"Found {selected_trace_ind.size} trajectories of length at least {min_trace_length}")

    hc_cc = nb.bwconncomp(high_count_mask, return_labeled_array_Q=True)
    print(f"Found {hc_cc['num_cc']} high count connected components.")
    hc_lb_array = hc_cc['labeled_array']
    hc_cc_traces = defaultdict(list)

    # The following part is not very efficient. To be improved. The main bottleneck is probably particle instance construction. 

    # bin traces according to cc
    for tmp_p_idx in selected_trace_ind:
        tmp_table = active_detections.iloc[tmp_p_idx]
        tmp_particle = particle.Particle(tmp_table)
        tmp_cc_label = np.unique(hc_lb_array.flat[tmp_table.ind.values]) 
        assert(np.all(tmp_cc_label > 0) and tmp_cc_label.size == 1), print(f"{tmp_cc_label}")
        hc_cc_traces[int(tmp_cc_label)].append(tmp_particle)

    # Select cc with sufficiently long trajectories and sort particles 
    hc_cc_traces = {k: sorted(v, key=lambda x : x.first_frame) for k, v in hc_cc_traces.items() \
        if max([p.num_frame for p in v]) >= min_hccc_max_length}

    # Merging nearby particles
    for k, v in hc_cc_traces.items():
        new_list = [v[0]]
        for p in v[1:]:
            pdist_ts = particle.dist_ts(new_list[-1], p)
            if pdist_ts[0] <= max_time_gap: 
                new_list[-1].merge_with(p)
            else: 
                new_list.append(p)
        hc_cc_traces[k] = new_list
    # Get rid of short, unconnected trajectories 
    hc_cc_traces = {k : [p for p in v if p.num_frame > min_final_trace_length] for k, v in hc_cc_traces.items()}
    # TODO: Add optional selection here ? 

    # Get all particles from the dictionary of list of particles
    p_list = [p for pl in hc_cc_traces.values() for p in pl]
    return p_list

def analyze_edge_velocity_from_tracking(fg:FG.FlowGraph, trace_result, valid_traces_ind, particle_key='particle', 
                                        gm_max_num_est=3, min_data_size=5, min_gmc_dist_std_n=2):
    skl_speed_map, _ = fg.construct_vxl_speed_map(trace_result, valid_traces_ind, include_node_Q=True, 
                                               return_type='array', update_fg_Q=True, particle_key=particle_key)
    
    pvalid_edge_labels = trace_result.edge_label.values[np.concatenate(valid_traces_ind)]
    tmp_p_idx, tmp_el = util.bin_data_to_idx_list(pvalid_edge_labels)
    pvalid_e_count = {e : i.size for e, i in zip(tmp_el, tmp_p_idx)}

    print(f"\nFinish constructing skeleton speed map.")
    trace_el_to_idx = util.get_table_value_to_idx_dict(trace_result, key='edge_label', filter=lambda x: x>=0)
    record_keys = ['est_v', 'est_v_std', 'weight_ratio', 'frac_tracked_frame', 'num_tracked_detection', 
                   'frac_tracked_detection', 'frac_tk_int_detection', 'same_dir_Q', 'forced_1_comp_Q', 
                   'exit_Q', 'length']
    edge_tk_v = defaultdict(lambda : np.full(fg.edge.num_cc, np.nan))
    for i, test_el in enumerate(trace_el_to_idx.keys()):
        test_ef = fg.get_edgeflow_object(test_el, trace_result.iloc[trace_el_to_idx[test_el]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = test_ef.analyze_voxel_velocity_map(test_ef.vxl_speed_map, gm_max_num_est=gm_max_num_est,
                                                                min_data_size=min_data_size, min_gmc_dist_std_n=min_gmc_dist_std_n, 
                                                                v_type='map', vis_Q=False)
        
        result['exit_Q'] = (test_ef.num_endpoints == 1) and (result['est_v'] > 0)
        result['length'] = test_ef.length
        # could be greater than 1 as the velocity map is constructed with splitted particle trajectories
        result['frac_tracked_detection'] = result['num_tracked_detection'] / test_ef.detections.shape[0]
        if test_el in pvalid_e_count: 
            result['frac_tk_int_detection'] = pvalid_e_count[test_el] / test_ef.detections.shape[0]
        else: 
            result['frac_tk_int_detection'] = 0
        for k in record_keys:
            if k in result: 
                edge_tk_v[k][test_el] = result[k]
        if i % 10 == 0: 
            print(f"\rFinish analyzing flow velocity in edge {i}. ", end='', flush=True)
    print(f"\nFinish analyzing edge velocity from the tracking result")
    return edge_tk_v

def select_edge_velocity_from_tracking_analysis(p_edge_tk_info, min_weight_ratio=1.5, min_track_fraction=0.3, 
                                                max_std_to_ignore_cv=1, max_abs_cv=0.8): 
    # need a better name for max_std_to_ignore_cv... 
    edge_trkb_v_std = p_edge_tk_info['est_v_std'].copy()
    edge_trkb_v = p_edge_tk_info['est_v'].copy()
    with np.errstate(divide='ignore'):
        edge_trkb_cv = np.abs(edge_trkb_v_std / edge_trkb_v)
    is_valid_Q = np.logical_or(edge_trkb_v_std < max_std_to_ignore_cv, edge_trkb_cv < max_abs_cv)
    is_valid_Q = np.logical_and(is_valid_Q, p_edge_tk_info['weight_ratio'] > min_weight_ratio)
    is_valid_Q = np.logical_and(is_valid_Q, p_edge_tk_info['frac_tracked_detection'] > min_track_fraction)
    edge_trkb_v[~is_valid_Q] = np.nan
    edge_trkb_v_std[~is_valid_Q] = np.nan

    return edge_trkb_v, edge_trkb_v_std

def estimate_flow_velocity_using_detection_map(fg:FG, detections, para_dm, para_dms, para_hfs, para_mea): 

    dm_log = fg.analyze_edges_detection_map(detections, **para_dm)
    reliable_edge_v, reliable_edge_v_std = select_edge_velocity_from_correlation_analysis(dm_log, **para_dms)

    # Select high-flow vessel for secondary analysis: 
    is_hc_Q = dm_log['num_hc_vxl'] > para_hfs['min_hc_num']
    hc_e_to_est = np.nonzero(np.logical_and(is_hc_Q, np.isnan(reliable_edge_v)))[0]


    # for edge with high detection counts, we assume a high flow rate associated with it and 
    # therefore, the reason corr-based velocity estimation failed is because the edge itself is not
    # long enough. Therefore, we want to find a long edge that is its downstream. 
    # For PA, requiring two connected edges to have the same flow direction in the sub-correlation matrix 
    # does not work as the flow in the second edge might still be too fast to determine in the sub-array. 
    # In this case, shall we just assign both edges with the same estimated velocity? 
    # But the correlaton based estiamtion actually just make sense for the first segment (the upstream one)
    el_to_idx = util.get_table_value_to_idx_dict(detections, key='edge_label', filter=lambda x: x>=0)
    est_results = {}
    for test_el in hc_e_to_est: 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp_result = fg.estimate_v_using_multi_edge_detection_map(test_el, detections, el_to_idx, 
                                                                    reliable_edge_v, **para_mea)
            tmp_result = fg.select_multi_edge_detection_map_results(tmp_result)
        if tmp_result is not None: 
            est_results[test_el] = tmp_result

    # Update
    for k, d in est_results.items():
        tmp_v = d['avg_v']
        if np.abs(tmp_v) >= para_hfs['min_high_v']: 
            # print(f"Update velocity in edge {k} to be {tmp_v}")
            reliable_edge_v[k] = tmp_v
            reliable_edge_v_std[k] = d['avg_v_std']
        else: 
            pass
            # print(f"Edge {k} has velocity {tmp_v:.2f} which is lower than the update limit of {para_hfs['min_high_v']}")

    return dm_log, reliable_edge_v, reliable_edge_v_std

def estimate_flow_velocity_using_tracking(fg, trace_result, para_ts, para_teve, para_teves): 

    # # trace selection 
    # para_ts = {'min_length': 4, 
    #         'min_cos': 0.25, 
    #         'max_ignore_v': 2} 
    # # tracking-based edge velocity estimation 
    # para_teve = {'gm_max_num_est': 3,  # maximum number of mixture-of-gaussian fitting (random initialization) 
    #             'min_data_size': 5,  # minimal data size for fitting
    #             'min_gmc_dist_std_n': 2} # minimal mixture-of-gaussian normalized mean difference. If 2 components fit the distribution better but their spacing is smaller than this value, re-do the fitting
    # para_teves = {'min_weight_ratio': 1.5, 
    #             'min_track_fraction': 0.3, 
    #             'max_abs_cv': 0.8, 
    #             'max_std_to_ignore_cv': 1}
    
    long_traces_ind = get_particle_trajectory_from_tracking(trace_result.particle.values, para_ts['min_length'], sortedQ=0) 
    
    valid_traces_ind = select_direction_consistent_traces(trace_result, long_traces_ind, **para_ts)

    edge_tk_v_results = analyze_edge_velocity_from_tracking(fg, trace_result, valid_traces_ind, 
                                                            **para_teve)

    edge_trkb_v, edge_trkb_v_std = select_edge_velocity_from_tracking_analysis(edge_tk_v_results, **para_teves)

    return edge_tk_v_results, edge_trkb_v, edge_trkb_v_std

def select_edge_velocity_from_correlation_analysis(dm_log, min_tot_corr_r=2, min_major_corr=0.25,
                                                   max_cv=0.8, max_ok_std=2.0):
    high_confidence_est_Q = dm_log['total_corr_ratio'] >= min_tot_corr_r
    high_confidence_est_Q = np.logical_and(high_confidence_est_Q, dm_log['major_total_corr'] > min_major_corr)
    if max_cv is not None: 
        valid_variance_Q = np.logical_or(np.abs(dm_log['major_diff_cv']) < max_cv, 
                                         dm_log['avg_v_std'] < max_ok_std)
        high_confidence_est_Q = np.logical_and(high_confidence_est_Q, valid_variance_Q)
    reliable_edge_v = np.asarray(dm_log['avg_v'].copy())
    reliable_edge_v[~high_confidence_est_Q] = np.nan
    reliable_edge_v_std = np.asarray(dm_log['avg_v_std'].copy())
    reliable_edge_v_std[~high_confidence_est_Q] = np.nan
    return reliable_edge_v, reliable_edge_v_std

def combine_initial_edge_v_estimation(v_corr_result, v_corr, std_corr, v_trk, std_trk, 
                                      max_tk_v, min_std=2, max_v_diff_z=2, 
                                      min_nonzero_v=1, log_conflict_Q=True): 
    num_edge = v_corr.size

    edge_v = np.full((num_edge, ), np.nan)
    edge_v_std = np.full((num_edge, ), np.nan)
    result = {'e_missing': [], 'e_w_div_v_dir': [], 'e_w_large_v_diff': []}

    for el in range(num_edge):
        tmp_corr_v = v_corr[el]
        tmp_corr_std = std_corr[el]
        tmp_tk_v = v_trk[el]
        tmp_tk_v_std = std_trk[el]

        if np.isnan(tmp_corr_v) and np.isfinite(tmp_tk_v):
            edge_v[el] = tmp_tk_v
            edge_v_std[el] = tmp_tk_v_std
        
        elif np.isnan(tmp_tk_v) and np.isfinite(tmp_corr_v):
            edge_v[el] = tmp_corr_v
            edge_v_std[el] = tmp_corr_std

        elif np.isfinite(tmp_corr_v) and np.isfinite(tmp_tk_v):
            
            tmp_corr_v = tmp_corr_v if np.abs(tmp_corr_v) > min_nonzero_v else 0
            tmp_tk_v = tmp_tk_v if np.abs(tmp_tk_v) > min_nonzero_v else 0

            if tmp_corr_v * tmp_tk_v < 0: 
                result['e_w_div_v_dir'].append(el)
                print(f"Edge {el} has inconsistent velosity estimation: corr-based: {tmp_corr_v}, tracking-based {tmp_tk_v}. Skip.")
                if log_conflict_Q:
                    continue
                else: 
                    raise ValueError(f"Inconsistent velocity direction in edge {el}")
            
            tmp_ad_v = np.abs(tmp_corr_v - tmp_tk_v)
            if np.isfinite(tmp_corr_std) and np.isfinite(tmp_tk_v_std):
                tmp_s = np.sqrt((tmp_corr_std ** 2 + tmp_tk_v_std ** 2) / 2)
            else:
                assert np.isfinite(tmp_tk_v_std)
                tmp_s = tmp_tk_v_std

            if (tmp_ad_v / np.maximum(min_std, tmp_s)) < max_v_diff_z: 
                # consistent estimated velocity 
                edge_v[el] = (tmp_corr_v + tmp_tk_v) / 2
                edge_v_std[el] = tmp_s
            else:
                # if exiting exit & length < v - both estimations are unreliable ... 

                if (np.abs(tmp_tk_v) > max_tk_v) and (np.abs(tmp_corr_v) > np.abs(tmp_tk_v)):
                    # Unrelable tracking-based velocity estimation
                    print(f"Edge {el} has v_t {tmp_tk_v:.2f}, which is higher than the reliable threshold of {max_tk_v}. Use corr-based velocity {tmp_corr_v:.2f}")
                    edge_v[el] = tmp_corr_v
                    edge_v_std[el] = tmp_corr_std
                elif (np.abs(tmp_tk_v) > v_corr_result['max_estimatable_v'][el]): 
                    print(f"Edge {el} has v_t greater than the maximum corr-based estimatable velocity. Use tracking-based velocity. ")
                    edge_v[el] = tmp_tk_v
                    edge_v_std[el] = tmp_tk_v_std
                else: 
                    print(f"Edge {el} v_corr {tmp_corr_v:.2f} +/- {tmp_corr_std:.2f}, v_t {tmp_tk_v:.2f} +/- {tmp_tk_v_std:.2f}")
                    if log_conflict_Q: 
                        result['e_w_large_v_diff'].append(el)
                    else: 
                        raise ValueError(f"Debug edge {el}")
        else: 
            result['e_missing'].append(el)

    print(f"Number of edges without any good velocity estimation: {len(result['e_missing'])}")
    print(f"Number of edges with at least one good velocity estimation: {np.count_nonzero(np.isfinite(edge_v))}")
    result['v'] = edge_v
    result['std'] = edge_v_std
    return result

def compute_direction_consistent_subtraces(trace_result, min_length=4, min_cos=0.25, max_ignore_v=2):
    valid_traces_ind = []
    long_traces_ind = get_particle_trajectory_from_tracking(trace_result.particle.values, min_length, sortedQ=0) 
    print(f"Found {long_traces_ind.size} trajectories of length at least {min_length}")

    for i_trace in range(long_traces_ind.size):
        p_t_idx = long_traces_ind[i_trace]
        p = particle.Particle(trace_result.iloc[p_t_idx])
        tmp_t_msk = p.get_trajectory_mask_by_adj_v_cos(min_cos, max_ignore_v)           
        tmp_t_ints = util.get_intervals_in_1d_binary_array(tmp_t_msk)
        tmp_t_int_len = tmp_t_ints[:, 1] - tmp_t_ints[:, 0]
        # TODO: maybe include more selections on exiting particles
        if tmp_t_int_len.size > 0 and p.exit_network_Q: 
            tmp_t_int_len[-1] += 1            
        tmp_selected_int_Q = (tmp_t_int_len >= min_length)
        tmp_t_ints = tmp_t_ints[tmp_selected_int_Q]        
        for tmp_int in tmp_t_ints: 
            valid_traces_ind.append(p_t_idx[tmp_int[0] : tmp_int[1]])
            
        if i_trace % 50 == 0: 
            print(f"\rFinish processing trace {i_trace}", end='', flush=True)
    valid_traces_ind = np.asarray(valid_traces_ind, object) 
    return valid_traces_ind

def select_direction_consistent_traces(trace_result, trace_t_ind, mask_size,
                                        min_length=3, min_med_cos=0.75, min_cos=0, max_ignore_v=2):
    valid_traces_ind = []
    for i_t, t_idx in enumerate(trace_t_ind):
        tmp_valid_subtraces = particle.compute_direction_consistent_subtraces(trace_result, t_idx, mask_size=mask_size, 
                                                                              min_length=min_length, min_med_cos=min_med_cos, 
                                                                              min_cos=min_cos, max_ignore_v=max_ignore_v)
        valid_traces_ind.extend(tmp_valid_subtraces)       
        if i_t % 50 == 0: 
            print(f"\rFinish processing trace {i_t}. ", end='', flush=True)

    valid_traces_ind = np.asarray(valid_traces_ind, object) 
    print(f"Found {valid_traces_ind.size} valid trajectories of length at least {min_length}")
    return valid_traces_ind

def update_edge_flow_estimation(old_v, old_v_std, old_track_frac, tk_result, max_v_diff_z=2, min_nonzero_v=1, 
                                min_std=1, min_tk_f=0.1, min_trust_f=0.5, allow_conflicting_Q=True):  
    
    num_edge = old_v.size
    if old_track_frac is None:
        old_track_frac = np.full((num_edge, ), np.nan)
    new_est_edge_v = np.full((num_edge, ), np.nan)
    new_est_edge_v_std = np.full((num_edge, ), np.nan)
    new_est_edge_tk_f = np.full((num_edge, ), np.nan)
    info = {'e_missing': [], 'e_exit': [], 'e_w_div_v_dir': [], 'e_unresolved': [], 'e_t_tk': []}
    for el in range(num_edge):
        tmp_curr_v = old_v[el]
        tmp_curr_std = old_v_std[el]
        tmp_curr_f = old_track_frac[el]

        if tk_result['frac_tracked_detection'][el] > min_tk_f: 
            tmp_tk_v = tk_result['est_v'][el] 
            tmp_tk_v_std = tk_result['est_v_std'][el]
            tmp_tk_f = tk_result['frac_tk_int_detection'][el]
        else: 
            tmp_tk_v = np.nan
            tmp_tk_v_std = np.nan
            tmp_tk_f = np.nan

        if np.isnan(tmp_curr_v) and np.isfinite(tmp_tk_v):
            new_est_edge_v[el] = tmp_tk_v
            new_est_edge_v_std[el] = tmp_tk_v_std
            new_est_edge_tk_f[el] = tmp_tk_f

        elif np.isnan(tmp_tk_v) and np.isfinite(tmp_curr_v):
            new_est_edge_v[el] = tmp_curr_v
            new_est_edge_v_std[el] = tmp_curr_std
            new_est_edge_tk_f[el] = tmp_curr_f

        elif np.isfinite(tmp_curr_v) and np.isfinite(tmp_tk_v):
            tmp_curr_v = tmp_curr_v if np.abs(tmp_curr_v) > min_nonzero_v else 0
            tmp_tk_v = tmp_tk_v if np.abs(tmp_tk_v) > min_nonzero_v else 0

            if tmp_curr_v * tmp_tk_v < 0: 
                info['e_w_div_v_dir'].append(el)
                if allow_conflicting_Q: 
                    print(f"Edge {el}: Inconsistent velocity direction: {tmp_curr_v} vs {tmp_tk_v}")
                    continue
                else: 
                    raise ValueError(f"Edge {el}: Inconsistent velocity direction: {tmp_curr_v} vs {tmp_tk_v}")
            
            tmp_ad_v = np.abs(tmp_curr_v - tmp_tk_v)
            if np.isfinite(tmp_curr_std) and np.isfinite(tmp_tk_v_std):
                tmp_s = np.sqrt((tmp_curr_std ** 2 + tmp_tk_v_std ** 2) / 2)
            else:
                assert np.isfinite(tmp_tk_v_std)
                tmp_s = tmp_tk_v_std
            to_exit_Q = (tk_result['exit_Q'][el] == 1) and (tmp_tk_v > 0)
            if (tmp_ad_v / np.maximum(min_std, tmp_s)) < max_v_diff_z: 
                # consistent estimated velocity 
                if np.isnan(tmp_curr_f) or to_exit_Q: 
                    new_est_edge_v[el] = (tmp_curr_v + tmp_tk_v) / 2
                    new_est_edge_v_std[el] = tmp_s 
                    new_est_edge_tk_f[el] = tmp_tk_f # not sure
                elif tmp_curr_f < tmp_tk_f : 
                    new_est_edge_v[el] = tmp_tk_v
                    new_est_edge_v_std[el] = tmp_tk_v_std
                    new_est_edge_tk_f[el] = tmp_tk_f
                else: 
                    new_est_edge_v[el] = tmp_curr_v
                    new_est_edge_v_std[el] = tmp_curr_std
                    new_est_edge_tk_f[el] = tmp_curr_f
                
            elif np.isnan(tmp_curr_f) and (tk_result['frac_tracked_detection'][el] > min_trust_f): 
                print(f"Edge {el}: curr {tmp_curr_v:.1f} +/- {tmp_curr_std:.1f} vs new {tmp_tk_v:.1f} +/- {tmp_tk_v_std:.1f}. Has {tk_result['frac_tracked_detection'][el]:.2f} fraction of passing particle tracked. Trust the estimation from tracking")
                new_est_edge_v[el] = tmp_tk_v
                new_est_edge_v_std[el] = tmp_tk_v_std
                new_est_edge_tk_f[el] = tmp_tk_f
                info['e_t_tk'].append(el)
            elif to_exit_Q: 
                # Exit edge. TODO: a better way to estimate the 'tracked' fraction. 
                new_est_edge_v[el] = tmp_tk_v
                new_est_edge_v_std[el] = tmp_tk_v_std
                new_est_edge_tk_f[el] = tmp_tk_f
                info['e_exit'].append(el)
            else:
                print(f"Edge {el}: {tmp_curr_v:.1f} +/- {tmp_curr_std:.1f}, {tmp_tk_v:.1f} +/- {tmp_tk_v_std:.1f} does not make sense")
                info['e_unresolved'].append(el)
                if allow_conflicting_Q: 
                    continue
                else:
                    raise NotImplementedError
        else: 
            info['e_missing'].append(el)

    info['v'] = new_est_edge_v
    info['std'] = new_est_edge_v_std
    info['track_frac'] = new_est_edge_tk_f
    return info

def merge_detection_table(old_detection, new_detection): 
    assert np.all(new_detection.did.values[1:] > new_detection.did.values[:-1]), 'Particle detection id is not sorted'
    p_pid_to_idx = util.get_table_value_to_idx_dict(new_detection, key='particle')
    tmp_did = new_detection['did'].values
    p_did_l = [tmp_did[v] for k, v in p_pid_to_idx.items() if v.size >= 2]
    m_pid = Linking.update_with_detection_id(old_detection, p_did_l, inplace_Q=True)
    p_idx_l = np.concatenate([v for v in p_pid_to_idx.values() if v.size >= 2])
    for k in ['cost', 'num_cands', 'v']: 
        old_detection[k] = np.nan
        tmp_col = old_detection.columns.get_loc(k)
        old_detection.iloc[tmp_did[p_idx_l], tmp_col] = new_detection[k].values[p_idx_l]
    
    return old_detection

#endregion

#region Linking class
class Linking: 
    def __init__(self, frame_detection:pd.DataFrame, vol_shape, force_continuous_did_Q=True):
        self.data = frame_detection.copy()
        if force_continuous_did_Q: 
            assert np.all(np.diff(self.data.did.values) == 1), 'Detection ID in frame_detection is not continuous'
        self.num_frame = self.data.frame.max() + 1
        self.data['activeQ'] = True
        # self.p_detect_id = defaultdict(list) # particle detection ID
        self.vol_shape = vol_shape
        self._edge_idx = None
        self._node_idx = None
        self.construct_kdtrees()
        if 'edge_label' in self.data:
            self.edge_ind, self.edge_label = self.group_detection_by_key('edge_label', sorted=0)

    def construct_kdtrees(self, selectedQ=None):
        if selectedQ is None: 
            data = self.data
        else: 
            data = self.data[selectedQ]
        # About 4 seconds for 1200+ frames. Not that bad. 
        t_list = util.split_table_by_key(data, 'frame')
        self.tree = []
        self._tree_idx_to_table_idx = []
        self._tree_idx_to_did = []
        for t in t_list:
            if isinstance(t, list) and len(t) == 0: 
                p = np.empty((0, 3), dtype=np.float32)
                self._tree_idx_to_table_idx.append(np.empty((0,), dtype=np.int32))
                self._tree_idx_to_did.append(np.empty((0,), dtype=np.int32))
            else:
                p = t[['z', 'y', 'x']].to_numpy()
                self._tree_idx_to_table_idx.append(t.index.values)
                self._tree_idx_to_did.append(t.did.values)
    
            t_tree = KDTree(p)
            self.tree.append(t_tree)
            
    def get_neighbors(self, t:int, zyx, k, max_dist=np.inf):
        """
            Output: 
                nb_dist: distance to the k nearest neighbors 
                nb_idx: indices of the knns in the table of each frame (start from 0, not the pandas indices)
                table_idx: detection id of the knns
        """
        nb_dist, nb_idx = self.tree[t].query(zyx, k=k, distance_upper_bound=max_dist)
        table_idx = np.full(nb_dist.shape, int(-1))
        is_valid_dist_Q = np.isfinite(nb_dist)
        table_idx[is_valid_dist_Q] = self._tree_idx_to_table_idx[t][nb_idx[is_valid_dist_Q]]
        return nb_dist, nb_idx, table_idx
    
    def get_all_neighbors_in_dist(self, t:int, zyx, max_dist):
        nb_idx = self.tree[t].query_ball_point(zyx, max_dist)
        nb_idx = np.asarray(nb_idx)
        table_idx = np.full(nb_idx.shape, int(-1))
        nb_dist = np.full(nb_idx.shape, np.nan)
        if nb_idx.size > 0: 
            nb_pts_pos = self.tree[t].data[nb_idx]
            nb_dist = np.sqrt(np.sum((zyx[None, :] - nb_pts_pos) ** 2, axis=1))
            dist_s_idx = np.argsort(nb_dist)
            nb_idx = nb_idx[dist_s_idx]
            nb_dist = nb_dist[dist_s_idx]
            table_idx = self._tree_idx_to_table_idx[t][nb_idx]
        return nb_dist, nb_idx, table_idx


    def get_neighbors_distance(self, t:int, zyx, k, max_dist=np.inf):
        nb_dist, _ = self.tree[t].query(zyx, k=k, distance_upper_bound=max_dist)
        return nb_dist
    
    def group_detection_by_key(self, key, sorted=-1):
        idx, key_val = util.bin_data_to_idx_list(self.data[[key]].to_numpy().flatten())
        if 'label' in key:
            is_valid_Q = (key_val >= 0)
            idx = idx[is_valid_Q]
            key_val = key_val[is_valid_Q]
        if sorted != 0: 
            num_record_in_edge = np.asarray(list(map(lambda x: x.size, idx)))
            s_idx = np.argsort(num_record_in_edge)[::sorted]
            idx = idx[s_idx]
            key_val = key_val[s_idx]
        return idx, key_val
    
    def get_edge_data(self, edge_label):
        if self._edge_idx is None: 
            idx, e_label = self.group_detection_by_key('edge_label', sorted=0)
            self._edge_idx = np.empty(np.max(e_label) + 1, object)
            for i, e in zip(idx, e_label):
                self._edge_idx[e] = i
        edge_label = np.atleast_1d(np.asarray(edge_label)) 
        edge_did = np.concatenate(self._edge_idx[edge_label])
        return self.data.iloc[edge_did]
    
    def get_node_data(self, node_label):
        if self._node_idx is None: 
            idx, n_label = self.group_detection_by_key('node_label', sorted=0)
            self._node_idx = np.empty(np.max(n_label) + 1, object)
            for i, e in zip(idx, n_label):
                self._node_idx[e] = i
        node_label = np.atleast_1d(np.asarray(node_label)) 
        node_did = np.concatenate(self._node_idx[node_label])
        return self.data.iloc[node_did]
    
    def update_with_particles(self, p_list:list, inplace_Q=False):
        if inplace_Q:
            current_pid = self.data.pid.values
        else: 
            current_pid = self.data.pid.values.copy()
        unlabeled_pid = int(-1)

        unique_pid = np.unique(current_pid)
        min_new_pid = np.max(unique_pid) + 1

        num_accepted_trace = 0
        for i, tmp_particle in enumerate(p_list):
            # Trajectory selection
            tmp_curr_pid = current_pid[tmp_particle.detections.did.values]        
            tmp_intersect = (tmp_curr_pid[1:-1] != unlabeled_pid)

            # Deal with pre-termination intersect 
            if np.any(tmp_intersect):
                # Need to further improving this part
                raise NotImplementedError
                # split the trace, exclude the endpoint, add the rest of the trace. 
                # tmp_particle.select_largest_sub_trajectory_by_masking(~tmp_intersect)
            
            tmp_d_id = tmp_particle.detections.did.values
            tmp_curr_pid = current_pid[tmp_d_id]                
            num_accepted_trace += 1
            if np.all(tmp_curr_pid[[0, -1]] == unlabeled_pid):
                # All unlabeled
                current_pid[tmp_d_id] = min_new_pid
                min_new_pid += 1
            else: 
                tmp_exist_pid = tmp_curr_pid[[0, -1]]
                tmp_exist_pid = np.asarray(tmp_exist_pid[tmp_exist_pid != -1])
                if tmp_exist_pid.size == 1: 
                    tmp_min_exist_pid = tmp_exist_pid
                else: 
                    assert tmp_exist_pid[0] != tmp_exist_pid[1], "New trace is connected to two points in one existing trace"
                    tmp_min_exist_pid = np.min(tmp_exist_pid)
                # merging existing particles 
                # If we keep track of the particles, this could be more efficient 
                for tepid in tmp_exist_pid:
                    current_pid[current_pid == tepid] = tmp_min_exist_pid
                # Add new particle
                current_pid[tmp_d_id] = tmp_min_exist_pid
        print(f"Number of accepted traces: {num_accepted_trace}")
        
        return current_pid

    @staticmethod
    def update_with_detection_id(detections, p_to_did, inplace_Q=False):
        if inplace_Q:
            current_pid = detections.pid.values
        else: 
            current_pid = detections.pid.values.copy()

        unlabeled_pid = int(-1)
        unique_pid = np.unique(current_pid)
        min_new_pid = np.max(unique_pid) + 1

        num_accepted_trace = 0
        for i, tmp_d_id in enumerate(p_to_did):
            # Trajectory selection
            tmp_curr_pid = current_pid[tmp_d_id]        
            tmp_intersect = (tmp_curr_pid[1:-1] != unlabeled_pid)

            # Deal with pre-termination intersect 
            if np.any(tmp_intersect):
                # Need to further improving this part
                raise NotImplementedError
                # split the trace, exclude the endpoint, add the rest of the trace. 
                # tmp_particle.select_largest_sub_trajectory_by_masking(~tmp_intersect)
            
            tmp_curr_pid = current_pid[tmp_d_id]                
            num_accepted_trace += 1
            if np.all(tmp_curr_pid[[0, -1]] == unlabeled_pid):
                # All unlabeled
                current_pid[tmp_d_id] = min_new_pid
                min_new_pid += 1
            else: 
                tmp_exist_pid = tmp_curr_pid[[0, -1]]
                tmp_exist_pid = np.asarray(tmp_exist_pid[tmp_exist_pid != -1])
                if tmp_exist_pid.size == 1: 
                    tmp_min_exist_pid = tmp_exist_pid
                else: 
                    assert tmp_exist_pid[0] != tmp_exist_pid[1], "New trace is connected to two points in one existing trace"
                    tmp_min_exist_pid = np.min(tmp_exist_pid)
                # merging existing particles 
                # If we keep track of the particles, this could be more efficient 
                for tepid in tmp_exist_pid:
                    current_pid[current_pid == tepid] = tmp_min_exist_pid
                # Add new particle
                current_pid[tmp_d_id] = tmp_min_exist_pid
        print(f"Number of accepted traces: {num_accepted_trace}")
        
        return current_pid

    def get_active_did(self, relabel_pid_Q=False, include_particle_endpoints_Q=True):
        """
        Get active detection id, which includes the detections that
        do not belong to any particles, and all the internal endpoints 
        in the particle trajectories (particles might have discontinuous 
        frame count as we allow gaps to exist in the trajectories). 

        """
        current_pid = self.data.pid.values
        p_id_did, p_id_v = util.bin_data_to_idx_list(current_pid)
        valid_pid_Q = (p_id_v >= 0)
        active_did = p_id_did[~valid_pid_Q]
        p_id_v = p_id_v[valid_pid_Q]
        p_id_did = p_id_did[valid_pid_Q]
        assert active_did.size == 1, "More than 1 invalid pid value"
        active_did = active_did[0]
        
        if include_particle_endpoints_Q: 
            max_frame_num = self.data.frame.max()
            particle_ep_did = []
            for p_id, tmp_did in enumerate(p_id_did):
                tmp_p = particle.Particle(self.data.iloc[tmp_did])
                if relabel_pid_Q:
                    current_pid[tmp_did] = p_id
                tmp_p_ep_did = tmp_p.get_trajectory_endpoint(output='did', excluded_frame=[0, max_frame_num])
                if tmp_p_ep_did.size > 0: 
                    particle_ep_did.append(tmp_p_ep_did)
            particle_ep_did = np.concatenate(particle_ep_did)
            active_did = np.concatenate((active_did, particle_ep_did))
        
        return np.sort(active_did)

#endregion

#region Util functions

def register_detections(frame_detection, disp_vec, vol_shape, inplace_Q=True, verboseQ=True, detection_vol_shape=None):
    disp_vec = np.asarray(disp_vec).astype(np.int16)
    if not inplace_Q:
        data_t_list = []

    for i in range(len(frame_detection)):
        if inplace_Q: 
            tmp_df = frame_detection[i]
        else: 
            tmp_df = frame_detection[i].copy()
        # Does not need to update the z coordinate
        tmp_df.z += disp_vec[0]
        tmp_df.y += disp_vec[1]
        tmp_df.x += disp_vec[2]
        if 'sub_0' not in tmp_df: 
            sub_0 = np.round(tmp_df.z).astype(np.int32)
            sub_1 = np.round(tmp_df.y).astype(np.int32)
            sub_2 = np.round(tmp_df.x).astype(np.int32)
        else: 
            tmp_df.sub_0 += disp_vec[0]
            tmp_df.sub_1 += disp_vec[1]
            tmp_df.sub_2 += disp_vec[2]

            sub_0 = tmp_df.sub_0.values
            sub_1 = tmp_df.sub_1.values
            sub_2 = tmp_df.sub_2.values

        is_valid_Q = np.logical_and(np.logical_and(sub_1 >= 0, sub_1 < vol_shape[1]), 
                                    np.logical_and(sub_2 >= 0, sub_2 < vol_shape[2]))
        is_valid_Q = np.logical_and(is_valid_Q, 
                                    np.logical_and(sub_0 >= 0, sub_0 < vol_shape[0]))
        
        if not np.all(is_valid_Q):
            tmp_df = tmp_df[is_valid_Q]
            sub_0 = sub_0[is_valid_Q]
            sub_1 = sub_1[is_valid_Q]
            sub_2 = sub_2[is_valid_Q]
            if verboseQ:
                print(f"Remove {np.count_nonzero(~is_valid_Q)} detections in frame {i}")
        
        # Ind should always be updated 
        tmp_df.loc[:, 'ind'] = np.ravel_multi_index((sub_0, sub_1, sub_2), vol_shape)
        
        # update exit endpoint 
        transform_ind_list = []
        for n in ['skl_ind', 'exit_ind']: 
            if n in tmp_df: 
                transform_ind_list.append(n)
        if len(transform_ind_list): 
            if detection_vol_shape is None: 
                print(f"detection_vol_shape is not provided. Assume it to be the same as vol_shape")
                detection_vol_shape = vol_shape
            
            for ind_name in transform_ind_list: 
                tmp_ind = tmp_df[ind_name].values
                is_valid_Q = (tmp_ind >= 0)
                valid_ind = tmp_ind[is_valid_Q].astype(np.int64)
                valid_sub = np.vstack(np.unravel_index(valid_ind, detection_vol_shape)) 
                valid_sub += disp_vec[:, None]
                t_ind = np.ravel_multi_index([valid_sub[i] for i in range(3)], vol_shape).astype(np.int64)
                tmp_df[ind_name].values[is_valid_Q] = t_ind

        if not inplace_Q:
            data_t_list.append(tmp_df)
        else: 
            # Do I need this? 
            frame_detection[i] = tmp_df

    if inplace_Q: 
        return frame_detection
    else: 
        return data_t_list

def parse_detection_data(data:dict, vol_shape=None):
    df, info = LFBFP.Detection.convert_data_to_pandas_dataframe(data, mask_size=vol_shape)
    df = df.reset_index(drop=True)
    df['pid'] = int(-1)
    return df, info

def get_particle_trajectory_from_tracking(detect_pid, min_length=0, sortedQ=1, return_dict_Q=False):
    p_idx, p_id = util.bin_data_to_idx_list(detect_pid)
    track_length = np.asarray(list(map(lambda x: x.size, p_idx)))
    if min_length: 
        long_track_idx = np.nonzero(track_length >= min_length)[0]
        track_length = track_length[long_track_idx]
        p_id = p_id[long_track_idx]
    else: 
        long_track_idx = np.arange(track_length.size)
    print(f"Found {long_track_idx.size} trajectories of length at least {min_length}")
    
    if return_dict_Q: 
        p_id_to_table_idx = {p: i for p, i in zip(p_id, p_idx[long_track_idx])}
        return p_id_to_table_idx
    else: 
        if sortedQ != 0:
            assert (sortedQ == 1 or sortedQ == -1), NotImplementedError
            s_idx = np.argsort(track_length)[::sortedQ] # descending
            long_track_idx = long_track_idx[s_idx]
        
        return np.asarray(p_idx[long_track_idx])

def select_detections(detections, min_peak_snr=2, mask_dilate_r=2, max_extra_dist_to_skl=3):
    # Detection selection
    detections = pd.concat(data_t_list).drop(columns=['sub_0', 'sub_1', 'sub_2', 'eig3', 'bg_std', 'nb_mean'])
    detections = detections[detections.peak_nb_snr >= min_peak_snr]
    non_bg_mask = skim.dilation(vsl_mask, skim.ball(radius=mask_dilate_r))
    in_mask_Q = non_bg_mask.flat[detections.ind.values]
    detections = detections[in_mask_Q]

    dist_2_skl = fg.nearest_map.ind_to_nearest_dist(detections.ind.values)
    nearest_skl_r = skl_r_array.flat[detections.ind.values]
    near_skl_Q = (dist_2_skl <= nearest_skl_r + max_extra_dist_to_skl)
    detections = detections[near_skl_Q]

    detections = detections.reset_index(drop=True)
    # Add information
    detections['skl_ind'] = fg.nearest_map.ind_to_nearest_ind(detections.ind.values)
    detections['edge_label'] = fg.edge.ind_to_label(detections['skl_ind'].values)
    detections['node_label'] = fg.node.ind_to_label(detections['skl_ind'].values)
    # Define unique detection id 
    detections['did'] = np.arange(detections.shape[0], dtype=np.int64)

    print(f"Number of detections: {detections.shape[0]}")
    return detections


#endregion
class LinkFrame():
    def __init__(self):
        pass

    @staticmethod
    def track_near_stationary_particles(detect_table:pd.DataFrame, max_speed_pxl, min_duration, active_mask=None, verboseQ=False):
        """
        
        """
        if active_mask is not None:
            detect_table = detect_table[active_mask].sort_value(by='did')

        trace_result = pd.concat(tp.link_df_iter(util.split_table_by_key(detect_table, 'frame'), max_speed_pxl, pos_columns=['x', 'y', 'z'], 
                                            adaptive_stop=1, adaptive_step=0.9))
        long_traces_ind = LFBFP.Linking.select_long_trajectories_from_trackpy_result(trace_result, min_duration, sortedQ=-1)
        if verboseQ:
            print(f"Found {long_traces_ind.size} trajectories of length at least {min_duration}")
        return trace_result, long_traces_ind
    

    @staticmethod
    def update_detection_table(detections:pd.DataFrame, long_traces_ind, trace_result, 
               min_num_frame, min_cos_vv, max_stationary_disp, max_v_norm, add_staionary_Q=True, inplace_Q=True):
        if inplace_Q:
            current_pid = detections.pid.values # reference
        else: 
            current_pid = detections.pid.values.copy() 
        
        unique_pid = np.unique(current_pid)
        min_new_pid = np.max(unique_pid) + 1

        num_traces = long_traces_ind.size
        track_endpoint_d_id = np.full((num_traces, 2), int(-1))
        num_accepted_trace = 0
        for i, p_t_idx in enumerate(long_traces_ind):
            tmp_p_table = trace_result.iloc[p_t_idx]
            # Trajectory selection
            tmp_particle = particle.Particle(tmp_p_table)
            if not tmp_particle.is_stationary_Q(max_disp=max_stationary_disp, max_v_norm=max_v_norm):
                tmp_particle.select_sub_trajectory_by_adj_v_cos(min_cos_vv, inplace_update_Q=True, smooth_Q=False)
                tmp_particle.select_sub_trajectory_by_int_traces(inplace_update_Q=True)
            else: 
                if add_staionary_Q: 
                    # Remove the part of trajectory that moves awary from the stationary position
                    tmp_particle.select_sub_trajectory_by_dist_to_med_pos()
                else: 
                    continue
                
            tmp_d_id = tmp_particle.detections.did.values
            tmp_curr_pid = current_pid[tmp_d_id]        
            tmp_intersect = (tmp_curr_pid[1:-1] != -1)
            if np.any(tmp_intersect):
                # Need to further improving this part
                if np.count_nonzero(tmp_intersect) == 1: 
                    # split the trace, exclude the endpoint, add the rest of the trace. 
                    tmp_particle.select_largest_sub_trajectory_by_masking(~tmp_intersect)
                else: 
                    tmp_particle.select_largest_sub_trajectory_by_masking(~tmp_intersect)
                    # raise NotImplementedError
                
            if tmp_particle.num_frame >= min_num_frame:
                tmp_d_id = tmp_particle.detections.did.values
                tmp_curr_pid = current_pid[tmp_d_id]                
                num_accepted_trace += 1
                if np.all(tmp_curr_pid[[0, -1]] == -1):
                    # All unlabeled
                    current_pid[tmp_d_id] = min_new_pid
                    min_new_pid += 1
                else: 
                    tmp_exist_pid = tmp_curr_pid[[0, -1]]
                    tmp_exist_pid = np.asarray(tmp_exist_pid[tmp_exist_pid != -1])
                    if tmp_exist_pid.size == 1: 
                        tmp_min_exist_pid = tmp_exist_pid
                    else: 
                        assert tmp_exist_pid[0] != tmp_exist_pid[1], "New trace is connected to two points in one existing trace"
                        tmp_min_exist_pid = np.min(tmp_exist_pid)

                    for tepid in tmp_exist_pid:
                        # also need to check the new segment does not overlap with the existing ones 
                        current_pid[current_pid == tepid] = tmp_min_exist_pid
                    current_pid[tmp_d_id] = tmp_min_exist_pid
                
                if tmp_particle.detections.frame.iloc[0] != 0: 
                    track_endpoint_d_id[i, 0] = tmp_d_id[0]
                if tmp_particle.detections.frame.iloc[-1] != (current_pid.size - 1):
                    track_endpoint_d_id[i, 1] = tmp_d_id[-1]
        print(f"Number of accepted traces: {num_accepted_trace}")
        # remaining detections: 
        need_tracking_Q = np.zeros(current_pid.shape, bool)
        need_tracking_Q[np.asarray(track_endpoint_d_id[track_endpoint_d_id != -1])] = True
        need_tracking_Q = np.logical_or(need_tracking_Q, (detections.pid == -1))
        if inplace_Q:
            return detections, need_tracking_Q
        else: 
            result = detections.copy()
            result.pid = current_pid
            return result, need_tracking_Q
