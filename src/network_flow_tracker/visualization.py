import os
from collections import defaultdict

import pandas as pd 
import numpy as np
import scipy as sp 
from matplotlib import pyplot as plt
import matplotlib as mpl

from .utils import neighbors as nb
from .utils import stat, vis, util, io

def vis_single_edge_stat_traces(edge_trace_stat):
    """
    Input: 
        tmp_stat: output of test_ef.analyze_edge_spatiotemporal_dynamics(mv_avg_wd, mv_min_num, **std_para)

    """
    vis_t = edge_trace_stat['t']
    test_el = edge_trace_stat['label']
    mv_avg_wd_t = edge_trace_stat['mv_avg_wd'] / edge_trace_stat['frame_rate_Hz']

    f = plt.figure(figsize=(8, 6))
    a1 = f.add_subplot(2, 1, 1)
    a1.errorbar(np.arange(edge_trace_stat['skl_v_mean'].size), edge_trace_stat['skl_v_mean'], edge_trace_stat['skl_v_std'], color='b')
    a1.set_xlabel('Skl vxl idx')
    a1.set_ylabel("Velocity (um/s)", color='b')
    a1.set_title(f"Edge {test_el} Voxel average speed")
    a1.tick_params(axis='y', labelcolor='b')
    a1.grid()
    a11 = a1.twinx()
    v_cv = np.abs(edge_trace_stat['skl_v_std'] / edge_trace_stat['skl_v_mean'])
    a11.plot(np.arange(edge_trace_stat['skl_v_mean'].size), v_cv, color='g')
    # a11.set_title(f"CV {v_cv.mean():.2f} +/- {v_cv.std():.2f}")
    a11.set_ylabel(f"CV", color='g')
    a11.tick_params(axis='y', labelcolor='g')

    a2 = f.add_subplot(2, 1, 2)
    a2.scatter(vis_t, np.abs(edge_trace_stat['p_t_v_mean']), 2, color='r', alpha=0.5, label='v')
    a2.plot(vis_t, np.abs(edge_trace_stat['p_t_v_mean_sm']), alpha=0.75, linewidth=2, color='b', label='mov. avg. v')
    a2.legend()
    a2.set_xlabel('t (s)')
    a2.set_ylabel("Speed (um/s)", color='b')
    a2.tick_params(axis='y', labelcolor='b')
    a2.grid()
    a2.set_title(f"Edge {test_el} v and cell count (mw {mv_avg_wd_t:.2f} s)")

    a22 = a2.twinx()
    a22.plot(vis_t, edge_trace_stat['e_t_cell_count'], color='g', alpha=0.5, linewidth=2)
    a22.set_ylabel('# RBC', color='g')
    a22.tick_params(axis='y', labelcolor='g')
    
    f.tight_layout()
    return f

def vis_vxl_path_v(path_vxl_ind, fg, voxel_size_um=1, mm2s_to_pxl2s=1, merge_node_Q=False): 
    # Compute skeleton voxel speed mean and std
    tmp_vmap = [fg.vxl_speed_map[fg.ind_to_vxl_idx(i)] for i in path_vxl_ind]
    tmp_vmap = np.vstack(tmp_vmap)
    tmp_s_map = np.abs(tmp_vmap / mm2s_to_pxl2s)
    path_v_avg = np.nanmean(tmp_s_map, axis=1)
    path_v_std = np.nanstd(tmp_s_map, axis=1)

    tmp_is_node_Q = (fg.node.ind_to_label(path_vxl_ind) >= 0)
    tmp_is_node_int = util.get_intervals_in_1d_binary_array(tmp_is_node_Q, including_end_Q=True)
    # Shall we merge the node cc here? 
    if merge_node_Q: 
        for i, tmp_int in enumerate(tmp_is_node_int):
            tmp_int_ind = path_vxl_ind[tmp_int[0] : tmp_int[1] + 1]
            if tmp_int_ind.size > 1: 
                tmp_com_ind = fg.node.compute_cc_com_ind(tmp_int_ind, fg.num['mask_size']) 
                path_vxl_ind[tmp_int[0] : tmp_int[1] + 1] = tmp_com_ind
        _, tmp_idx = np.unique(path_vxl_ind, return_index=True)
        tmp_idx.sort()
        path_vxl_ind = path_vxl_ind[tmp_idx]
        path_v_avg = path_v_avg[tmp_idx]
        path_v_std = path_v_std[tmp_idx]

        tmp_is_node_Q = (fg.edge.ind_to_label(path_vxl_ind) < 0)
        tmp_is_node_int = util.get_intervals_in_1d_binary_array(tmp_is_node_Q, including_end_Q=True)
        
    tmp_x = np.concatenate(([0], np.cumsum(fg.edge.compute_adj_ind_dist(path_vxl_ind, fg.edge.space_shape)))) * voxel_size_um

    f = plt.figure(figsize=(20, 4))
    a = f.add_subplot()
    a.errorbar(tmp_x, path_v_avg, path_v_std)
    a.plot(tmp_x, path_v_avg)
    a.set_xlabel("Distance (um)")
    a.set_ylabel("Speed (mm/s)")
    for tmp_int in tmp_is_node_int:
        a.axvspan(tmp_x[tmp_int[0]], tmp_x[tmp_int[1]], color='g', alpha=0.50)
    a.set_title(f"vxl path; length {tmp_x[-1]:.2f} um")
    a.grid(True, axis='y', alpha=0.5)
    return f, path_vxl_ind

def vis_vxl_path_v_between_two_edges(source_edge, target_edge, fg, voxel_size_um=1, mm2s_to_pxl2s=1, merge_node_Q=False): 
    source_ind = fg.edge.cc_ind[source_edge][0]if fg.edge_v_pxl[source_edge] > 0 else fg.edge.cc_ind[source_edge][-1]
    target_ind = fg.edge.cc_ind[target_edge][-1] if fg.edge_v_pxl[target_edge] >0 else fg.edge.cc_ind[target_edge][0]
    tmp_path_node, tmp_path_len = fg.abs_g.compute_shortest_path_between_two_voxel_indices(source_ind, target_ind)
    path_vxl_ind, tmp_dir = fg.abs_g.get_voxel_path_from_node_path(tmp_path_node, source_ind, target_ind, include_node_Q=True)

    f, edge_path = vis_vxl_path_v(path_vxl_ind, fg, voxel_size_um=voxel_size_um, mm2s_to_pxl2s=mm2s_to_pxl2s, merge_node_Q=merge_node_Q)
    return f, edge_path


def vis_separation_with_edge_traces(p_el, c_el, ds_e_count, edges_traces, 
                                    v_name='e_t_v_mean_sm', rho_name='e_t_cell_density_sm', flux_name='e_t_flux_smp', 
                                    p_std_z=1, ns_z=1.96):
    p_el = int(p_el)
    # f = plt.figure(figsize=(10, 8))
    f, axs = plt.subplots(nrows=5, sharex=True, figsize=(10, 8))
    # num_sp = 5
    # axs[0] = f.add_subplot(num_sp, 1, 1)
    # axs[1] = f.add_subplot(num_sp, 1, 2)
    # axs[2] = f.add_subplot(num_sp, 1, 3)
    # axs[3] = f.add_subplot(num_sp, 1, 4)
    # axs[4] = f.add_subplot(num_sp, 1, 5)
    vis_t = edges_traces[p_el]['t']
    for i, c in enumerate(c_el):
        axs[0].fill_between(vis_t, ds_e_count[c]['p_ci_l'], ds_e_count[c]['p_ci_h'], alpha=0.25)
        # axs[0].fill_between(vis_t, ds_e_count[c]['p'] - p_std_z * ds_e_count[c]['p_se'], ds_e_count[c]['p'] + p_std_z * ds_e_count[c]['p_se'], alpha=0.25)
        axs[0].plot(vis_t, ds_e_count[c]['p'], label=f'CE{c}', alpha=0.75)
        axs[1].plot(vis_t, ds_e_count[c]['count'], label=f'CE{c}')

        axs[2].plot(vis_t, np.abs(edges_traces[c][v_name]), label=f'CE{c}', alpha=0.75)
        axs[3].plot(vis_t, np.abs(edges_traces[c][rho_name]), label=f'CE{c}', alpha=0.75)
        axs[4].plot(vis_t, np.abs(edges_traces[c][flux_name]), label=f'CE{c}', alpha=0.75)

    axs[2].plot(vis_t, np.abs(edges_traces[p_el][v_name]), label=f'PE{p_el}', alpha=0.75)
    axs[3].plot(vis_t, np.abs(edges_traces[p_el][rho_name]), label=f'PE{p_el}', alpha=0.75)
    axs[4].plot(vis_t, np.abs(edges_traces[p_el][flux_name]), label=f'PE{p_el}', alpha=0.75)
    axs[0].legend()
    axs[2].legend()
    # Deviation from stationary
    dfs_trace = ds_e_count[c_el[0]]
    p_avg = dfs_trace['avg_p']
    # p_se = dfs_trace['p_to_avg_p_se']
    # axs[0].fill_between(vis_t, np.maximum(0, p_avg - ns_z * p_se), np.minimum(1, p_avg + ns_z * p_se), alpha=0.25)
    axs[0].axhline(p_avg, xmin=vis_t[0], xmax=vis_t[-1], linestyle='--', alpha=0.75)
    axs[0].set_ylim(-0.05, 1.05)
    axs[1].plot(vis_t, dfs_trace['parent_count'], label=f'PE{p_el}')
    # tmp_z_mask = (dfs_trace['p_2_avg_p_z'] > ns_z)
    tmp_z_mask = np.logical_or(dfs_trace['p_ci_l'] > p_avg, dfs_trace['p_ci_h'] < p_avg)
    if np.any(tmp_z_mask): 
        tmp_z_gt_2_ints = util.get_intervals_in_1d_binary_array(tmp_z_mask)
        tmp_z_gt_2_len = tmp_z_gt_2_ints[:, 1] - tmp_z_gt_2_ints[:, 0]
        tmp_long_Q = tmp_z_gt_2_len >= 7
        tmp_z_gt_2_ints = tmp_z_gt_2_ints[tmp_long_Q, :]
        for tmp_int in tmp_z_gt_2_ints:
            axs[0].axvspan(vis_t[tmp_int[0]], vis_t[tmp_int[1] - 1], color='r', alpha=0.25)
    
    axs[0].set_ylabel("Fraction")
    axs[0].set_title(f"wd. sz. {ds_e_count[c]['mv_wd_sz']}")
    axs[1].set_ylabel("# cell in mv. wd.")
    axs[2].set_ylabel("Speed (um/s)")
    axs[3].set_ylabel("Linear density (/um)")
    axs[4].set_ylabel("RBC Flux (/s)")
    
    axs[4].set_xlabel("Time (s)")
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[3].grid()
    axs[4].grid()
    return f


def vis_edge_info(edge_label, trace_result, el_to_idx, fg, vis_dm_corr_v_Q=False, vis_mips_Q=False, vis_dm_Q=False, vis_vxl_v_map_Q=False):
    if edge_label in el_to_idx: 
        tmp_edge_table = trace_result.iloc[el_to_idx[edge_label]]
        test_ef = fg.get_edgeflow_object(edge_label, detection_in_edge=tmp_edge_table)
        print(f"Length {test_ef.length:.2f}\nep2ep vec {np.round(test_ef.vec_ep2ep, decimals=2)}\nconnected node: {fg.edge.connected_node_label[edge_label]}")
        # Estimate velocity based on dealyed correlation 
        # If the segment is long, the delay could be larger. 
        dm_info = test_ef.iterative_est_avg_velocity_from_detection_map(test_ef.detect_map, vis_Q=vis_dm_corr_v_Q)
        if vis_mips_Q: 
            test_ef.vis_w_3_mip(vsl_mips, figsize=(10, 10))
        if vis_dm_Q and vis_vxl_v_map_Q: 
            test_ef.vis_detection_and_vxl_v_map(frame_range=None)
        elif vis_dm_Q: 
            test_ef.vis_detection_map()
        
        return test_ef
    else: 
        print(f"Edge {edge_label} does not have any detection.")


def compute_skeleton_speed_mips(fg, mm2s_to_pxl2s=1):
    vxl_avg_speed = np.abs(np.nanmedian(fg.vxl_speed_map, axis=1)) # average over time
    valid_speed_Q = np.isfinite(vxl_avg_speed)
    mask_size = fg.num['mask_size']
    vis_skl_flow = np.zeros(mask_size, dtype=np.float32)
    vis_skl_flow.flat[fg.pos_ind[valid_speed_Q]] = vxl_avg_speed[valid_speed_Q] / mm2s_to_pxl2s
    vis_skl_flow_mips = vis.compute_three_view_mip(vis_skl_flow)
    return vis_skl_flow_mips


def vis_branch_acc_vs_flow_speed(edge_v_mm_s, trk_stat, v_bin, max_speed_pxl, mm2s_to_pxl2s, vis_val='accuracy', vis_stat='ptrl'):
    acc_bin = np.linspace(0, 1, 21)
    v_bin_val = (v_bin[1:] + v_bin[:-1]) / 2

    f, a = plt.subplots(1, 1, figsize=(4, 3))
    h = a.hist2d(edge_v_mm_s, trk_stat['branch'][vis_val], bins=[v_bin, acc_bin], norm=mpl.colors.LogNorm())
    
    vis_eb_x = v_bin_val
    vis_eb_y = trk_stat['branch'][f'{vis_val}_{vis_stat}']
    vis_valid_Q = np.all(~np.isnan(vis_eb_y), axis=1)
    vis_eb_x = vis_eb_x[vis_valid_Q]
    vis_eb_y = vis_eb_y[vis_valid_Q]
    
    a.errorbar(vis_eb_x, vis_eb_y[:, 1], 
                    yerr=[vis_eb_y[:, 1] - vis_eb_y[:, 0], 
                          vis_eb_y[:, 2] - vis_eb_y[:, 1]], 
                    fmt='-', color='r')
    
    a.vlines(max_speed_pxl / mm2s_to_pxl2s, -0.05, 1.05, colors='k', linestyles='dashed')
    f.colorbar(h[-1], ax=a, label='# Branches')
    a.set_xlabel('Flow speed (mm/s)')
    a.set_ylabel(f'Branch {vis_val}')
    a.set_ylim(-0.05, 1.05)
    a.grid()
    a.set_title(f"Overall {vis_val}: {trk_stat['stat'][vis_val]:.3f}")
    return f, a