
import logging, warnings
import inspect
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import graph
from .utils import util 
from .utils import vis   
from .utils import stat

class EdgeFlow():
    def __init__(self, el, cc_ind, mask_size, connected_node_labels, cc_pos, eff_len, 
                 detections=None, duration=None, vxl_speed_map=None, cell_passenger=None):
        """
        Initialize the EdgeFlow object.
        Inputs: 
            el (int): The label of the edge.
            cc_ind (list): The indices of the connected component.
            mask_size (tuple): The size of the mask.
            connected_node_labels (np.ndarray): A 1D numpy array of shape (N,), where
              N is the number of voxels in the connected component.
            cc_pos (np.ndarray): A 2D numpy array of shape (3, N), where
              N is the number of voxels in the connected component.
            eff_len (scalar): effective length of an edge - do I need this?
            detections (pd.DataFrame): A pandas DataFrame containing detection information.
            duration (scalar): duration of the edge flow.
            vxl_speed_map (np.ndarray): A 2D numpy array of shape (N, T), where
              N is the number of voxels and T is the number of time frames.
            

        """
        self.label = el
        self.ind = cc_ind
        self.mask_size = mask_size
        self._construct_ind_to_ep_dist_dict()
        self.pos = cc_pos # (3, N)
        self.length_w_node = eff_len        
        self.connected_node_label = connected_node_labels
        self.detections = detections
        self.duration = duration
        self.vxl_speed_map = vxl_speed_map
        self.cell_passenger = cell_passenger
        self.parse_detection(num_frame=self.duration)
        # self._check_input()

    @property
    def vec_ep2ep(self):
        return (self.pos[:, -1] - self.pos[:, 0]).flatten()
    
    @property
    def itk_xyz_mid(self):
        mid_idx = self.pos.shape[1] // 2
        return self.pos[:, mid_idx][[1, 2, 0]].flatten()
    
    @property
    def num_endpoints(self):
        return np.sum(self.connected_node_label == -1)

    def parse_detection(self, num_frame=None):
        if self.detections is not None:
            # Do not modify the table. If needed, use property
            # self.detections['d_l_idx'] = self.ind_to_ep_dist(self.detections.skl_ind.values)[:, 0].astype(np.int16)
            d_in_f, frame = util.bin_data_to_idx_list(self.detections.frame.values)
            if num_frame is None: 
                num_frame = np.max(frame) + 1 # start from 0
            self.frame_ind = np.empty(num_frame, dtype=object)
            for i in range(num_frame):
                self.frame_ind[i] = np.array([])
            self.frame_ind[frame] = d_in_f
            # for f, ind in zip(frame, d_in_f): 
            #     self.frame_ind[f] = ind
            self.frame_num_detect = np.zeros(self.frame_ind.shape, dtype=np.uint32)
            self.frame_num_detect[frame] = np.asarray(list(map(lambda x:x.size, d_in_f)))

            self.detect_map = self.construct_detection_map()
            self.vxl_detection_map = self.construct_voxel_detection_map()

    def _construct_ind_to_ep_dist_dict(self):
        ep_dist = graph.GraphEdge.compute_dist_to_endpoints(self.ind, self.mask_size)
        # ep_dist[:, 0] *= -1 # backward 
        self.length = ep_dist[-1, 0]
        self._ind_to_ep_dist = {k : v for k, v in zip(self.ind, ep_dist)}
        self._ind_to_cc_idx = {i : idx for idx, i in enumerate(self.ind)}
    
    def _check_input(self):
        # Found one node with 10 voxels in its cc -> distance between the 
        # node and the first voxel in the segment was 4.2 pxl...... 
        vxl_pdist = np.sqrt(np.sum(np.diff(self.pos, axis=1) ** 2, axis=0))
        assert np.all(vxl_pdist < 5), "voxel pos spacing greater than expected"

    def ind_to_ep_dist(self, vxl_ind):
        if isinstance(vxl_ind, (int, np.integer)):
            return self._ind_to_ep_dist[vxl_ind]
        else: 
            vxl_ind = np.asarray(vxl_ind)
            if vxl_ind.size > 0: 
                return np.vstack([self._ind_to_ep_dist[i] for i in vxl_ind])
            else: 
                return np.zeros((0, 2))
    
    def ind_to_cc_ind_idx(self, vxl_ind):
        if isinstance(vxl_ind, (int, np.integer)):
            return self._ind_to_cc_idx[vxl_ind]
        else: 
            vxl_ind = np.asarray(vxl_ind)
            if vxl_ind.size > 0: 
                return np.vstack([self._ind_to_cc_idx[i] for i in vxl_ind])
            else: 
                return np.zeros((0, 1))


    def get_frame_detection(self, frame):
        idx = np.concatenate((self.frame_ind[frame])) 
        return self.detections.iloc[idx]

    def construct_voxel_detection_map(self):
        vxl_detection_map = np.zeros((self.frame_ind.size, self.ind.size))
        for i, idx in enumerate(self.frame_ind):
            if idx.size > 0: 
                i_skl_ind = self.ind_to_cc_ind_idx(self.detections.skl_ind.values[idx])[:, 0].astype(np.int16)
                # i_skl_ind = self.detections.d_l_idx.values[idx]
                vxl_detection_map[i, i_skl_ind] += 1
        return vxl_detection_map

    def construct_detection_map(self, bin_size=1):
        detection_map = np.zeros((self.frame_ind.size, int(np.ceil(self.length + 1))))
        for i, idx in enumerate(self.frame_ind):
            if idx.size > 0: 
                i_skl_ind = self.ind_to_ep_dist(self.detections.skl_ind.values[idx])[:, 0].astype(np.int16)
                # i_skl_ind = self.detections.d_l_idx.values[idx]
                detection_map[i, i_skl_ind] += 1
        
        if bin_size > 1: 
            detection_map, div_edge = EdgeFlow.bin_detection_map(detection_map, bin_size)
            return detection_map, div_edge
        else: 
            return detection_map
        
    def analyze_detection_map(self, high_count_frac):
        if self.detections is not None:             
            occupied_frac = np.mean(self.detect_map, axis=0)
            result = {}
            result['total_detection'] = np.nansum(self.detect_map)
            result['occupied_frac'] = occupied_frac
            result['num_hc_vxl'] = np.count_nonzero(occupied_frac > high_count_frac)
            result['avg_occ_frac'] = np.mean(occupied_frac)
            result['avg_occ_frac_nz'] = np.mean(occupied_frac[occupied_frac > 0])
            result['frac_hc_vxl'] = result['num_hc_vxl'] / occupied_frac.size

            # idx_0, idx_1 = util.find_ind_in_sub_array(occupied_frac > 0)
            # est_avg_v = EdgeFlow.estimate_avg_velocity_from_detection_map(self.detect_map, bin_size=2, 
            #                                                               dt=1, vis_Q=False)
            return result
        else: 
            print("Detection is not available.")
            return None
            
    @staticmethod    
    def bin_detection_map(detection_map, bin_size:int):
        num_div = np.floor(detection_map.shape[1] / bin_size).astype(np.uint32)
        div_edge = np.arange(0, (num_div + 1) * bin_size, bin_size).astype(np.uint32)
        if bin_size > 1:
            div_bin_count = np.full((num_div, detection_map.shape[0]), np.nan)
            for i in range(num_div):
                div_bin_count[i] = np.sum(detection_map[:, div_edge[i] : div_edge[i + 1]], axis=1).flatten()
            return div_bin_count.T, div_edge
        elif bin_size == 1: 
            return detection_map, div_edge
        else: 
            raise ValueError("bin_size should be an positive integer")
    
    @staticmethod
    def estimate_avg_velocity_from_detection_map(detection_map, min_corr=0, bin_size=2, dt=1, return_corr_Q=False, vis_Q=False): 
        assert dt >= 0, "dt must be non-negative"
        assert min_corr >= 0, "Only non-negative corr make sense"
        div_bin_count, div_edge = EdgeFlow.bin_detection_map(detection_map, bin_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = EdgeFlow.compute_shifted_pair_corr_coeff(div_bin_count, dt, min_corr=0, visQ=vis_Q)
            info = EdgeFlow.analyze_shifted_pair_corr_mat(corr, min_corr=min_corr, method='all')
            info['v_resolution'] = bin_size / dt
            info['avg_v'] = info['avg_diff_w'] * info['v_resolution']
            info['avg_v_std'] = np.abs(np.maximum(info['v_resolution'], info['avg_v'] * info['major_diff_cv']))
            info['avg_v_cv'] = info['avg_v_std'] / info['avg_v']
        info['max_estimatable_v'] = (corr.shape[0] - 1) * info['v_resolution']  # this is the extreme case, probably an over estimation
        info['bin_size'] = bin_size
        info['dt'] = dt
        if return_corr_Q: 
            info['corr'] = corr
        return info
    
    @staticmethod
    def iterative_est_avg_velocity_from_detection_map(detection_map, init_bin_size=2, init_dt=1, min_peak_corr=0.05, min_num_div=3,
                                                      max_num_div=15, re_est_min_tot_cor_r=3, max_dt=16, return_corr_Q=False, vis_Q=False):
        edge_len = detection_map.shape[1] - 1
        if edge_len >= min_num_div: 
            max_bin_size = np.floor(edge_len / min_num_div) 
            min_bin_size = np.round(edge_len / max_num_div)
            bin_size = np.maximum(min_bin_size, np.minimum(init_bin_size, max_bin_size)) 
            dt = init_dt
            dm_info = EdgeFlow.estimate_avg_velocity_from_detection_map(detection_map, min_corr=min_peak_corr, bin_size=bin_size,
                                                                        dt=dt, return_corr_Q=return_corr_Q, vis_Q=vis_Q)
            if vis_Q: 
                print(f"Average velocity {dm_info['avg_v']:.2f}, average corr {dm_info['avg_corr']:.2f}")

            while (np.abs(dm_info['avg_v']) < (bin_size / dt * 2)) and (dm_info['total_corr_ratio'] > re_est_min_tot_cor_r) and (dt <= max_dt): 
                # For slowly moving segments 
                dt *= 2
                if vis_Q:
                    print(f"Low speed edge. Re-estimate velocity with a delayed time step = {dt}")
                dm_info = EdgeFlow.estimate_avg_velocity_from_detection_map(detection_map, min_corr=min_peak_corr, bin_size=bin_size, 
                                                                            dt=dt, return_corr_Q=return_corr_Q, vis_Q=vis_Q)
                if vis_Q:
                    print(f"Average velocity {dm_info['avg_v']:.2f}, average corr {dm_info['avg_corr']:.2f}")

            while np.isnan(dm_info['avg_corr']) and (bin_size < max_bin_size): 
                if vis_Q:
                    print(f"Unable to estimate flow speed. Increase the bin size and try again")
                
                bin_size = np.minimum(bin_size * 2, max_bin_size)
                
                dm_info = EdgeFlow.estimate_avg_velocity_from_detection_map(detection_map, min_corr=min_peak_corr, bin_size=bin_size,
                                                                            dt=dt, return_corr_Q=return_corr_Q, vis_Q=vis_Q)
                
                if vis_Q:
                    print(f"Average velocity {dm_info['avg_v']:.2f}, average corr {dm_info['avg_corr']:.2f}")
                    
            dm_info['validQ'] = True if not np.isnan(dm_info['avg_v']) else False
        else: 
            dm_info = {'validQ': False}
            if vis_Q: 
                print(f"The edge is too short (len = {edge_len}) for the following analysis, which requries a minimal length of {min_num_div}. ")
        
        return dm_info
    
    @staticmethod
    def analyze_concatenated_detection_map(dm_1, dm_2, min_corr=0.05, min_major_corr=0.25, min_tot_corr_r=2, 
                                           diag_nb_r=1, vis_Q=False): 
        """
        
        Assume dm_1 (edge 1) is upstream of dm_2 (edge_2)

        
        """
        joint_dm = np.hstack([dm_1, dm_2])
        joint_d_result = EdgeFlow.iterative_est_avg_velocity_from_detection_map(joint_dm, min_peak_corr=min_corr, 
                                                                                return_corr_Q=True, vis_Q=vis_Q) 
        joint_d_result['edge_l_1'] = dm_1.shape[1]
        joint_d_result['edge_l_2'] = dm_2.shape[1]       
        num_col_1 = int(np.round(dm_1.shape[1] / joint_d_result['bin_size']))
        corr_mat = joint_d_result['corr']
        del joint_d_result['corr']

        # Determine if the result is reliable or not 
        if (joint_d_result['avg_v'] > 0) and \
            (joint_d_result['major_total_corr'] > min_major_corr) and \
            (joint_d_result['total_corr_ratio'] > min_tot_corr_r): 

            # If yes, presumably the high correction terms are near the upper right corner. 
            # Analyze the correlation matrix after zeroing the terms for the second edge only
            upper_corr_mat = corr_mat.copy()
            upper_corr_mat[num_col_1:, num_col_1:] = 0
            # Analyze diagonal terms 
            corr_mat_da = stat.analyze_matrix_diagonals(corr_mat)
            upper_mat_da = stat.analyze_matrix_diagonals(upper_corr_mat)
            # Get the diagonal terms near the estiamted peak 
            avg_v_offset = int(np.round(joint_d_result['avg_diff_w']))
            selected_Q = np.logical_and(corr_mat_da['offset'] >= np.maximum(1, avg_v_offset - diag_nb_r), 
                                        corr_mat_da['offset'] <= (avg_v_offset + diag_nb_r)) 
            joint_d_result['corr_diag_sum'] = np.sum(corr_mat_da['sum'][selected_Q])
            joint_d_result['upper_mat_diag_sum'] = np.sum(upper_mat_da['sum'][selected_Q])
            # Normalized by length ratio
            joint_d_result['diag_corr_ratio'] = joint_d_result['upper_mat_diag_sum'] / joint_d_result['corr_diag_sum'] \
                * (joint_dm.shape[1] / dm_1.shape[1])
            
            upper_dm_result = EdgeFlow.analyze_shifted_pair_corr_mat(upper_corr_mat, min_corr=min_corr)

            if (upper_dm_result['avg_diff_w'] > 0) and \
                (upper_dm_result['major_total_corr'] > min_major_corr) and \
                (upper_dm_result['total_corr_ratio'] > min_tot_corr_r): 
                joint_d_result.update(upper_dm_result)
                joint_d_result['updated_to_1_Q'] = True
                joint_d_result['avg_v'] = joint_d_result['avg_diff_w'] * joint_d_result['v_resolution']
                joint_d_result['avg_v_std'] = joint_d_result['avg_v'] * joint_d_result['major_diff_cv']
                joint_d_result['avg_v_to_max_v_1'] = joint_d_result['avg_v'] / joint_d_result['edge_l_1']
            else: 
                joint_d_result['avg_v_to_max_v_1'] = np.nan
                joint_d_result['updated_to_1_Q'] = False
        else: 
            joint_d_result['avg_v'] = np.nan
                
        
        return joint_d_result
    
    @staticmethod
    def compute_shifted_pair_corr_coeff(matTN, dt: int, min_corr=None, visQ=False): 
        """
        For each column in `matTN`, compute the pairwise shifted 
        correlation coefficient. 
        Args: 
            matTN: numpy array of shape (T, n), where T is the number 
            of time point and N is the number of traces (variables)
            dt: scalar, shift 
        Returns: 
            corr: the (i, j) element is the corr coeff of the i-th 
            column and the j-th column after shifrting the i-th column
            in the positive direction by `dt` (and shift the j-th column
            backward correspondingly). 
            col_max_idx: the row indices of the maximum in each column 
            col_max_corr: the value of the maximum in each column 
        """
        abs_dt = np.abs(dt).astype(np.int32)
        if abs_dt != 0: 
            X = matTN[:-abs_dt] # (T, n)
            Y = matTN[abs_dt:]
            X = X - X.mean(axis=0)
            Y = Y - Y.mean(axis=0)
            X /= np.sqrt(np.mean(X ** 2, axis=0))
            Y /= np.sqrt(np.mean(Y ** 2, axis=0))
                
            if dt < 0: 
                X, Y = Y, X # exchange order
        else: 
            X = matTN.copy()
            X -= X.mean(axis=0)
            X /= np.sqrt(np.mean(X ** 2, axis=0))
            Y = X

        corr = np.matmul(X.T, Y) / X.shape[0]
        corr[np.isnan(corr)] = 0 # manually fix the nan problem
        if min_corr is not None: 
            corr = np.maximum(corr, min_corr)

        if visQ: 
            f, a = EdgeFlow.vis_shifted_pair_corr_mat(corr, dt)

        return corr

    @staticmethod
    def analyze_shifted_pair_corr_mat(corr, min_corr, method='all'): 
        info = {'avg_corr': np.nan, 'avg_diff_w': np.nan, 'major_diff_cv': np.nan, 
                'total_corr_ratio': np.nan, 'major_total_corr': np.nan}

        if method == 'all':
            hc_r, hc_c = np.nonzero(corr > min_corr)
            hc_v = corr[hc_r, hc_c]
            # Select the top num_row values
            if hc_r.size > corr.shape[0]:
                v_idx = np.argsort(hc_v)[::-1][:corr.shape[0]]
                hc_r = hc_r[v_idx]
                hc_c = hc_c[v_idx]
                hc_v = hc_v[v_idx]

        elif method == 'row_peak':
            hc_c = np.argmax(corr, axis=1)
            hc_r = np.arange(hc_c.size)
            hc_v = corr[hc_r, hc_c]
            selected_Q = hc_v > min_corr
            hc_c = hc_c[selected_Q]
            hc_r = hc_r[selected_Q]
            hc_v = hc_v[selected_Q]
            
        # Diagonal analysis
        # tmp_corr_len = tmp_corr_upper.shape[0]
        # tmp_diag_offset = np.arange(-tmp_corr_len + 1, tmp_corr_len)
        # tmp_diag_mean = np.full(tmp_diag_offset.shape, np.nan)
        # tmp_diag_sum = np.full(tmp_diag_offset.shape, np.nan)

        # for i, j in enumerate(tmp_diag_offset): 
        #     tmp_diag_terms = np.diagonal(tmp_corr_upper, offset=j)
        #     tmp_diag_mean[i] = np.mean(tmp_diag_terms)
        #     tmp_diag_sum[i] = np.sum(tmp_diag_terms)
        
        if hc_r.size > 0: 
            hc_diff = hc_c - hc_r
            is_positive_Q = hc_diff > 0
            is_negative_Q = hc_diff < 0
            info['num_p'] = np.sum(is_positive_Q)
            info['num_n'] = np.sum(is_negative_Q)
            info['num_0'] = hc_diff.size - info['num_p'] - info['num_n']
            # corr weighted average difference
            avg_hc_diff = np.sum(hc_diff * hc_v) / np.sum(hc_v)

            if avg_hc_diff > 0: 
                major_sign_Q = is_positive_Q
                minor_sign_Q = is_negative_Q
            elif avg_hc_diff < 0: 
                major_sign_Q = is_negative_Q
                minor_sign_Q = is_positive_Q
            else: 
                major_sign_Q = np.ones(hc_diff.shape, bool)
                minor_sign_Q = ~ major_sign_Q
            # info['diff'] = hc_diff
            # info['corr'] = np.round(hc_v, decimals=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                info['avg_corr'] = np.mean(hc_v[major_sign_Q])
                info['avg_diff_w'] = np.sum(hc_diff[major_sign_Q] * hc_v[major_sign_Q]) / np.sum(hc_v[major_sign_Q])
                
                info['major_total_corr'] = np.sum(hc_v[major_sign_Q])
                info['major_avg_corr'] = np.mean(hc_v[major_sign_Q])
                info['major_diff_mean'] = np.mean(hc_diff[major_sign_Q])
                info['major_diff_cv'] = np.abs(np.std(hc_diff[major_sign_Q]) / info['major_diff_mean'])

                info['minor_diff_mean'] = np.mean(hc_diff[minor_sign_Q])
                info['minor_diff_cv'] = np.abs(np.std(hc_diff[minor_sign_Q]) / info['minor_diff_mean']) 
                info['minor_avg_corr'] = np.mean(hc_v[minor_sign_Q])
                info['minor_total_corr'] = np.sum(hc_v[minor_sign_Q])

                info['total_corr_ratio'] = info['major_total_corr'] / info['minor_total_corr']


        return info

    @staticmethod
    def analyze_voxel_velocity_map(tmp_e_skl_v_map, gm_max_num_est=3, min_data_size=5, min_gmc_dist_std_n=2, max_gmc_cv_to_merge=0.3, 
                                   v_type='map', vis_Q=False):


        tmp_e_skl_v_map = np.atleast_2d(tmp_e_skl_v_map)
        result = {'est_v': np.nan, 'est_v_std': np.nan, 'v_type': v_type,
                  'weight_ratio': np.nan, 'same_dir_Q': False, 'forced_1_comp_Q': False}
        result['num_tracked_frame'] = np.sum(np.any(np.isfinite(tmp_e_skl_v_map), axis=0)) 
        result['frac_tracked_frame'] = result['num_tracked_frame'] / tmp_e_skl_v_map.shape[1]
        # get unique velocity for each frame - approxiate the number of particles 
        frame_vsl_speed_map = tmp_e_skl_v_map.T
        particle_v = []
        for tmp_v in frame_vsl_speed_map:
            tmp_v = tmp_v[np.isfinite(tmp_v)]
            if tmp_v.size > 0: 
                if tmp_v.size > 1: 
                    tmp_v = np.unique(tmp_v)
                particle_v.append(tmp_v)
        particle_v = np.concatenate(particle_v) if len(particle_v) > 0 else np.asarray(particle_v)
        result['num_tracked_detection'] = particle_v.size
        
        num_gc_comp = [1, 2]
        if v_type == 'map': 
            v_data = tmp_e_skl_v_map[np.isfinite(tmp_e_skl_v_map)]
            v_data_size = result['num_tracked_frame']
        elif v_type == 'particle':
            v_data = particle_v
            v_data_size = particle_v.size

        if v_data_size >= min_data_size: 
            need_est_Q = True
            num_est = 0
            while need_est_Q and (num_est < gm_max_num_est): 
                tmp_v_all_gm = stat.analyze_1d_data_with_gaussian_mixture(v_data, num_gc_comp, vis_Q=vis_Q)
                num_est += 1
                need_est_Q = False
                if tmp_v_all_gm['num_component'] == 2: 
                    tmp_mean_diff = np.abs(tmp_v_all_gm['mean'][0] - tmp_v_all_gm['mean'][1])
                    tmp_min_std = np.min(tmp_v_all_gm['std'])
                    tmp_spacing_n = tmp_mean_diff / tmp_min_std
                    if (tmp_spacing_n < min_gmc_dist_std_n): 
                        if vis_Q:
                            print(f"The two mean is within {min_gmc_dist_std_n} std of the narrower component. Re-estimate")
                        if num_est < gm_max_num_est: 
                            need_est_Q = True
                        else: 
                            result['forced_1_comp_Q'] = True
                            tmp_v_all_gm = stat.analyze_1d_data_with_gaussian_mixture(v_data, [1], vis_Q=vis_Q)
                    elif (tmp_v_all_gm['mean'][0] * tmp_v_all_gm['mean'][1]) > 0: 
                        # same side. no need to do anything if opposite side 
                        if np.all(np.abs(tmp_v_all_gm['cv']) < max_gmc_cv_to_merge): 
                            # if both peak has small cv
                            result['forced_1_comp_Q'] = True
                            tmp_v_all_gm = stat.analyze_1d_data_with_gaussian_mixture(v_data, [1], vis_Q=vis_Q)
                        
                            
            # result |= tmp_v_all_gm
            if tmp_v_all_gm['num_component'] == 2: 
                tmp_major_idx = np.argmax(tmp_v_all_gm['weight'])
                tmp_minor_idx = 1 if (tmp_major_idx == 0) else 0
                tmp_weight_ratio = tmp_v_all_gm['weight']
                result['same_dir_Q'] = True if tmp_v_all_gm['mean'][0] * tmp_v_all_gm['mean'][1] > 0 else False
                result['weight_ratio'] = tmp_weight_ratio[tmp_major_idx] / tmp_weight_ratio[tmp_minor_idx]
                if vis_Q:
                    print(f"GM component weight ratio: {result['weight_ratio']:.2f}")

            elif tmp_v_all_gm['num_component'] == 1: 
                tmp_major_idx = 0
                result['weight_ratio'] = np.inf
            
            result['est_v'] = tmp_v_all_gm['mean'][tmp_major_idx]
            result['est_v_std'] = tmp_v_all_gm['std'][tmp_major_idx]
        else: 
            if vis_Q:
                print(f"This edge has less than {min_data_size} detections for flow estimation. ")
        
        return result
    
    @staticmethod
    def analyze_voxel_velocity_map_spatial_statistics(e_skl_v_map):
        """ Compute the median velocity of each skeleton and the percentiles of the skeleton median velocity
        Input: 
            e_skl_v_map: (N, T) np.array, where N is the number of skeleton 
                and T is the number of time point
        """
        tmp_valid_Q = np.any(np.isfinite(e_skl_v_map), axis=0)
        v_med, v_std = np.nan, np.nan
        if np.any(tmp_valid_Q): 
            e_skl_v_map = e_skl_v_map[:, tmp_valid_Q]
            skl_med_v = np.nanmedian(e_skl_v_map, axis=1)
            skl_ptrl_v = np.nanpercentile(skl_med_v, [15.9, 50, 84.1]) # +/- 1 sigma
            v_std = (skl_ptrl_v[2] - skl_ptrl_v[0]) / 2
            v_med = skl_ptrl_v[1]
        
        return v_med, v_std

    @staticmethod
    def analyze_edge_passengers(passengers):
        """
            passengers: list of tuple, each tuple is (passing time, detection_id, pid, speed_pxl2frame)
        
        """
        ps_info = {'num_detection': np.nan, 'pid': None, 'p_count': None}

        ps_data = np.vstack(passengers)
        ps_info['num_detection'] = ps_data.shape[0]
        ps_info['pid'], ps_info['p_count'] = np.unique(ps_data[:, 2].flatten().astype(np.int64), return_counts=True)
        passing_v = ps_data[:, 3].flatten()

        
        return ps_info


    def analyze_conditional_counts(self, bin_val, cond_count_multiedge, cond_count_single_edge, num_gm_comp):
        self.bin_val = bin_val
        self.cond_count_single_edge = cond_count_single_edge
        self.cond_count_multiedge = cond_count_multiedge
        self.cond_prob_single_edge = self.cond_count_single_edge / self.cond_count_single_edge.sum()
        self.cond_prob_multiedge = self.cond_count_multiedge / self.cond_count_multiedge.sum()
        self.gm_stat_multiedge = stat.analyze_hist_count_peaks_with_gaussian_mixture(bin_val, cond_count_multiedge, num_gm_comp, vis_Q=False)
        self.gm_stat_single_edge = stat.analyze_hist_count_peaks_with_gaussian_mixture(bin_val, cond_count_single_edge, num_gm_comp, vis_Q=False)
    
    def estimate_conditional_prob(self, x, use_stat='multiedge'):
        min_num_sample = 10
        max_cv = 0.4

        if use_stat == 'multiedge':
            gmstat = self.gm_stat_multiedge
        elif use_stat == 'single_edge':
            gmstat = self.gm_stat_single_edge
        
        if gmstat['num_sample'] < min_num_sample:
            return np.nan
        else: 
            gmstat['cv'] = gmstat['std'] / gmstat['mean']

            pass

    def vis(self):
        print(f"Length: {self.length_w_node}")
        print(f"Number of observations: {self.gm_stat_single_edge['num_sample']}, {self.gm_stat_multiedge['num_sample']}")
        plt.figure()
        plt.plot(self.bin_val, self.cond_count_multiedge)
        plt.plot(self.bin_val, self.cond_count_single_edge)
        if self.gm_stat_multiedge['num_sample'] > 0: 
            print(f"Multiedge MLE speed: {self.gm_stat_multiedge['peak_mean']:.2f} +/- {self.gm_stat_multiedge['peak_std']:.2f}")
        else: 
            print("Insufficient multiedge stat")
        if self.gm_stat_single_edge['num_sample'] > 0: 
            print(f"Single edge MLE speed: {self.gm_stat_single_edge['peak_mean']:.2f} +/- {self.gm_stat_single_edge['peak_std']:.2f}")
        else: 
            print("Insufficient single edge stat")

    def get_raw_cond_count(self, x, use_stat='multiedge'):
        x = np.asarray(x)
        max_val = np.ceil(self.bin_val[-1])
        count = np.zeros(x.shape)
        in_range_Q = np.abs(x) <= max_val
        x = x[in_range_Q]
        x_bin_idx = np.floor(x + max_val).astype(np.int16)
        if use_stat == 'multiedge':
            count[in_range_Q] = self.cond_count_multiedge[x_bin_idx]
        elif use_stat == 'single_edge':
            count[in_range_Q] = self.cond_count_single_edge[x_bin_idx]
        return count
    
    def get_raw_cond_prob(self, x, use_stat='multiedge'):
        x = np.asarray(x)
        max_val = np.ceil(self.bin_val[-1])
        prob = np.zeros(x.shape)
        in_range_Q = np.abs(x) <= max_val
        x = x[in_range_Q]
        x_bin_idx = np.floor(x + max_val).astype(np.int16)
        if use_stat == 'multiedge':
            prob[in_range_Q] = self.cond_prob_multiedge[x_bin_idx]
        elif use_stat == 'single_edge':
            prob[in_range_Q] = self.cond_prob_single_edge[x_bin_idx]
        return prob
    
#region Visualization
    def vis_w_mip(self, mip):
        f = plt.figure(figsize=(8, 8))
        a = f.add_subplot()
        a.imshow(mip, cmap='gray')
        a.scatter(self.pos[1], self.pos[2], 1, color='r')

    def vis_w_3_mip(self, mips, figsize=(10, 10)):
        f = vis.vis_mips(mips, pts_zyx=self.pos, figsize=figsize, fig_title=f"Edge {self.label} skeleton on MIPs")
        return f
    
    def vis_w_3_mip_local(self, im_vol, pad=30):
        f = vis.vis_pts_w_local_mips(self.pos, im_vol, bbox_expand=pad,
                                      show_axes_Q=True, fig_title=f"Edge #{self.label}")
        return f
    
    def vis_detection_map(self, frame_range=None, figsize=(20, 4)):
        if frame_range is None: 
            data = self.detect_map.T
        else:
            data = self.detect_map[frame_range[0] : frame_range[1]].T
        m_h, m_w = data.shape        
        i_w, i_h = figsize
        asp_r = (m_w / i_w) / (m_h / i_h)
        
        f = plt.figure(figsize=figsize)
        a = f.add_subplot()
        a.imshow(data, cmap='jet', interpolation='nearest')
        a.set_aspect(asp_r)
        a.set_xlabel('Frame')
        a.set_ylabel('Voxel index')

    def vis_voxel_velocity_map(self, pxl2f_to_mm2s=None, figsize=(20, 3)):
        vis_skl_map = self.vxl_speed_map.copy()
        if vis_skl_map.ndim == 1: 
            vis_skl_map = vis_skl_map[None, :]

        vis_skl_map[np.isnan(vis_skl_map)] = 0
        if pxl2f_to_mm2s is not None: 
            vis_skl_map *= pxl2f_to_mm2s
            c_label = 'speed (mm/s)'
        else: 
            c_label = 'speed (vxl/frame)'
        m_h, m_w = vis_skl_map.shape        
        i_w, i_h = figsize
        asp_r = (m_w / i_w) / (m_h / i_h)
        
        f = plt.figure(figsize=figsize)
        a = f.add_subplot()
        im = a.imshow(np.abs(vis_skl_map), cmap=vis.get_cmap_with_black_for_0(), interpolation='nearest')
        a.set_aspect(asp_r)
        a.set_xlabel("Frame index")
        a.set_ylabel("Voxel index")
        cbar = f.colorbar(im, label=c_label)

    def vis_detection_and_vxl_v_map(self, frame_range=None, pxl2f_to_mm2s=None, figsize=(20, 6)): 
        vis_skl_map = self.vxl_speed_map.copy()
        if vis_skl_map.ndim == 1: 
            vis_skl_map = vis_skl_map[None, :]
        vis_skl_map[np.isnan(vis_skl_map)] = 0

        if pxl2f_to_mm2s is not None: 
            vis_skl_map *= pxl2f_to_mm2s
            c_label = 'speed (mm/s)'
        else: 
            c_label = 'speed (vxl/frame)'

        dmap = self.detect_map.T

        if frame_range is not None: 
            dmap = dmap[:, frame_range[0] : frame_range[1]]
            vis_skl_map = vis_skl_map[:, frame_range[0] : frame_range[1]]

        i_w, i_h = figsize
        m_h, m_w = dmap.shape        
        asp_r_1 = (m_w / i_w) / (m_h / (i_h / 2))

        m_h, m_w = vis_skl_map.shape        
        asp_r_2 = (m_w / i_w) / (m_h / (i_h / 2))

        f = plt.figure(figsize=figsize)
        a1 = f.add_subplot(2, 1, 1)
        a2 = f.add_subplot(2, 1, 2)

        im_1 = a1.imshow(dmap, cmap='jet', interpolation='nearest')
        a1.set_aspect(asp_r_1)
        a1.set_ylabel('Distance (pxl)')
        cbar_1 = f.colorbar(im_1, label='Count')

        im = a2.imshow(np.abs(vis_skl_map), cmap=vis.get_cmap_with_black_for_0(), interpolation='nearest')
        a2.set_aspect(asp_r_2)
        a2.set_xlabel("Frame index")
        a2.set_ylabel("Voxel index")
        cbar_2 = f.colorbar(im, label=c_label)

        return f

    @staticmethod
    def vis_shifted_pair_corr_mat(corr, dt, figsize=(6, 5)):
        f = plt.figure(figsize=figsize)
        a = f.add_subplot()
        im = a.imshow(corr, cmap='jet')
        diag_x = np.arange(corr.shape[0])
        a.plot(diag_x, diag_x, color='w')
        a.set_title(f"<Fi(t), Fj(t + {dt})>")
        cb = f.colorbar(im)
        cb.set_label("Corr. Coeff.")        
        return f, a

#endregion

#region Linear assignment based particle matching - not used
    @staticmethod
    def compute_corr_in_range_along_axis_1(mat_1, mat_2, shift_list):
        shift_list = np.atleast_1d(np.asarray(shift_list))
        result = np.zeros((mat_1.shape[0], shift_list.size))
        for n, i in enumerate(shift_list): 
            if i > 0: 
                result[:, n] = np.sum(mat_1[:, i:] * mat_2[:, :-i], axis=1)
            elif i < 0: 
                result[:, n] = np.sum(mat_1[:, :i] * mat_2[:, -i:], axis=1)
            else: 
                result[:, n] = np.sum(mat_1 * mat_2, axis=1)
        return result

    @staticmethod
    def compute_detection_corr_in_range_w_smoothing(detect_map, shift_list, sigma=None, visQ=False, normalizedQ=False):
        """
        This approach is adapted from Zhang et al 2020. It make sense and appear to be an 
        easy way to deal with exicting particles. However, for a long segment containing more 
        than one particle, the resulting correlation profile strongly depends on the size 
        of gaussian kernel. Intuitively, we would expect a smaller gaussian kernel for smaller 
        shift (and thus lower speed), but this will produce high correlation for a single particle 
        match.
        The particle matching in large vessels is further complicated by: 
        1. Change of intensity along z
        2. The radial-dependent velocity: to what extent do the RBCs follow a laminar flow? 
        3. Low resolution - seems hard to determine the boundary of the vessels - hard for modeling 
        Even if the static structure is known, how much motion are there in the video? Seems hard 
        to quantify given the noisness of the data. 
        """
        shift_list = np.atleast_1d(np.asarray(shift_list))
        result = np.zeros((detect_map.shape[0] - 1, shift_list.size))
        if sigma is not None: 
            if sigma > 0: 
                smoothed_map = ndi.gaussian_filter1d(detect_map, sigma=sigma, axis=1, mode='constant')
            else: 
                smoothed_map = detect_map
        for n, i in enumerate(shift_list):
            if sigma is None:
                sig = max(1, 0.1 * abs(i))
                smoothed_map = ndi.gaussian_filter1d(detect_map, sigma=sig, axis=1, mode='constant')

            if i > 0: 
                result[:, n] = np.sum(smoothed_map[1:, i:] * smoothed_map[:-1, :-i], axis=1)
                # if normalizedQ: 
                #     result[:, n] /= 
            elif i < 0: 
                result[:, n] = np.sum(smoothed_map[1:, :i] * smoothed_map[:-1, -i:], axis=1)
            else: 
                result[:, n] = np.sum(smoothed_map[1:] * smoothed_map[:-1], axis=1)
        if visQ:
            f = plt.figure(figsize=(8, 4))
            a1 = f.add_subplot(2, 1, 1)
            a1.plot(shift_list, result.flatten())
            a1.set_xlabel("Shift (pxl)")
            a1.set_ylabel("Correlation")
            a1.set_title("Adaptive smoothing" if sigma is None else f"Fixed smoothing kernel (sig = {sigma})")
            a2 = f.add_subplot(2, 1, 2)
            a2.imshow(detect_map)
            a2.set_title("Detection map")
            f.show()

        return result

    @staticmethod
    def match_adj_t_points_in_single_edge_single_step(pos_t, pos_tp1, length, est_speed_pxl, exit_cost=None, 
                                                      verboseQ=False): 
        """
            Match the points in adjacent time frame within the single edge given the estiamted velocity and exit cost
            Args: 
                pos_t: 1D vector, position of the points at frame (t) in the segment, measured from the left node 
                pos_tp1: 1D vector, position of the points at frame (t+1) in the segment, measured from the left node 
                est_pseed_pxl: real scalar, positive if the flow is along the edge voxel indices order 
                esit_cost: positive scalar in the dimension of "distance", cost of assignment the particle 
                        to the exit. 
                verboseQ: logical scalar. Print information if being True.
        """
        # eliminate detections using estimated speed
        result = {'initial_speed': est_speed_pxl, 'number_of_match': 0, 'matched_idx_t': None,
                   'exit_idx_t': None, 'matched_idx_tp1': None, 'matched_dist': None, 'matched_model_error': None,
                   'avg_matched_cost': None, 'avg_match_error_n': None, 'avg_exit_error_n': None, 'avg_error_n': None, 
                    'continue_Q': False, 'new_speed':None, }
        pos_t = np.asarray(pos_t)
        pos_tp1 = np.asarray(pos_tp1)
        if pos_t.size == 0 or pos_tp1.size == 0: 
            return result
        
        # Particles are within one edge - sort position 
        t_s_idx = np.argsort(pos_t)
        pos_t = pos_t[t_s_idx]
        tp1_s_idx = np.argsort(pos_tp1)
        pos_tp1 = pos_tp1[tp1_s_idx]

        # Analyze the distance between the particle and the edge endpoint
        if exit_cost is None:
            # Need a positive, and large enough exit cost, otherwise the algorithm 
            # would keep increasing the velocity to move all the particles at (t) 
            # out of the interval 
            # not sure if is a good choice. 
            exit_cost = est_speed_pxl * 0.5 
        if est_speed_pxl > 0: 
            test_dist_to_exit = length - pos_t - est_speed_pxl
            test_dist_to_exit = np.maximum(exit_cost, test_dist_to_exit)
            # points @ (t + 1) in the forward direction 
            # how to deal with stalled cells (and at the boundary of the skeleton voxels ...?)
            # Add noise? Subtract a small value from the minimal position? 
            tp1_forward_Q = pos_tp1 > np.min(pos_t) 
        elif est_speed_pxl < 0:
            test_dist_to_exit = pos_t + est_speed_pxl
            test_dist_to_exit = np.maximum(exit_cost, np.abs(test_dist_to_exit))
            tp1_forward_Q = pos_tp1 < np.max(pos_t)
        else: 
            raise NotImplementedError("The estimated velocity must be non-zero.")
        
        if np.any(tp1_forward_Q):
            if not np.all(tp1_forward_Q):
                forward_tp1_idx = np.nonzero(tp1_forward_Q)[0]
                pos_tp1 = pos_tp1[tp1_forward_Q]
            else: 
                forward_tp1_idx = np.arange(pos_tp1.size)
            # Construct the cost of exit matrix 
            exit_cost_mat = np.full((pos_t.size, pos_t.size), np.inf)
            exit_cost_mat[np.eye(pos_t.size).astype(bool)] = test_dist_to_exit
            # pairwise distance 
            pdist_mat = pos_tp1[None, :] - pos_t[:, None]
            predict_pdist_mat = pdist_mat - est_speed_pxl
            cost_mat = np.concatenate((np.abs(predict_pdist_mat), exit_cost_mat), axis=1)
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            matched_Q = col_ind < pos_tp1.size

            result['number_of_match'] = np.count_nonzero(matched_Q)
            if np.any(matched_Q):
                matched_row_ind = row_ind[matched_Q]
                matched_col_ind = col_ind[matched_Q]
                exit_row_ind = row_ind[~matched_Q]
                exit_col_ind = col_ind[~matched_Q]
                match_dist = pdist_mat[matched_row_ind, matched_col_ind]
                exit_dist = cost_mat[exit_row_ind, exit_col_ind]
                
                new_speed_pxl = np.round(match_dist.mean(), decimals=2) 
                if new_speed_pxl == 0: 
                    new_speed_pxl = 1e-2 if (est_speed_pxl > 0) else -1e-2

                delta_speed_pxl = new_speed_pxl - est_speed_pxl
                # Matched particles
                model_error = match_dist - new_speed_pxl
                result['avg_matched_cost'] = np.sqrt(np.sum((model_error) ** 2)) / match_dist.size # l2 cost
                result['avg_match_error_n'] = np.mean(np.abs(model_error) / np.abs(new_speed_pxl))
                # Missing particles 
                if est_speed_pxl > 0: 
                    # exit_dist is always positive
                    exit_error = exit_dist - new_speed_pxl
                else: 
                    exit_error = exit_dist + new_speed_pxl
                # Consider the particles that are still in the segment but assigned to exit
                # Not sure how to compute the error for the exitted particles 
                non_match_non_exit_Q = exit_error > 0
                num_nonmne = np.count_nonzero(non_match_non_exit_Q)
                result['avg_exit_error_n'] = np.mean(exit_error[non_match_non_exit_Q]) / np.abs(new_speed_pxl) \
                    if np.any(non_match_non_exit_Q > 0) else 0
                result['avg_error_n'] = (result['avg_match_error_n'] * match_dist.size + 
                                         result['avg_exit_error_n'] * num_nonmne) / (match_dist.size + num_nonmne)

                result['matched_idx_t'] = t_s_idx[matched_row_ind]
                result['matched_idx_tp1'] = tp1_s_idx[forward_tp1_idx[matched_col_ind]]
                result['exit_idx_t'] = t_s_idx[exit_row_ind]
                result['new_speed'] = new_speed_pxl
                result['matched_model_error'] = model_error
                result['matched_dist'] = match_dist
                result['continue_Q'] = np.abs(delta_speed_pxl) > 1e-1

                if verboseQ:
                    print("pos @ t:",  np.round(pos_t))
                    print("pos @ (t + 1):", np.round(pos_tp1))
                    # print("Dist matrix:\n", np.round(pdist_mat))
                    # print(f"Cost matrix:\n", np.round(cost_mat))
                    print(f"Match: ({matched_row_ind}) -> ({matched_col_ind})")
                    print(f"Matched distance:", np.round(result['matched_dist']))
                    print(f"Model error:", np.round(result['matched_model_error']))
                    print(f"Average absolute model error: {result['avg_match_error_n']:.2f}")
                    print(f"Average exit error: {result['avg_exit_error_n']:.2f}")
                    print(f"Total match cost: {result['avg_matched_cost']:.2f}")  
                    print(f"New test speed: {result['new_speed']:.1f}\nSpeed update: {delta_speed_pxl:.1f}")
            else: 
                if verboseQ: 
                    print("No matched points.")
        else: 
            if verboseQ:
                print(f"Does not exist points in frame (t + 1) in the forward direction.")

        return result
    
    @staticmethod
    def match_adj_t_points_in_single_edge_iterate(pos_t, pos_tp1, length, est_speed_pxl, exit_cost=None, verbose_Q=False, 
                                                  cached_Q=True, clear_cache_Q=False):     
        """
            Wrapper function for iteratively estimating average flow velocity and matching particle paris in adjacent
            time frame. 
        """
        if not hasattr(EdgeFlow.match_adj_t_points_in_single_edge_iterate, "_cache"):
            EdgeFlow.match_adj_t_points_in_single_edge_iterate._cache = {}
            EdgeFlow.match_adj_t_points_in_single_edge_iterate._final_cache = {}
        if clear_cache_Q: 
            EdgeFlow.match_adj_t_points_in_single_edge_iterate._cache.clear()
            EdgeFlow.match_adj_t_points_in_single_edge_iterate._final_cache.clear()
        init_speed_pxl = est_speed_pxl
        converged_Q = False
        i_count = 0
        i_max = 10
        computed_speed_list = []
        while not converged_Q:
            if verbose_Q:
                print(f"\nIteration {i_count}: initial velocity: {est_speed_pxl:.1f}")
            est_speed_pxl = np.round(est_speed_pxl, decimals=2)
            computed_speed_list.append(est_speed_pxl)

            if cached_Q and (est_speed_pxl in EdgeFlow.match_adj_t_points_in_single_edge_iterate._final_cache): 
                result = EdgeFlow.match_adj_t_points_in_single_edge_iterate._final_cache[est_speed_pxl]
                if verbose_Q: 
                    print("Use cached final result")
                break

            if cached_Q and (est_speed_pxl in EdgeFlow.match_adj_t_points_in_single_edge_iterate._cache): 
                result = EdgeFlow.match_adj_t_points_in_single_edge_iterate._cache[est_speed_pxl]
                if verbose_Q: 
                    print("Use cached result")
            else: 
                result = EdgeFlow.match_adj_t_points_in_single_edge_single_step(pos_t, pos_tp1, length, est_speed_pxl,
                                                                                exit_cost=exit_cost, verboseQ=verbose_Q)
                result['initial_speed'] = init_speed_pxl
                if cached_Q:
                    if verbose_Q: 
                        print(f"Add ({est_speed_pxl}) to cache")
                    EdgeFlow.match_adj_t_points_in_single_edge_iterate._cache[est_speed_pxl] = result
            
            converged_Q = ~ result['continue_Q']
            if not converged_Q: 
                # update speed estimation 
                est_speed_pxl = result['new_speed'] if (result['new_speed'] is not None) else None
                i_count +=1
                if est_speed_pxl in computed_speed_list: 
                    old_result = EdgeFlow.match_adj_t_points_in_single_edge_iterate._cache[est_speed_pxl]
                    if old_result['avg_error_n'] < result['avg_error_n']: 
                        result = old_result
                    if verbose_Q:
                        print(f"Get into infinite loop. Select result based on average normalized error")
                    break 
                
            if i_count > i_max: 
                raise Exception(f"Number of iteration exceeds {i_max}. Unexpected behavior")

        if cached_Q: 
            for es in computed_speed_list: 
                if es not in EdgeFlow.match_adj_t_points_in_single_edge_iterate._final_cache: 
                    EdgeFlow.match_adj_t_points_in_single_edge_iterate._final_cache[es] = result
                    if verbose_Q: 
                        print(f"Add ({es}) to final cache")

        return result

    @staticmethod
    def match_adj_t_points_in_single_edge_scan(pos_t, pos_tp1, length, est_speed_pxl_list, exit_cost=None, verbose_Q=False): 
        est_speed_pxl_list = np.asarray(est_speed_pxl_list)
        s_i = np.argsort(np.abs(est_speed_pxl_list))
        est_speed_pxl_list = est_speed_pxl_list[s_i]
        result = []
        prior_est_speed = []
        def skip_Q(est_v, done_v):
            # same direction 
            done_v = np.asarray(done_v)
            return np.any(np.logical_and(done_v * est_v > 0, np.abs(done_v) > np.abs(est_v))) 

        for i, est_speed_pxl in enumerate(est_speed_pxl_list):
            tmp_result = EdgeFlow.match_adj_t_points_in_single_edge_iterate(pos_t, pos_tp1, length, est_speed_pxl, 
                                                                            exit_cost=exit_cost, verbose_Q=verbose_Q, clear_cache_Q=False, cached_Q=True)
            if tmp_result['number_of_match'] > 0: 
                    prior_est_speed.append(tmp_result['new_speed'])
                    result.append(tmp_result)               
        # clear cache result
        EdgeFlow.match_adj_t_points_in_single_edge_iterate._cache.clear()
        EdgeFlow.match_adj_t_points_in_single_edge_iterate._final_cache.clear()

        if len(result) > 1: 
            tmp_speed = np.asarray([r['new_speed'] for r in result])
            tmp_speed = np.round(tmp_speed, decimals=2)
            tmp_v, tmp_idx = np.unique(tmp_speed, return_index=True)
            if verbose_Q: 
                print("Possible velocity: ", tmp_v)
            result = [result[i] for i in tmp_idx]

        return result
    
    @staticmethod
    def match_adj_t_points_in_single_edge_result_selection(results):
        if len(results) == 0: 
            return []
        elif len(results) == 1: 
            return results[0]
        else: 

            pass

    def compute_matching_in_single_edge(self, est_speed_list, exit_cost=None, verbose_Q=False):
        skl_ind = self.detections.skl_ind.values
        frame_skl_ind = list(map(lambda i: skl_ind[i].flatten() if (i.size > 0) else [], self.frame_ind))
        result = []
        for t_idx in range(self.frame_ind.size - 1): 
            try: 
                test_i_t = frame_skl_ind[t_idx]
                test_i_tp1 = frame_skl_ind[t_idx + 1]
                test_epdist_t = np.abs(self.ind_to_ep_dist(test_i_t))
                test_epdist_tp1 = np.abs(self.ind_to_ep_dist(test_i_tp1)) 
                tmp_result = EdgeFlow.match_adj_t_points_in_single_edge_scan(
                    test_epdist_t[:, 0].flatten(), test_epdist_tp1[:, 0].flatten(),
                    self.length, est_speed_list, exit_cost=exit_cost, verbose_Q=verbose_Q)
                result.append(tmp_result)
            except: 
                print(f"Encounder error at time {t_idx}")
                raise
        return result
    
#endregion

#region Analysis

    def analyze_edge_spatiotemporal_dynamics(self, mv_avg_wd=35, mv_min_num=3, 
                                             vxl_size_um=1, frame_rate_Hz=1, 
                                             labeled_fraction=1, dist_c_ctr_to_wall_um=1, dstd2radius=2):
        pxl2f_to_um2s = vxl_size_um * frame_rate_Hz

        vxl_vol_map = self.vxl_speed_map * pxl2f_to_um2s if pxl2f_to_um2s != 1 else self.vxl_speed_map # (l, T)
        eff_l = np.maximum(1, self.length) * vxl_size_um

        # log all the input arguments 
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        edge_stat = {arg: values[arg] for arg in args if arg != 'self'}
        edge_stat['label'] = self.label
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            edge_stat['t'] = np.arange(vxl_vol_map.shape[1]) / frame_rate_Hz

            edge_stat['stat_v'] = stat.compute_basic_statistics(vxl_vol_map)            
            # Temporal: Cell average density in each frame
            e_t_num_cell = np.sum(self.detect_map, axis=1)
            # should we set 0 to nan here ???
            # e_t_num_cell[e_t_num_cell == 0] = np.nan
            edge_stat['e_t_cell_count'] = e_t_num_cell
            edge_stat['stat_count'] = stat.compute_basic_statistics(edge_stat['e_t_cell_count'])
            edge_stat['e_t_cell_density'] = e_t_num_cell / labeled_fraction / eff_l      
            edge_stat |= EdgeFlow.__compute_moving_smoothed_traces(edge_stat['e_t_cell_density'], mv_avg_wd=mv_avg_wd, mv_min_num=mv_min_num, name='e_t_cell_density')

            # Spatial: Skeleton voxel velocity statistics over time 
            edge_stat['skl_num_valid'] = np.sum(np.isfinite(vxl_vol_map), axis=1)
            edge_stat['skl_detect_rate_mean'] = np.nanmean(self.vxl_detection_map, axis=0)
            edge_stat['skl_detect_rate_std'] = np.nanstd(self.vxl_detection_map, axis=0)
            edge_stat['skl_v_mean'] = np.nanmean(vxl_vol_map, axis=1)
            edge_stat['stat_skl_v'] = stat.compute_basic_statistics(edge_stat['skl_v_mean'])
            edge_stat['skl_v_std'] = np.nanstd(vxl_vol_map, axis=1)
            edge_stat['skl_v_med'] = np.nanmedian(vxl_vol_map, axis=1)
            # Temporal: edge velocity over time - based on voxel velocity map
            # Edge velocity estimated from the voxel velocity map : 
            #   (1) get the velocity across the entire segment, but faster particles contribute more than the slower particles
            #   (2) Include the information when a particle enters the segment (including the one disapper in the segment) - not sure if it's good or bad... 
            edge_stat['e_t_valid_frac'] = np.mean(np.isfinite(vxl_vol_map), axis=0)
            edge_stat['e_t_v_mean'] = np.nanmean(vxl_vol_map, axis=0) # average velocity inside the edge over time 
            edge_stat['stat_e_v'] = stat.compute_basic_statistics(edge_stat['e_t_v_mean'])
            pds_e_t_v_mean = pd.Series(edge_stat['e_t_v_mean']).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True)
            edge_stat['e_t_v_mean_sm'] = pds_e_t_v_mean.mean().values
            edge_stat['e_t_v_mean_smed'] = pds_e_t_v_mean.median().values
            
            # RBC flux
            edge_stat['e_t_c_flux'] = np.abs(edge_stat['e_t_v_mean'] * edge_stat['e_t_cell_density'])
            edge_stat['stat_flux'] = stat.compute_basic_statistics(edge_stat['e_t_c_flux'])
            edge_stat['e_t_c_flux_smp'] = np.abs(edge_stat['e_t_v_mean_sm'] * edge_stat['e_t_cell_density_sm'])
            
            if self.cell_passenger is not None: 
                cp_stat = self.analyze_cell_pessenger_traces(pxl2f_to_um2s)
                cp_stat['e_t_cxf'] = cp_stat['e_t_cxf'] * frame_rate_Hz / labeled_fraction
                edge_stat |= cp_stat
                edge_stat['p_t_v_mean_sm'] = pd.Series(edge_stat['p_t_v_mean']).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True).mean().values    
                edge_stat['p_t_rbc_flux'] = np.abs(edge_stat['p_t_v_mean'] * edge_stat['e_t_cell_density'])
                edge_stat['p_t_rbc_flux_smp'] = np.abs(edge_stat['p_t_v_mean_sm'] * edge_stat['e_t_cell_density_sm'])

                # Add the crossing RBC flux to the flux computed based on density and speed
                edge_stat['e_t_c_flux'] = EdgeFlow.__add_not_recorded_crossing_rbc_flux_to_rbc_flux(edge_stat['e_t_c_flux'].copy(), cp_stat['e_t_cxf'])
                edge_stat['p_t_rbc_flux'] =  EdgeFlow.__add_not_recorded_crossing_rbc_flux_to_rbc_flux(edge_stat['p_t_rbc_flux'].copy(), cp_stat['e_t_cxf'])

                # not sure how to add the crossing flux to the flux_smp...
                # crossing_flux[crossing_flux == 0] = np.nan
                # edge_stat['e_t_c_flux_smp'] += pd.Series(crossing_flux).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True).mean().values

            # Temporal: Cell radial distribution
            p_radial_dist = self.analyze_cell_radial_distribution(mv_avg_wd=mv_avg_wd, mv_min_num=mv_min_num, vis_Q=False)
            # edge_stat['e_t_radius'] = p_radial_dist['d2ctr_std'] * dstd2radius * vxl_size_um + dist_c_ctr_to_wall_um # um. this is too noisy
            edge_stat['e_t_radius_sm'] = p_radial_dist['d2ctr_std_sm'] * dstd2radius * vxl_size_um + dist_c_ctr_to_wall_um # um 
            edge_stat['stat_radius_sm'] = stat.compute_basic_statistics(edge_stat['e_t_radius_sm'])
                # Volumetric flux
            edge_stat['e_t_vol_flux_smp'] = np.abs(edge_stat['e_t_v_mean_sm']) * (np.pi * edge_stat['e_t_radius_sm'] ** 2)
            # e_t_vol_flux = np.abs(edge_stat['e_t_v_mean']) * (np.pi * edge_stat['e_t_radius'] ** 2)
            # edge_stat['e_t_vol_flux_smed'] = pd.Series(e_t_vol_flux).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True).median().values
            edge_stat['e_t_radius_t_avg'] = p_radial_dist['d2ctr_std_avg'] * dstd2radius * vxl_size_um + dist_c_ctr_to_wall_um # um 
            edge_stat['e_t_vol_flux_sm'] = np.abs(edge_stat['e_t_v_mean_sm'] * (np.pi * edge_stat['e_t_radius_t_avg'] ** 2))
            edge_stat['e_t_vol_flux_smed'] = np.abs(edge_stat['e_t_v_mean_smed'] * (np.pi * edge_stat['e_t_radius_t_avg'] ** 2))

            # recalculate again. Not all frames with a valid velocity has a valid density count - especially for the short segments. 
            edge_stat |= EdgeFlow.__compute_moving_smoothed_traces(edge_stat['e_t_c_flux'], mv_avg_wd=mv_avg_wd, mv_min_num=mv_min_num, name='e_t_c_flux')
            edge_stat |= EdgeFlow.__compute_moving_smoothed_traces(edge_stat['p_t_rbc_flux'], mv_avg_wd=mv_avg_wd, mv_min_num=mv_min_num, name='p_t_rbc_flux')

        return edge_stat

    def analyze_cell_pessenger_traces(self, pxl2f_to_um2s=1):
        cp_stat = {}
        num_t_pts = self.frame_ind.size
        crossing_flux, crossing_count, cross_speed_sum, cross_speed2_sum = (np.zeros(num_t_pts) for _ in range(4))
        # Does it over estiamte the flux or not? 
        # The particles recorded in cell_passenger are the TRACKED particles. Some particles are detected but not tracked. 
        # There might be some particles that are detected but not tracked - despite false positive detection. 
        # Therefore, RBC estimated here is probably an under estimation. 
        added_did = set(self.detections.did.values)
        p_v = np.full(len(self.cell_passenger), np.nan)
        for i, tmp_p in enumerate(self.cell_passenger): 
            tmp_t, tmp_did, tmp_pid, tmp_speed = tmp_p
            crossing_count[tmp_t] += 1
            tmp_speed *= pxl2f_to_um2s
            p_v[i] = tmp_speed
            cross_speed_sum[tmp_t] += tmp_speed # speed, non-negative
            cross_speed2_sum[tmp_t] += tmp_speed ** 2
            if tmp_did not in added_did: 
                crossing_flux[tmp_t] += 1
        
        cp_stat['stat_p_v'] = stat.compute_basic_statistics(p_v)
        if np.isfinite(cp_stat['stat_p_v']['median']): 
            reverse_frac = np.interp(0, cp_stat['stat_p_v']['prctile_val'], cp_stat['stat_p_v']['prctile_th']) / 100
            cp_stat['stat_p_v']['reverse_frac'] = reverse_frac if cp_stat['stat_p_v']['median'] > 0 else (1 - reverse_frac)
        else: 
            cp_stat['stat_p_v']['reverse_frac'] = np.nan
        cp_stat['e_t_cxf'] = crossing_flux
        cp_stat['p_t_v_mean'] = cross_speed_sum / crossing_count 
        cp_stat['p_t_v_std'] = np.sqrt(cross_speed2_sum / crossing_count - cp_stat['p_t_v_mean'] ** 2)
        return cp_stat
    
    @staticmethod
    def analyze_cell_pessenger_distribution(pv):
        cp_stat = {}
        cp_stat['stat_all'] = stat.compute_basic_statistics(pv)
        cp_stat['stat_p'] = stat.compute_basic_statistics(pv[pv >= 0])
        cp_stat['stat_n'] = stat.compute_basic_statistics(pv[pv <= 0])
        copy_keys = ['num_data', 'mean', 'std', 'cv', 'median', 'eff_ptrl_std', 'eff_ptrl_cv']
        

        pass
    
    @staticmethod
    def __add_not_recorded_crossing_rbc_flux_to_rbc_flux(rbc_flux, cxr_flux):
        tmp_is_nan_Q = np.isnan(rbc_flux)
        tmp_has_cross_Q = (cxr_flux > 0)
        tmp_replace_nan_Q = np.logical_and(tmp_is_nan_Q, tmp_has_cross_Q)
        rbc_flux[tmp_replace_nan_Q] = rbc_flux[tmp_replace_nan_Q]
        rbc_flux[~tmp_is_nan_Q] += rbc_flux[~tmp_is_nan_Q]
        return rbc_flux

    @staticmethod
    def __compute_moving_smoothed_traces(trace, mv_avg_wd, mv_min_num=1, name=None):
        name = f"{name}_" if name is not None else ""
        # count the number of valid data point in the moving window
        std_to_se = 1 / pd.Series(np.isfinite(trace)).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True).sum().values
        pds_e_t_c_flux = pd.Series(trace).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True)
        result = {}
        result[f"{name}sm"] = pds_e_t_c_flux.mean().values
        # result[f"{name}smed"] = pds_e_t_c_flux.median().values
        result[f"{name}sstd"] = pds_e_t_c_flux.std().values
        result[f"{name}sse"] = result[f"{name}sstd"] / std_to_se
        return result

    def analyze_cell_radial_distribution(self, mv_avg_wd, mv_min_num=3, vis_Q=False):
        
        edge_table = self.detections
        num_t_pts = self.frame_num_detect.size

        p_pos = edge_table[['z', 'y', 'x']].to_numpy()
        p_skl_sub = np.column_stack(np.unravel_index(edge_table.skl_ind.values, self.mask_size) )
        f_to_t_idx = util.get_table_value_to_idx_dict(edge_table, key='frame')
        ind_to_t_idx = util.get_table_value_to_idx_dict(edge_table, key='skl_ind')
        ind_to_cc_v_idx = {ind : i for i, ind in enumerate(self.ind)}
        p_cc_v_idx = np.asarray([ind_to_cc_v_idx[ind] for ind in edge_table.skl_ind.values])
        skl_sub = self.pos[:, 1:-1].T
        # compute the averaged position and std over time for each skeleton voxel 
        stat_p_to_ctr = defaultdict(list)
        for i, tmp_skl_ind in enumerate(self.ind):
            if tmp_skl_ind in ind_to_t_idx: 
                tmp_skl_t_idx = ind_to_t_idx[tmp_skl_ind]
                tmp_pos = p_pos[tmp_skl_t_idx]
                tmp_pos_avg = np.mean(tmp_pos, axis=0)
                tmp_avg_dist = tmp_pos_avg - skl_sub[i]
                tmp_t_dist = np.sqrt(np.sum((p_pos[tmp_skl_t_idx] - p_skl_sub[tmp_skl_t_idx]) ** 2, axis=1)) 
                tmp_t_dist_2_avg = np.sqrt(np.sum((p_pos[tmp_skl_t_idx] - tmp_pos_avg[None, :]) ** 2, axis=1)) 
            else: 
                tmp_t_dist = np.nan
                tmp_t_dist_2_avg = np.nan
                tmp_pos_avg = skl_sub[i]
                tmp_avg_dist = np.full(0, np.nan)

            stat_p_to_ctr['p_dist_to_skl_mean'].append(np.mean(tmp_t_dist))
            stat_p_to_ctr['p_dist_to_skl_std'].append(np.std(tmp_t_dist))
            stat_p_to_ctr['p_dist_to_avg_mean'].append(np.mean(tmp_t_dist_2_avg))
            stat_p_to_ctr['p_dist_to_avg_std'].append(np.std(tmp_t_dist_2_avg))
            stat_p_to_ctr['p_avg_pos'].append(tmp_pos_avg)
            stat_p_to_ctr['p_avg_pos_to_skl'].append(tmp_avg_dist)

        skl_avg_p_pos = np.vstack(stat_p_to_ctr['p_avg_pos'])
        num_detect = np.full(num_t_pts, np.nan)
        detect_to_skl_sum = np.full(num_t_pts, np.nan)
        detect_to_ctr_sum = np.full(num_t_pts, np.nan)
        for tmp_f in range(num_t_pts):
            if tmp_f in f_to_t_idx: 
                tmp_t_idx = f_to_t_idx[tmp_f]
                tmp_t_dist_to_skl = np.sum((p_pos[tmp_t_idx] - p_skl_sub[tmp_t_idx]) ** 2, axis=1)
                tmp_t_dist_to_avg = np.sum((p_pos[tmp_t_idx] - skl_avg_p_pos[p_cc_v_idx[tmp_t_idx]]) ** 2, axis=1)
                num_detect[tmp_f] = tmp_t_idx.size
                detect_to_skl_sum[tmp_f] = np.sum(tmp_t_dist_to_skl)
                detect_to_ctr_sum[tmp_f] = np.sum(tmp_t_dist_to_avg)

        d2skl_sum = pd.Series(detect_to_skl_sum).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True).sum().values
        d2ctr_sum = pd.Series(detect_to_ctr_sum).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True).sum().values
        num_detect_mw = pd.Series(num_detect).rolling(window=mv_avg_wd, min_periods=mv_min_num, center=True).sum().values

        d2skl_std = np.sqrt(d2skl_sum / (num_detect_mw - 1))
        d2skl_std_se = d2skl_std / np.sqrt(2 * (num_detect_mw - 1))
        d2ctr_std = np.sqrt(d2ctr_sum / (num_detect_mw - 1))
        d2ctr_std_se = d2ctr_std / np.sqrt(2 * (num_detect_mw - 1))

        total_detect = np.nansum(num_detect)
        
        result = {'stat_skl_p_t_avg': stat_p_to_ctr, 
                #   'num_p_per_f': num_detect, 
                #   'p_to_skl_dist_sum': detect_to_skl_sum, 
                #   'p_to_ctr_dist_sum': detect_to_ctr_sum, 
                #   'd2skl_std': np.sqrt(detect_to_skl_sum / (num_detect)), # the number is small - close to 1. Not sure if should subtract by 1
                #   'd2ctr_std': np.sqrt(detect_to_ctr_sum / (num_detect)),
                  'd2skl_std_sm': d2skl_std, 'd2skl_std_sm_se': d2skl_std_se, 
                  'd2ctr_std_sm': d2ctr_std, 'd2ctr_std_sm_se': d2ctr_std_se, 
                  'd2skl_std_avg': np.nansum(detect_to_skl_sum) / (total_detect - 1), 
                  'd2ctr_std_avg': np.nansum(detect_to_ctr_sum) / (total_detect - 1), 
                  'total_num_p': total_detect}
        
        if vis_Q: 
            vis_t = np.arange(num_t_pts)
            f = plt.figure(figsize=(8, 3))
            a = f.add_subplot()
            a.plot(vis_t, result['d2ctr_std'], label='dist2ctr')
            a.fill_between(vis_t, result['d2ctr_std'] + result['d2ctr_std_se'], 
                        result['d2ctr_std'] - result['d2ctr_std_se'], alpha=0.5)
            a.set_ylabel(f"cell to axial ctr dist std (pxl)")
            a.set_xlabel(f"Time (frame)")
            a.grid()
        
        return result
#endregion