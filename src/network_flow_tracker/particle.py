import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import pandas as pd
from IPython.display import display

from .utils import stat as stat
from .utils import filters as flt
from .utils import util
from .utils import vis

# import NFT.linking as NFTLinking # cannot do circular import

class Particle():
    def __init__(self, p_table:pd.DataFrame):
        self.detections = p_table
        self.t = []
        self.pos_t = []
        self.feature_t = []

    @property
    def first_frame(self):
        return self.detections.frame.values[0]
    
    @property
    def last_frame(self):
        return self.detections.frame.values[-1]
        
    @property
    def num_frame(self):
        return self.detections.shape[0]
    
    @property
    def is_continuous_Q(self):
        return np.all(np.diff(self.detections.frame.values) == 1)

    @property
    def pos_zyx(self):            
        return self.detections[['z', 'y', 'x']].to_numpy()

    @property
    def v_norm(self):
        v_ctr_diff = flt.central_difference_2d(self.pos_zyx, axis=0)
        return np.sqrt(np.sum(v_ctr_diff ** 2, axis=1))
    
    @property
    def v_norm_fd(self):
        return np.sqrt(np.sum(np.diff(self.pos_zyx, axis=0) ** 2, axis=1))

    @property
    def cos_adj_disp_vec(self):
        return compute_cos_disp_vec(self.pos_zyx, step=1)
    
    def cos_disp_vec_smooth(self, disp_pxl=2, max_wd_sz=3):
        wd_sz = np.minimum(max_wd_sz, np.ceil(disp_pxl / self.avg_e2e_v_norm)) 
        pos_zyx = self.pos_zyx
        if wd_sz > 1: 
            sm_kernel = np.ones((int(wd_sz), 1)) / wd_sz
            pos_zyx = sp.signal.convolve2d(pos_zyx, sm_kernel, mode='same')
        return compute_cos_disp_vec(pos_zyx, step=1)

    @property
    def pos_to_t0(self):
        pos = self.pos_zyx
        return pos - pos[0]
    
    @property
    def dist_to_t0(self):
        return np.sqrt(np.sum(self.pos_to_t0 ** 2, axis=1))
    
    @property
    def dist_to_median_pos(self):
        pos = self.pos_zyx
        med_pos = np.median(pos, axis=0)
        return np.sqrt(np.sum((pos - med_pos) ** 2, axis=1))
    
    @property
    def d_int(self):
        p_int = self.detections.peak_int.values.astype(np.float32)
        return np.diff(p_int)

    @property
    def d_int_n(self):
        p_int = self.detections.peak_int.values.astype(np.float32)
        if p_int.size > 1: 
            p_int_change = np.diff(p_int) / p_int[:-1]
        else: 
            p_int_change = np.asarray([])
        return p_int_change
    
    @property
    def avg_e2e_v_norm(self):
        return self.dist_e2e / self.num_frame
    
    @property
    def disp_e2e(self):
        # Total displacement vector
        endpoints = self.pos_zyx[[0, -1]]
        return endpoints[1] - endpoints[0]
    
    @property
    def dist_e2e(self):
        # total net distance traveled
        return np.sqrt(np.sum(self.disp_e2e ** 2))
    
    @property
    def dist_tot(self):
        return np.sum(self.v_norm_fd)
    
    @property
    def avg_v_norm(self):
        return np.mean(self.v_norm)

    @property
    def avg_peak_nb_snr(self):
        return self.detections.peak_nb_snr.mean()
    
    @property
    def nearest_edge_label(self):
        if 'edge_label' in self.detections:
            label = np.unique(self.detections.edge_label.values)
            return label
        else: 
            return -1
        
    @property
    def linking_cost_ratio(self):
        """ Ratio between the linking cost and the alternative linking cost. 
        < 1: linked to the nearest neighbor
        """
        return self.detections.cost.values / self.detections.alt_cost.values
    
    @property
    def exit_network_Q(self):
        return self.exit_ind is not None
    
    @property
    def exit_ind(self):
        if 'exit_ind' in self.detections: 
            ind = int(self.detections.exit_ind.values[-1])
            ind = None if ind < 0 else ind
        else: 
            ind = None
        return ind
    
    def pos_zyx_w_exit_ep(self, mask_size):
        exit_ind = self.exit_ind
        if exit_ind is None: 
            return self.pos_zyx
        else: 
            exit_sub = np.unravel_index(exit_ind, mask_size)
            return np.vstack((self.pos_zyx, exit_sub))

    def get_neighbors(self, self_t, nb_t, lk_hdl, num_nb):
        self_t_idx = np.nonzero(self.detections.frame.values == self_t)[0]
        nb_dist, _, nb_did = lk_hdl.get_neighbors(nb_t, self.pos_zyx[self_t_idx], num_nb)
        return nb_dist, nb_did.flatten()



    def vis_traces(self, vxl2f_to_mm2s=None, view_angles=None):
        frame = self.detections.frame.values
        if frame.size == 1: 
            print(f"Particle is only detected in 1 frame. Unable to plot traces")
            return

        mid_frame = (frame[1:] + frame[:-1]) / 2
        f = plt.figure(figsize=(20, 3))
        a1 = f.add_subplot(1, 4, 1)
        a1.plot(frame, self.detections.peak_int.values, color='b')
        a1.set_ylabel("Intensity", color='b')
        a1.set_xlabel("Frame")
        a1.set_title(f"Avg SNR {self.avg_peak_nb_snr:.2f}")
        a1.set_ylim(0, 2 ** 16 - 1)
        a11 = a1.twinx()
        rel_int_c = self.d_int_n
        
        a11.plot(mid_frame, rel_int_c, color='r')
        max_r_c = np.minimum(5, np.max(np.abs(rel_int_c)))
        if  max_r_c < 1:
            a11.set_ylim(-1, 1)
        else: 
            a11.set_ylim(-1.05, max_r_c)

        a11.set_ylabel("Rel Int Change", color='r')

        a2 = f.add_subplot(1, 4, 2, projection='3d')
        pos_zyx = self.pos_zyx
        a2.plot(pos_zyx[:, 2], pos_zyx[:, 1], pos_zyx[:, 0])
        scall_hdl = a2.scatter(pos_zyx[1:, 2], pos_zyx[1:, 1], pos_zyx[1:, 0], s=10, alpha=0.5)
        sc_hdl = a2.scatter(pos_zyx[0, 2], pos_zyx[0, 1], pos_zyx[0, 0], color='r', s=10, label='start', alpha=0.5)
        a2.legend()
        a2.set_title(f"Total travel: {self.dist_tot:.2f}")
        a2.set_xlabel("x")
        a2.set_ylabel("y")
        a2.set_zlabel("z")
        a2.invert_zaxis()
        if view_angles is not None: 
            a2.view_init(elev=view_angles[0], azim=view_angles[1])

        a3 = f.add_subplot(1, 4, 3)
        disp = self.v_norm
        if vxl2f_to_mm2s is not None: 
            disp = disp * vxl2f_to_mm2s
        a3.plot(frame, disp, color='b')
        ly_label = "ds" if vxl2f_to_mm2s is None else "v (mm/s)"
        a3.set_ylabel(ly_label, color='b')
        a3.set_xlabel("Frame")
        a3.set_title(f"Avg net v: {self.avg_e2e_v_norm:.2e}, Stat. {self.is_stationary_Q()}")
        a3.set_ylim(0,  np.ceil(np.max(disp)))
        a31 = a3.twinx()
        a31.plot(frame, self.dist_to_t0, color='r')
        a31.set_ylabel("r(t) - (0)", color='r')


        a4 = f.add_subplot(1, 4, 4)
        cos_vv_sm = self.cos_adj_disp_vec
        a4.plot(frame, cos_vv_sm)
        a4.set_xlabel("Frame")
        a4.set_ylabel("Cos<vt, vtp1>")
        a4.set_ylim(-1.05, 1.05)
        a4.set_title(f"Avg cos: {cos_vv_sm.mean():.2f}")
        a4.grid()
        f.tight_layout()

        return f

    def vis_traj_w_mip(self, vsl_mip):
        f = plt.figure(figsize=(8, 8))
        ax = f.add_subplot()
        ax.imshow(vsl_mip, cmap='gray')
        pos = self.pos_zyx
        ax.scatter(pos[:, 2], pos[:, 1], 1, color='r')
        return f

    def vis_traj_w_mips(self, vsl_mips):
        f = vis.vis_mips(vsl_mips, self.pos_zyx.T, vis_line_Q=True)
        return f


    def get_neighbor_points(self, particles:pd.DataFrame, max_dist, ctr_pos=None):
        if ctr_pos is None: 
            ctr_pos = self.pos_zyx[0]

        points_pos_zyx = particles[['z', 'y', 'x']].to_numpy()
        dist_to_vis_ctr = np.sqrt(np.sum((points_pos_zyx - ctr_pos) ** 2, axis=1)) 

        return points_pos_zyx[dist_to_vis_ctr < max_dist]
    
    def merge_with(self, p2):
        if isinstance(p2, pd.DataFrame):
            p2 = Particle(p2)
        else:
            assert(isinstance(p2, Particle))
        if not overlap_in_time_Q(self, p2):
            self.detections = pd.concat((self.detections, p2.detections)).sort_values(by='frame')
        else: 
            raise ValueError("The two particles have overlap in time")
    
    def get_trajectory_endpoint(self, output='idx', excluded_frame=None):
        frame = self.detections.frame.values
        tmp_mask = - np.ones((frame[-1] + 1,), dtype=np.int32)
        tmp_mask[frame] = np.arange(0, frame.size, 1, dtype=np.int32)
        endpoint_idx = util.get_intervals_in_1d_binary_array(tmp_mask >= 0, including_end_Q=True).flatten()
        endpoint_idx = tmp_mask[endpoint_idx]
        endpoint_frame = frame[endpoint_idx]
        if excluded_frame is not None: 
            selected_idx = [i for i, f in enumerate(endpoint_frame) if f not in excluded_frame]
            endpoint_frame = endpoint_frame[selected_idx]
            endpoint_idx = endpoint_idx[selected_idx]
        if output == 'idx':
            return endpoint_idx
        elif output =='frame':
            return endpoint_frame
        elif output == 'did':
            return self.detections.did.values[endpoint_idx]

    def split_trajectory_by_total_dist(self, max_dist):

        split_point = np.nonzero(self.dist_to_t0 >= max_dist)[0]

        split_point = np.concatenate(([0], split_point, [self.num_frame])).astype(np.uint64)
        new_tables = []
        for i, j in zip(split_point[:-1], split_point[1:]):
            new_tables.append(self.detections.iloc[i : j].copy())
        return new_tables
            
    def trajectory_outlier_features(self):
        features = {'max_d_int': np.max(np.abs(self.d_int)), \
                    'max_d_int_n': np.max(np.abs(self.d_int_n))}
        return features
    
    def get_trajectory_mask_by_adj_v_cos(self, min_cos=-0.5, ignore_max_v=None, smooth_Q=False):
        if smooth_Q:
            valid_interval_Q = self.cos_disp_vec_smooth() >= min_cos    
        else:
            valid_interval_Q = self.cos_adj_disp_vec >= min_cos
        if ignore_max_v is not None: 
            assert ignore_max_v >= 0, 'ignore_max_v must be a non-negative scalar'
            valid_interval_Q = np.logical_or(valid_interval_Q, self.v_norm <= ignore_max_v)
        return valid_interval_Q
    
    def get_trajectory_mask_w_min_cost(self, threshold_ratio=1):
        cost_ratio = self.detections.cost / self.detections.alt_cost
        return cost_ratio < threshold_ratio

    def select_sub_trajectory_by_adj_v_cos(self, min_cos=-0.5, ignore_max_v=None, inplace_update_Q=True, smooth_Q=False):
        # This needs to be improved
        # Deal with small movements
        # -0.5: - 60 deg
        valid_interval_Q = self.get_trajectory_mask_by_adj_v_cos(min_cos=min_cos, ignore_max_v=ignore_max_v, smooth_Q=smooth_Q)
        return self.select_largest_sub_trajectory_by_masking(valid_interval_Q, inplace_update_Q=inplace_update_Q)
        
    def select_sub_trajectory_by_int_traces(self, max_dint=5e3, min_dint_n=-0.66, 
                                            max_dint_n=0.75, inplace_update_Q=True):
        d_int = self.d_int
        d_int_n = self.d_int_n
        is_valid_Q = np.logical_or(np.abs(d_int) < max_dint, 
                                   d_int_n < max_dint_n)
        is_valid_Q = np.logical_and(is_valid_Q, d_int_n > min_dint_n)

        return self.select_largest_sub_trajectory_by_masking(is_valid_Q, inplace_update_Q=inplace_update_Q)
        
    def select_largest_sub_trajectory_by_masking(self, is_valid_Q, inplace_update_Q=False):
        idx_start, idx_end = util.get_largest_interval_in_1d_array(is_valid_Q)
        if idx_start is not None:
            new_table = self.detections.iloc[int(idx_start) : int(idx_end)]
        else: 
            new_table = self.detections.iloc[0:0]

        if inplace_update_Q:
            self.detections = new_table

        return new_table
    
    def remove_outliers_in_stationary_trace(self, ipr=1.5, min_dist=1):
        # Position seems to be more reliable. 
        # When the spacing between a moving particle and a stationary particle 
        # becomes less than the minimum spacing for peak detection, only one will be 
        # detected. This will cause a sudden shift in position and a abrupt increase in 
        # inter-frame velosity. However, the duration of this shift depends on particle speed
        # and therefore velocity outlier might not be very reliable. 
        # on the other hand, outlier rejection based on position assume the stationary particle 
        # has little displacement over time. 
        dist_2_med = self.dist_to_median_pos
        inlier_range = stat.compute_percentile_outlier_threshold(dist_2_med, ipr=ipr)
        dist_outlier_idx = np.nonzero(dist_2_med > np.maximum(min_dist, inlier_range[1]))[0]

        v_n = self.v_norm
        v_n_range = stat.compute_percentile_outlier_threshold(v_n, ipr=ipr)
        v_outlier_idx = np.nonzero(v_n > v_n_range[1])[0] + 1
        v_outlier_idx
    
    def is_stationary_Q(self, max_disp=2, max_v_norm=1.0):
        # Need to account for motion artefact ???

        return (self.dist_e2e <= max_disp) and (np.mean(self.v_norm) < max_v_norm)
    
    def select_sub_trajectory_by_dist_to_med_pos(self, max_dist=5, min_dist=1.5, ipr=1.5, inplace_update_Q=True):
        # Only works for stationary points 
        dist_2_med_pos = self.dist_to_median_pos
        dist_range = stat.compute_percentile_outlier_threshold(dist_2_med_pos, ipr=ipr)
        max_dist = max(min_dist, min(max_dist, dist_range[1]))
        is_valid_Q = (dist_2_med_pos < max_dist)
        return self.select_largest_sub_trajectory_by_masking(is_valid_Q, inplace_update_Q=inplace_update_Q)

    def is_good_trace_Q(self):
        cos_vv = self.cos_adj_disp_vec
        good_Q = np.mean(cos_vv) > 0.5
        
        return good_Q
    
    def compute_2nn_margin(self, lk_hdl):
        """
        Arguments:
            lk_hdl: an instance of NFTLinking.Linking
        """
        num_nb = 2 # this can cause problem - when the connected point are not among the 2 nearest neighbors
        frames = self.detections.frame.values
        num_pair = frames.size - 1
        connected_nn_Q = np.zeros(num_pair, bool)
        dist_margin = np.zeros(num_pair, dtype=np.float32)
        link_dist = np.zeros(num_pair, dtype=np.float32)

        tmp_pos = self.pos_zyx
        for i, t in enumerate(frames[:-1]):
            try: 
                tmp_nb_dist, _, tmp_nb_did = lk_hdl.get_neighbors(t + 1, tmp_pos[i], num_nb)
                tmp_connected_idx = np.nonzero(self.detections.iloc[i + 1].did == tmp_nb_did)[0] 
                connected_nn_Q[i] = (tmp_connected_idx == 0)
                dist_margin[i] = tmp_nb_dist[1] - tmp_nb_dist[0]
                if tmp_connected_idx.size == 1: 
                    link_dist[i] = tmp_nb_dist[tmp_connected_idx]
                elif tmp_connected_idx.size == 0: 
                    link_dist[i] = np.nan
                else: 
                    raise ValueError("More than 1 nearest neighbors share the same detection id.")
            except Exception as e: 
                print(f"Encounter error in the {i}-th frame of t = {t}")
                raise e
        return connected_nn_Q, dist_margin, link_dist
    
    def analyze_alternative_connection(self, lk_hdl, verboseQ=False):
        """
        Arguments:
            lk_hdl: an instance of NFTLinking.Linking
        """
        connected_nn_Q, dist_margin, link_dist = self.compute_2nn_margin(lk_hdl)
        result = {'connected_nn_Q': connected_nn_Q, 
                  'next_nb_dist': dist_margin, 
                  'link_length': link_dist}
        result['always_nn_Q'] = np.all(connected_nn_Q)
        result['min_margin'] = np.min(dist_margin)
        result['avg_margin'] = np.mean(dist_margin)
        result['max_link_length'] = np.max(link_dist)
        result['inlier_range'] = stat.compute_percentile_outlier_threshold(link_dist)
        if verboseQ:
            print(f"Number of frame tracked: {self.num_frame}")
            print(f"Always connected to the nearest neighbor: {result['always_nn_Q']}")
            print(f"Minimal next nearest dist: {result['min_margin']:.2f}")
            print(f"Average next nearest dist: {result['avg_margin']:.2f}")
            print(f"Maximum inter-frame displacement: {result['max_link_length']:.2f}")
            print(f"Inlinear displacement range: {np.round(result['inlier_range'], decimals=2)}")
        return result
            
    def analyze_within_edge_dir_consistency(self):
        if 'edge_label' in self.detections:
            e_idx, el = util.bin_data_to_idx_list(self.detections.edge_label.values)
            p_e_valid_Q = (el >= 0)
            el = el[p_e_valid_Q]
            e_idx = e_idx[p_e_valid_Q]

def overlap_in_time_Q(p1:Particle, p2:Particle):
    if p1.is_continuous_Q and p2.is_continuous_Q:
        return (p1.last_frame >= p2.first_frame and p2.last_frame >= p1.first_frame )
    else: 
        return (np.intersect1d(p1.detections.frame.values, p2.detections.frame.values).size > 0)

def merge_particles(p1:Particle, p2:Particle, output=None):
    if not overlap_in_time_Q(p1, p2):
        # Not sure if testing continuity then concatenate directly would save a lot of time
        new_table = pd.concat((p1.detections, p2.detections)).sort_values(by='frame')
        if output is None: 
            return Particle(new_table)
        else: 
            output.table = new_table
    else: 
        raise ValueError("The two particles have overlap in time")

def dist_tzyx(p1:Particle, p2:Particle):
    tzyx_key = ['frame', 'z', 'y', 'x']
    result = np.full((4, ), np.nan)
    if not overlap_in_time_Q(p1, p2):
        if p1.last_frame < p2.first_frame:
            p1_tzyx = p1.detections[tzyx_key].iloc[-1].to_numpy()
            p2_tzyx = p2.detections[tzyx_key].iloc[0].to_numpy()
        elif p2.last_frame < p1.first_frame: 
            p2_tzyx = p2.detections[tzyx_key].iloc[-1].to_numpy()
            p1_tzyx = p1.detections[tzyx_key].iloc[0].to_numpy()
        else: 
            raise ValueError(f"p1: {p1.first_frame} to {p1.last_frame}; p2: {p2.first_frame} to {p2.last_frame}")
        result = p2_tzyx - p1_tzyx
    return result

def dist_ts(p1:Particle, p2:Particle):
    tzyx = dist_tzyx(p1, p2)
    tzyx[1] = np.sqrt(np.sum(tzyx[1:] ** 2))
    return tzyx[:2]


def compute_particle_feature_similarity(p1:pd.DataFrame, p2:pd.DataFrame, compare_feature=('peak_int', 'eig1'), 
                       verboseQ=False, method='relative_change'):
    prob = 1
    for f in compare_feature:
        val_1 = p1[[f]].values
        val_2 = p2[[f]].values
        if method == 'relative_change':
            prob *= np.exp(-np.abs(val_2 - val_1) / np.abs(val_1)) 
        elif method == 'relative_difference':
            prob *= np.exp(-stat.relative_difference(val_1, val_2)) 

    if verboseQ:
        display(p1)
        display(p2)
        print(f"Compared feature: {compare_feature}")
        print(f"Probability: {prob.flatten()}")

    return prob

def compute_cos_disp_vec(pos_zyx, step=1):
    """
    Args: 
        pos_zyx: (n, ndim) numpay array 
        step: integer scalar
    """
    disp = np.diff(pos_zyx[::step, :], axis=0)
    if disp.shape[0] >= 2 : 
        # At least 3 points 
        vec_n = np.sqrt(np.sum(disp ** 2, axis=1))
        disp = disp / vec_n[:, None]
        cos_val = np.sum(disp[:-1, :] * disp[1:, :], axis=1)
        return np.concatenate(([cos_val[0]], cos_val, [cos_val[-1]]))
    else: 
        return np.full((disp.shape[0] + 1, ), np.nan)
    
def compute_direction_consistent_subtraces(trace_result, t_idx, mask_size, min_length=3, min_med_cos=0.75, min_cos=0, max_ignore_v=2): 
    valid_traces_ind = []

    tmp_p = Particle(trace_result.iloc[t_idx])
    added_ep_Q = False
    tmp_sub = tmp_p.pos_zyx
    if tmp_p.exit_network_Q: 
        tmp_exit_Q = True
        exit_ind = tmp_p.exit_ind
        exit_sub = np.unravel_index(exit_ind, mask_size)
        tmp_sub_2_ep_dist = np.sqrt(np.sum((exit_sub - tmp_sub[-1]) ** 2)) 
        if tmp_sub_2_ep_dist > 2 * max_ignore_v: 
            added_ep_Q = True
            tmp_sub = np.vstack((tmp_sub, exit_sub))
    else: 
        tmp_exit_Q = False

    v_ctr_diff = flt.central_difference_2d(tmp_sub, axis=0)
    tmp_v = np.sqrt(np.sum(v_ctr_diff ** 2, axis=1))
    tmp_cos = compute_cos_disp_vec(tmp_sub, step=1)
    tmp_t_msk = (tmp_cos >= min_cos)
    if np.mean(tmp_v) < max_ignore_v: 
        tmp_t_msk = np.logical_or(tmp_t_msk, (tmp_v > max_ignore_v))
    if np.any(tmp_t_msk) and (np.median(tmp_cos[tmp_t_msk]) > min_med_cos): 
        if added_ep_Q: 
            tmp_t_msk[-1] = False
        tmp_t_ints = util.get_intervals_in_1d_binary_array(tmp_t_msk)
        tmp_t_int_len = tmp_t_ints[:, 1] - tmp_t_ints[:, 0]
        if tmp_exit_Q and tmp_t_int_len.size > 0: 
            if (added_ep_Q and tmp_t_ints[-1, 1] == (tmp_t_msk.size - 1)) : 
                tmp_t_int_len[-1] += 1    
            elif tmp_t_ints[-1, 1] == tmp_t_msk.size: 
                tmp_t_int_len[-1] += 1
        
        tmp_selected_int_Q = (tmp_t_int_len >= min_length)
        tmp_t_ints = tmp_t_ints[tmp_selected_int_Q]      
        for tmp_int in tmp_t_ints: 
            valid_traces_ind.append(t_idx[tmp_int[0] : tmp_int[1]])
    
    return valid_traces_ind
    

def vis_trajectory_with_mip(trace_result, trace_ind, vsl_im_mip, vis_method='dot', figsize=(8, 8)):
    f = plt.figure(figsize=figsize)
    a = f.add_subplot()
    a.imshow(vsl_im_mip, cmap='gray')
    for i, ind in enumerate(trace_ind):
        tmp_p = trace_result.iloc[ind]
        tmp_xy = tmp_p[['x', 'y']].to_numpy()
        if vis_method == 'dot':
            a.scatter(tmp_xy[:, 0], tmp_xy[:, 1], 0.5)
        elif vis_method == 'line':
            a.plot(tmp_xy[:, 0], tmp_xy[:, 1], linewidth=0.5)
    a.axis('off')
    return f