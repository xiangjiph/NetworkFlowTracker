import numpy as np 
import os

import scipy.ndimage as ndi
import scipy.sparse as ssp
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import pandas as pd

from .utils import io as io
from .utils import image as im
from .utils import neighbors as nb
from .utils import stat as stat
from .utils import filters as flt
from .utils import util as util
from .utils import graph as graph

from . import particle

#region Distributed functions

def dist_compute_single_stack_cell_positions(root_fp: str, z_idx: int, t_idx: int, para: dict, saveQ=True, overwriteQ=False):
    process_data_root = os.path.join(root_fp, 'processed_data')
    info_fp = os.path.join(process_data_root, 'data_info.pickle')
    data_info = io.load_data(info_fp)
    lfp = LFBFProcessing(root_fp, data_info)

    # Check file 
    fp = lfp.fp_cell_pos(z_idx, t_idx)
    if (not os.path.isfile(fp)) or overwriteQ:
        im_vol = lfp.load_and_preprocess_raw_data_to_match_mask(z_idx, t_idx) # (75, 400, 400), circular masked
        nb_features, log_stat = Detection.process_single_volume(im_vol, para)
        data_info = {'para' : para, 'log' : log_stat, 'z' : z_idx, 't' : t_idx}
        nb_features['info'] = data_info
        if saveQ:
            io.save_data(fp, nb_features)
    else:
        print(f"File {fp} already exist. Load existing files")
        nb_features = io.load_data(fp)
    return nb_features

#endregion

class LFBFProcessing:
    def __init__(self, root_fp, info:dict, verboseQ=True):
        info['raw_data_root'] = os.path.join(root_fp, 'raw_data')
        info['processing_data_root'] = os.path.join(root_fp, 'processed_data')
        info['vis_file_root'] = os.path.join(root_fp, 'visualization')
        self.info = info
        self.verboseQ = verboseQ

    def fp_raw_data(self, z_idx, t_idx):
        return os.path.join(self.info['raw_data_root'], self.info['raw_data_folders'][z_idx], self.info['raw_data_file'][z_idx][t_idx])
    
    def fp_cell_pos(self, z_idx, t_idx):
        fn, _ = os.path.splitext(self.info['raw_data_file'][z_idx][t_idx])
        t_fn = f"{fn}.h5"
        return os.path.join(self.info['processing_data_root'], 'cell_pos', self.info['raw_data_folders'][z_idx],
                             t_fn)

    def load_and_preprocess_raw_data_to_match_mask(self, z_idx=None, t_idx=None, fp=None):
        if fp is None: 
            fp = self.fp_raw_data(z_idx, t_idx)
        data = io.load_data(fp, [self.info['bbox_mm'][0], self.info['bbox_xx'][0]])

        bbox_mm = self.info['bbox_mm']
        bbox_xx = self.info['bbox_xx']
        target_zoom = [v / self.info['target_voxel_size_um'] for v in self.info['voxel_size_um']]
        data = self._preprocess_raw_data(data, bbox_mm, bbox_xx, target_zoom, slice_z_Q=False)
        # scikit-image resize is based on scipy.ndimage.zoom but normalize the data to [0, 1]
        # target_shape = [int(s * v / self.info['target_voxel_size_um']) for s, v in zip(new_data.shape, self.info['voxel_size_um'])]
        # new_data = resize(new_data, target_shape, anti_aliasing=True)
        return data

#endregion cell detection 

#region Point linking 
    def load_and_preprocess_cell_pos(self, z_idx, t_idx, valid_z_range=None):
        pos_fp = self.fp_cell_pos(z_idx, t_idx)
        data = io.load_data(pos_fp)
        data['sub'] = []
        for ind in data['ind']:
            tmp_sub = np.unravel_index(ind, data['mask_shape'])
            if valid_z_range is not None: 
                is_valid_Q = np.logical_and(tmp_sub[0] >= valid_z_range[0], \
                                            tmp_sub[0] <= valid_z_range[1])
                tmp_sub = [sub[is_valid_Q] for sub in tmp_sub]
            data['sub'].append(tmp_sub)
        return data

#endregion
    
    @staticmethod
    def _preprocess_raw_data(data, bbox_mm, bbox_xx, target_zoom, slice_z_Q=True):

        data_type = data.dtype
        # Z-slicing done upon data loading
        if slice_z_Q: 
            data = data[bbox_mm[0] : bbox_xx[0], bbox_mm[1] : bbox_xx[1], bbox_mm[2] : bbox_xx[2]]
        else:
            data = data[:, bbox_mm[1] : bbox_xx[1], bbox_mm[2] : bbox_xx[2]]

        circ_mask = Utility.construct_circular_mask(data.shape[1:3], r=data.shape[1]/2)    
        data = (data.astype(np.float32) * circ_mask).astype(data_type)
        data = ndi.zoom(data, target_zoom, order=1)
        return data
    

    @staticmethod
    def load_and_parse_annotation_data(stitch_data_fp):
        stitch_data = io.read_mat_file_as_h5(stitch_data_fp, reorder_dimensions=False)
        # Process stitching data: 
        py_axis_in_mat = [2, 0, 1]
        stitch_data['skl_sub'] = (stitch_data['skl_sub'][py_axis_in_mat] - 1).astype(np.int32) # MATLAB indices start from 1, shape (3, N)
        stitch_data['mask_size'] = stitch_data['mask_size'].flatten()[py_axis_in_mat].astype(np.int64)
        stitch_data['disp_vec_c'] = stitch_data['disp_vec_c'][:, [2, 0, 1]] # (num_vol, 3)
        stitch_data['disp_vec'] = stitch_data['disp_vec'][:, [2, 0, 1]] # (num_vol, 3)
        stitch_data['skl_ind'] = np.ravel_multi_index([stitch_data['skl_sub'][i] for i in range(stitch_data['skl_sub'].shape[0])], 
                                                        stitch_data['mask_size'])

        # This vascular image has been high-pass-filtered. Large vessel might have black reigon inside. 
        for k in ['cc_label_array', 'label_array_annotated', 'label_array_recon', 'stitched_im', 'stitched_mask']: 
            # for 3d array, somehow the z axis has already been placed to the 0-th axis in np.array
            stitch_data[k] = stitch_data[k].transpose((0, 2, 1)) 

        # Make sure skeleton label is consistent with the label_array_annotated
        # skl_label_annotated = stitch_data['label_array_annotated'].flat[stitch_data['skl_ind']]
        # inconsistent_artery_idx = np.nonzero(np.logical_and(skl_label_annotated == 1, stitch_data['skl_label'] == 2))[0]
        # inconsistent_vein_idx = np.nonzero(np.logical_and(skl_label_annotated == 1, stitch_data['skl_label'] == 3))[0]
        # if inconsistent_artery_idx.size > 0: 
        #     print(f"{inconsistent_artery_idx.size} voxels is labeled as artery in skl_ind but capillary in labeld_array_annotated. Correct to be capillary.")
        #     stitch_data['skl_label'][inconsistent_artery_idx] = 1
        # if inconsistent_vein_idx.size > 0: 
        #     print(f"{inconsistent_vein_idx.size} voxels is labeled as vein in skl_ind but capillary in labeld_array_annotated. Correct to be capillary.")
        #     stitch_data['skl_label'][inconsistent_vein_idx] = 1
        # stitch_data['skl_label']
                
        return stitch_data

    @staticmethod
    def get_subvol_data(data_info, stitch_data, z_idx, rm_node_vxl_on_boundary_Q=True): 
        extra_z = 0 # not used. 
        sv = {}
        sv['mask_size'] = (data_info['bbox_ll'] * data_info['voxel_size_um'] / data_info['target_voxel_size_um']).astype(np.int16)
        # select voxels in range
        ## sub-volume bounding box
        z_range_0 = np.asarray(data_info['z_valid_range_um'][z_idx]) / data_info['target_voxel_size_um']
        z_disp_vec = stitch_data['disp_vec_c'][z_idx]
        z_bbox_mm_r = (np.asarray([z_range_0[0], 0, 0]) + z_disp_vec).astype(np.int16)
        z_bbox_xx_r = (np.asarray([z_range_0[1], sv['mask_size'][1], sv['mask_size'][2]]) + z_disp_vec).astype(np.int16)

        z_in_range_Q = np.logical_and(stitch_data['skl_sub'][0] >= (z_bbox_mm_r[0] - extra_z), 
                                      stitch_data['skl_sub'][0] < (z_bbox_xx_r[0] + extra_z)) 

        sv['skl_label_v'] = stitch_data['skl_label'].flatten()[z_in_range_Q]
        sv['skl_r_v'] =  stitch_data['skl_r_pxl'].flatten()[z_in_range_Q]
        subvol_skl_sub = stitch_data['skl_sub'][:, z_in_range_Q].astype(np.int16)
        subvol_skl_sub[0] -= z_bbox_mm_r[0]
        sv['skl_ind'] = np.ravel_multi_index((subvol_skl_sub[0], subvol_skl_sub[1], subvol_skl_sub[2]), sv['mask_size'])
        sv['skl_ind_g'] = stitch_data['skl_ind'][z_in_range_Q]
        
        sv['im'] = stitch_data['stitched_im'][z_bbox_mm_r[0] : z_bbox_xx_r[0], :, :]
        sv['label_array'] = stitch_data['label_array_annotated'][z_bbox_mm_r[0] : z_bbox_xx_r[0], :, :]
        sv['disp_vec'] = np.asarray([z_bbox_mm_r[0], 0, 0]) # we translate the detections to the stitched volume space later
        ## Remove node voxel near the edge
        if rm_node_vxl_on_boundary_Q: 
            
            whole_graph = graph.SpatialGraph(stitch_data['skl_ind'], stitch_data['mask_size'])
            # Get node voxel on the edge 
            whole_graph_node_sub = whole_graph.ind2sub(whole_graph.node.pos_ind)
            node_on_subvol_edge_Q = np.logical_or(whole_graph_node_sub[0] == z_bbox_mm_r[0], 
                                                  whole_graph_node_sub[0] == (z_bbox_xx_r[0] - 1))
            n_vxl_sub = np.asarray([sub[node_on_subvol_edge_Q] for sub in whole_graph_node_sub])
            n_vxl_sub[0] -= z_bbox_mm_r[0]
            subvol_edge_node_ind = np.ravel_multi_index((n_vxl_sub[0], n_vxl_sub[1], n_vxl_sub[2]),
                                                         sv['mask_size'])
            selected_Q = np.ones(sv['skl_ind'].shape, bool)
            for i in subvol_edge_node_ind: 
                selected_Q[sv['skl_ind'] == i] = False
            if not np.all(selected_Q):
                for k in ['skl_label_v', 'skl_r_v', 'skl_ind', 'skl_ind_g']: 
                    sv[k] = sv[k][selected_Q]
                print(f"Remove {np.count_nonzero(~selected_Q)} node voxels on the sub-volume boundary")
        
        sv['skl_sub'] = np.vstack(np.unravel_index(sv['skl_ind'], sv['mask_size']))
        return sv
#region Visualization 


#endregion
class Detection:
    @staticmethod
    def process_single_volume(im_vol, para:dict):
        """
        Args: 
            im_vol: 3D numpy array 
            para: dict
        """
        
        # High pass filtering 
        data_hp = flt.dog(im_vol.astype(np.float32), para['dog_sig1'], para['dog_sig2'])
        data_hp = np.maximum(data_hp, 0)
        if 'bg_min_int' not in para: 
            para['bg_min_int'] = 0
        # Detect local maxima
        valid_lm_ind, info = Detection.detect_local_maxima(
            data_hp, para['bg_vxl_f'], para['bg_max_int'], para['min_peak_snr'], 
            para['min_peak_dist'], para['peak_int_diff_sm_wd'], para['peak_int_diff_th'], 
            para['bg_est_sample_step'], bg_min_int=para['bg_min_int'])

        # Which volume should be used for computing the features? Raw or high-pass-filtered?
        nb_features = Detection.compute_features(valid_lm_ind, im_vol, para['nb_wd_r'], para['nb_bg_min_dist'])
        return nb_features, info
    
    @staticmethod
    def detect_local_maxima(im_vol, bg_vxl_f, bg_max_int, min_peak_snr, min_peak_dist, 
                            peak_int_diff_sm_wd=10, peak_int_diff_th=1, bg_est_sample_step=10, bg_min_int=0, exclude_border_Q=False):
        
        # Assuming im_vol was from uint16 data, peak_int_diff_th = 1, which is 
        # the minimal difference between voxel intensity

        # Estimate background level 
        circ_mask = Utility.construct_circular_mask(im_vol.shape[1:3])
        circ_mask = np.repeat(circ_mask[None, :, :], im_vol.shape[0], axis=0)
        info = stat.estimate_background_by_fraction(im_vol, bg_vxl_f, bg_est_sample_step, circ_mask)
        info['est_th'] = max(bg_min_int, min(bg_max_int, info['mean'] + info['std'] * min_peak_snr))
        # Not quite sure if the border local max should be excluded or not
        lm_pos = peak_local_max(im_vol, min_distance=min_peak_dist, 
                                       threshold_abs=info['est_th'], exclude_border=exclude_border_Q)
        lm_pos = lm_pos.transpose()
        lm_ind = np.ravel_multi_index((lm_pos[0], lm_pos[1], lm_pos[2]), im_vol.shape)
        info['num_detected'] = lm_ind.size
        # Select local maxima
        # sort intensity in desending order
        lm_int = im_vol.flat[lm_ind]
        lm_int_idx = np.argsort(lm_int)[::-1]
        lm_int = lm_int[lm_int_idx]

        lm_int_diff = - np.diff(lm_int)
        lm_int_diff_sm = np.convolve(lm_int_diff, np.ones(peak_int_diff_sm_wd) / peak_int_diff_sm_wd, 'same')
        num_lm = np.nonzero(lm_int_diff_sm < peak_int_diff_th)[0][0] # Get the first 
        info['int_th'] = lm_int[num_lm]

        valid_lm_ind = lm_ind[lm_int_idx[:num_lm]]
        info['num_selected'] = valid_lm_ind.size

        info['mask_size'] = np.array(im_vol.shape)
        return valid_lm_ind, info

    @staticmethod
    def compute_features(peak_ind, im_vol, nb_wd_r, nb_bg_min_dist):

        # Compute sampling voxel indices 
        nb_wd_r = np.array(nb_wd_r) if isinstance(nb_wd_r, (list, tuple)) else nb_wd_r
        sample_wd_d = 2 * nb_wd_r + 1
        sample_kernel = np.ones(sample_wd_d, bool)
        dind, dsub = nb.compute_kernel_offset_pos_and_ind_in_padded_mask(sample_kernel, im_vol.shape)
        dsub_a = np.vstack(dsub)
        # Determine background voxels in the sampled volume
        # sample_bg_mask = np.any(np.abs(dsub_a) > 1, axis=0) 
        sample_bg_mask = (np.sqrt(np.sum(dsub_a ** 2, axis=0)) >= nb_bg_min_dist)

        im_vol_pad = np.pad(im_vol, ((nb_wd_r[0], nb_wd_r[0]), 
                                     (nb_wd_r[1], nb_wd_r[1]), 
                                     (nb_wd_r[2], nb_wd_r[2])))
        valid_lm_ind_p = nb.compute_voxel_offseted_linear_indices(peak_ind, im_vol.shape, nb_wd_r)
        nb_ind = valid_lm_ind_p[:, None] + dind.flatten()[None, :]
        nb_int = im_vol_pad.flat[nb_ind].astype(np.float32) # (n, nb)

        features = {}
        features['peak_int'] = im_vol.flat[peak_ind]
        features['nb_mean'] = np.mean(nb_int, axis=1)        
        # SNR
        nb_bg_int = nb_int[:, sample_bg_mask]
        features['bg_mean'] = np.mean(nb_bg_int, axis=1)
        features['bg_std'] = np.std(nb_bg_int, axis=1)
        features['peak_nb_snr'] = (features['peak_int'] - features['bg_mean']) / features['bg_std']
        # Normalized background int
        nb_int /= np.sum(nb_int, axis=1)[:, None]
        tmp_com = np.sum(dsub_a[None, :, :] * nb_int[:, None, :], axis=2)# (n, 3)
        valid_lm_sub = np.unravel_index(peak_ind, im_vol.shape)
        features['pos'] = (np.vstack(valid_lm_sub) + tmp_com.transpose()).T
        features['peak_ind'] = peak_ind
        # Covariance matrix of the neighboring intensity profile
        nb_dist_to_com = dsub_a[None, :, :] - tmp_com[:, :, None] # (n, 3, nb)
        features['cov'] = np.einsum('nil,njl->nij', nb_dist_to_com * nb_int[:, None, :], nb_dist_to_com) # (n, 3, nb) x (n, 3, nb) -> (n, 3, 3)
        features['eig_val'], eig_vecs = np.linalg.eigh(features['cov'])
        features['eig_vec_0'] = eig_vecs[:, :, 0] # smallest eigenvalue
        features['eig_vec_1'] = eig_vecs[:, :, 1]
        features['eig_vec_2'] = eig_vecs[:, :, 2] # largest eigenvalue
        return features

    @staticmethod
    def convert_data_to_pandas_dataframe(tmp_data, mask_size=None):
        info = tmp_data['info']
        copy_keys = ('bg_mean', 'bg_std', 'nb_mean', 'peak_int', 'peak_nb_snr')
        tmp_df = {k : v for k, v in tmp_data.items() if k in copy_keys}
        tmp_pos = tmp_data['pos']
        tmp_df['x'] = tmp_pos[:, 2]
        tmp_df['y'] = tmp_pos[:, 1]
        tmp_df['z'] = tmp_pos[:, 0]
        tmp_sub = np.round(tmp_pos).astype(np.uint16)
        tmp_df['sub_0'] = tmp_sub[:, 0]
        tmp_df['sub_1'] = tmp_sub[:, 1]
        tmp_df['sub_2'] = tmp_sub[:, 2]
        if 'mask_size' in info['log']: 
            mask_size = info['log']['mask_size']
        if mask_size is not None:
            tmp_df['ind'] = np.ravel_multi_index((tmp_df['sub_0'], tmp_df['sub_1'], tmp_df['sub_2']), 
                                                 mask_size)
        tmp_df['eig1'] = tmp_data['eig_val'][:, 2] # largest 
        tmp_df['eig2'] = tmp_data['eig_val'][:, 1]
        tmp_df['eig3'] = tmp_data['eig_val'][:, 0] # smallest 
        tmp_df = pd.DataFrame(tmp_df)
        tmp_df['frame'] = info['t']
        return tmp_df, info
    
    @staticmethod
    def vis_detected_centroid(im_mip, sub, c_val=None, mip_t=None, figsize=(12, 12)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        if mip_t is not None: 
            ax.imshow(mip_t, cmap='gray')
            alpha = 0.5
        else:
            alpha = 1.0

        if c_val is None: 
            c_val = 1
        ax.imshow(im_mip, cmap='hot', alpha=alpha)
        ax.set_ylabel("Y (2 um)")
        ax.set_xlabel("X (2 um)")
        if c_val is None: 
            ax.scatter(sub[2], sub[1], s=1, color='g')
        else:
            sc = ax.scatter(sub[2], sub[1], s=1, c=c_val, cmap='winter', vmin=0, vmax=np.percentile(c_val, 95))
            fig.colorbar(sc, label='SNR')

        plt.show()
        fig.tight_layout()
        return fig, ax


#region Linking
class Linking: 
    def __init__(self, mask_size, bin_size):
        bin_size = np.array(bin_size) if not isinstance(bin_size, np.ndarray) else bin_size
        if bin_size.size == 1:
            bin_size = np.repeat(bin_size, mask_size.size)
        self.mask_size = mask_size
        self.bin_size = bin_size
        self.bin_mask_size = np.int64(np.round(mask_size / bin_size))

    def parse_pos_data(self, pos_data, min_peak_nb_snr=None):
        self.num_t = len(pos_data)
        self.data = []
        self.info = []
        self.min_peak_nb_snr = min_peak_nb_snr
        for data in pos_data:
            if isinstance(data, dict):
                if min_peak_nb_snr is not None: 
                    mask = data['peak_nb_snr'] >= min_peak_nb_snr
                    tmp_dict = util.mask_data_in_dict(data, mask)
                else:
                    tmp_dict = data.copy()
                if 'info' in tmp_dict:
                    info = tmp_dict['info']
                    del tmp_dict['info']
                    self.info.append(info['log'])
                self.data.append(tmp_dict)
            else:
                raise "To be implemented"
    
    def pos_to_bin_pos(self, pos):
        # Not internally used yet
        # pos (3, n) np.ndarray
        b_sub = np.int64(np.round(pos / self.bin_size[:, None]))
        b_ind = np.ravel_multi_index((b_sub[0], b_sub[1], b_sub[2]),
                                      self.bin_mask_size)
        return b_sub, b_ind

    def compute_count_stat(self, cell_pos:list, search_radius=None):
        # Count the number of detected centroid at each voxel position
        acc_count = Linking.compute_accumulated_count_dist(cell_pos, self.mask_size, self.bin_size)
        self.ctr_count = acc_count['count']
        self.num_ctr_vxl = np.count_nonzero(self.ctr_count)
        print(f"Number of centroid voxels: {self.num_ctr_vxl}")
        # Count the conditional count distribution
        self.cond_count = Linking.compute_conditional_count(cell_pos, self.bin_mask_size, self.bin_size,
                                                             search_radius=search_radius, return_matrix_Q=True)
        return self.ctr_count, self.cond_count

    @staticmethod
    def compute_accumulated_count_dist(pos_list:list, mask_size=(75, 400, 400), bin_size=1):
        # pos_list: List of numpy array, each one is a (num_particle, 3) float array 
        bin_size = np.array(bin_size) if not isinstance(bin_size, np.ndarray) else bin_size
        if bin_size.size == 1:
            bin_size = np.repeat(bin_size, mask_size.size)
        result = {}
        result['bin_size'] = bin_size
        result['bin_mask_size'] = np.int64(np.round(mask_size / bin_size))
        count = np.zeros(result['bin_mask_size'], dtype=np.uint16)
        for data in pos_list:
            t_sub = np.int64(np.round(data / bin_size[None, :])) # (n, 3)
            count[(t_sub[:, 0], t_sub[:, 1], t_sub[:, 2])] += 1
        result['count'] = count
        return result
    
    @staticmethod
    def compute_conditional_count(pos_list:list, bin_mask_size, bin_size, search_radius=None, return_matrix_Q=False):
        bin_size = np.array(bin_size) if not isinstance(bin_size, np.ndarray) else bin_size
        if bin_size.size == 1:
            bin_size = np.repeat(bin_size, bin_mask_size.size)
        c1 = []
        c2 = []
        # Compute conditional accumulated count distribution 
        for t_idx in range(len(pos_list)-1):
            t_data = pos_list[t_idx]
            tp1_data = pos_list[t_idx + 1]

            t_pos = t_data.T
            tp1_pos = tp1_data.T
            t_sub = np.int64(np.round(t_pos / bin_size[:, None]))
            tp1_sub = np.int64(np.round(tp1_pos / bin_size[:, None]))
            t_ind = np.ravel_multi_index((t_sub[0], t_sub[1], t_sub[2]), bin_mask_size)
            tp1_ind = np.ravel_multi_index((tp1_sub[0], tp1_sub[1], tp1_sub[2]), bin_mask_size)

            if (search_radius is not None) and (np.isfinite(search_radius)):
                tmp_pdist = np.sqrt(((t_pos[:, :, np.newaxis] - tp1_pos[:, np.newaxis, :]) ** 2).sum(axis=0))
                tmp_in_range_Q = tmp_pdist < search_radius
                tmp_r, tmp_c = np.nonzero(tmp_in_range_Q)
                t_ind = t_ind[tmp_r]
                tp1_ind = tp1_ind[tmp_c]
            
            c1.append(t_ind)
            c2.append(tp1_ind)

        c1 = np.concatenate(c1)
        c2 = np.concatenate(c2)
        if return_matrix_Q:
            bin_num_vsl = np.prod(bin_mask_size)
            cond_acc_count = ssp.coo_matrix((np.ones(c1.shape), (c1, c2)), (bin_num_vsl, bin_num_vsl), dtype=np.uint16).tocsr()
            return cond_acc_count
        else:
            return c1, c2

    def get_single_ind_stat(self, ind):
        pass

    @staticmethod
    def compute_single_edge_conditional_count(e_ind, cond_acc_count, abs_g:graph.AbstractGraph, max_bin_val, left_node_label, clear_cache_Q=False):
        bin_count = np.zeros(max_bin_val * 2, dtype=np.uint16)
        bin_count_same_edge = np.zeros(max_bin_val * 2, dtype=np.uint16)
        for v_idx in range(e_ind.size):
            v_ind = e_ind[v_idx]
            row_start = cond_acc_count.indptr[v_ind]
            row_end = cond_acc_count.indptr[v_ind + 1]
            v_nb_ind = cond_acc_count.indices[row_start : row_end]
            if v_nb_ind.size:
                tmp_path, v_nb_dist = abs_g.compute_shortest_path_between_point_pairs(v_ind, v_nb_ind, cached_Q=True, clear_cache_Q=clear_cache_Q)
                if left_node_label is not None: 
                    tmp_path_dir = graph.AbstractGraph.compute_path_dir_in_edges(left_node_label, tmp_path)
                    # Compute the geodesit distance between these points - do not need the path? 
                    v_nb_dist[tmp_path_dir == -1] *= -1
                
                v_nb_valid_Q = np.abs(v_nb_dist) < max_bin_val

                v_nb_dist = v_nb_dist[v_nb_valid_Q]
                v_nb_c = cond_acc_count.data[row_start : row_end][v_nb_valid_Q]
                v_nb_bin_idx = np.floor(v_nb_dist + max_bin_val).astype(np.uint8)
                bin_count[v_nb_bin_idx] += v_nb_c
                # Histogram count for the points in the same edge
                is_sampe_edge_Q = (np.array([p.size for p in tmp_path[v_nb_valid_Q]]) == 2)
                bin_count_same_edge[v_nb_bin_idx[is_sampe_edge_Q]] += v_nb_c[is_sampe_edge_Q]
                
        return bin_count, bin_count_same_edge

    @staticmethod
    def compute_multi_edge_conditional_count(abs_g: graph.AbstractGraph, cond_acc_count, max_bin_val, clear_cache_Q=False):
        """
            abs_g: AbstractGraph
            cond_acc_count: sparse matrix. Conditional count on skeleton voxel level 
            max_bin_val: numeric scalar, absolute maximum bin value in the histogram of count vs geodesic distance
        """
        num_bin_half = np.ceil(max_bin_val).astype(np.int16)
        bin_edge = np.arange(-num_bin_half, num_bin_half + 1, 1)
        bin_val = (bin_edge[1:] + bin_edge[:-1]) / 2

        node_cc = abs_g.spatial_graph.node.cc_ind
        num_node = node_cc.size
        node_cond_count = np.zeros((num_node, bin_val.size))
        node_single_edge_cond_count = np.zeros((num_node, bin_val.size))
        for n_idx in range(num_node):
            # Do not compute the direction for node: ill-defined. 
            node_cond_count[n_idx], node_single_edge_cond_count[n_idx] = Linking.compute_single_edge_conditional_count(\
                node_cc[n_idx], cond_acc_count, abs_g, num_bin_half, None, clear_cache_Q=clear_cache_Q)
            if n_idx % 50 == 0:
                print(f"Finish processing node {n_idx}")
        
        edge_cc = abs_g.spatial_graph.edge.cc_ind
        num_edge = edge_cc.size
        edge_cond_count = np.zeros((num_edge, bin_val.size))
        same_edge_cond_count = np.zeros((num_edge, bin_val.size))
        for e_idx in range(num_edge):
            e_ind = edge_cc[e_idx]
            edge_cond_count[e_idx], same_edge_cond_count[e_idx] = Linking.compute_single_edge_conditional_count(\
                e_ind, cond_acc_count, abs_g, num_bin_half, abs_g.node_pair[e_idx][0], clear_cache_Q=clear_cache_Q)
            if e_idx % 50 == 0:
                print(f"Finish processing edge {e_idx}")

        result = {'bin_edge': bin_edge, 'bin_val': bin_val, 'max_edge_val': max_bin_val, 
                  'edge_cond_count': edge_cond_count, 'same_edge_cond_count': same_edge_cond_count,
                   'node_cond_count': node_cond_count, 'node_se_cond_count': node_single_edge_cond_count}
        return result
    
    # @staticmethod
    # def analyze_single_edge_conditional_count()

    @staticmethod
    def select_long_trajectories_from_trackpy_result(tp_table, min_length=0, sortedQ=1, return_dict_Q=False, particle_key='particle'):
        p_idx, p_id = util.bin_data_to_idx_list(tp_table[particle_key].values)
        track_length = np.asarray(list(map(lambda x: x.size, p_idx)))
        if min_length: 
            long_track_idx = np.nonzero(track_length >= min_length)[0]
            track_length = track_length[long_track_idx]
            p_id = p_id[long_track_idx]
        else: 
            long_track_idx = np.arange(track_length.size)

        if return_dict_Q: 
            p_id_to_table_idx = {p: i for p, i in zip(p_id, p_idx[long_track_idx])}
            return p_id_to_table_idx
        else: 
            if sortedQ != 0:
                assert (sortedQ == 1 or sortedQ == -1), NotImplementedError
                s_idx = np.argsort(track_length)[::sortedQ] # descending
                long_track_idx = long_track_idx[s_idx]
            
            return np.asarray(p_idx[long_track_idx])

    @staticmethod
    def update(detections:pd.DataFrame, long_traces_ind, trace_result, 
               min_num_frame, min_cos_vv, max_stationary_disp, max_v_norm, add_staionary_Q=True, inplace_Q=True):
        if inplace_Q:
            current_pid = detections.pid.values
        else: 
            current_pid = detections.pid.values.copy() # reference
        
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

#endregion 
class Utility:
    @staticmethod
    def construct_circular_mask(xy_size, r=None):
        if r is None: 
            r = xy_size[0] / 2
        s1, s2 = np.mgrid[0:xy_size[0], 0:xy_size[1]]
        circ_mask = ((s1 - xy_size[0]/2) ** 2 + (s2 - xy_size[1]/2) ** 2 < r ** 2)
        return circ_mask
    
class Visualization:
    @staticmethod
    def compute_mip_for_vis(data, gamma=0.5):
        im_mip = np.max(data, axis=0)
        im_mip = im_mip.astype(np.double)
        im_mip = ((im_mip - im_mip.min()) / (im_mip.max() - im_mip.min())) ** gamma
        return im_mip
    
    @staticmethod
    def vis_local_maxima(im_mip, lm_info):
        diff_ind = [np.setdiff1d(lm_info['ind'][i-1], lm_info['ind'][i]) for  i in range(1, len(lm_info['ind']))]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()
        ax.imshow(im_mip, cmap='gray')
        ax.set_aspect('equal')
        ax.set_title(f"Detected local maximums")
        for i in range(len(lm_info['min_dist_pxl'])):
            if i == 0:
                vis_pos = np.unravel_index(lm_info['ind'][i], lm_info['mask_shape'])
            else:
                vis_pos = np.unravel_index(diff_ind[i-1], lm_info['mask_shape'])
            ax.scatter(vis_pos[2], vis_pos[1], 1, label=f"MD {lm_info['min_dist_pxl'][i]} pxl")
        ax.legend()
        ax.set_xlabel(r'$X\;(2\; \mu m)$')
        ax.set_ylabel(r'$Y\;(2\; \mu m)$')
        
        return fig    

def get_subgraph_cc_feature_from_whole_graph_cc_feature(wg, wgef, sg, sv_disp_vec, obj='edge'):
    """ Get the edge feature in the subgraph from the edge feature of the whole graph 

    For edge feature like artery / vein branch order, we use the entire graph for feature 
    computation. 
    Inputs: 
        wg: whole graph, spatial graph object
        wgef: 1d np.array, whole graph edge feature
        sg: subgraph, spatial graph object 
        sv_disp_vec: (3, ) np.array, displacement vector that transform the coordiante in
          the subgraph to the coordinate in the whole graph. 
    Outputs: 
        sv_ef: sub-graph edge feature
    """
    if obj == 'edge':
        wg_obj = wg.edge
        sg_obj = sg.edge
    elif obj == 'node':
        wg_obj = wg.node
        sg_obj = sg.node

    vxl_ef_val = wg_obj.get_cc_vxl_value_from_cc_feature(wgef, concatenated_Q=True)
    map_ind_to_ef = nb.construct_array_sparse_representation(wg_obj.pos_ind, [np.prod(wg.num['mask_size']), 1], 
                                                            vxl_ef_val)
    # Subvolume map
    sg_e_ind_in_wg = util.ind_coordinate_transform_euclidean(sg_obj.pos_ind, sg.num['mask_size'], 
                                                            sv_disp_vec, wg.num['mask_size'])
    sg_cc_val = map_ind_to_ef[0, sg_e_ind_in_wg].toarray().flatten()
    sv_ccf = sg_obj.get_cc_feature_from_vxl_val_vec(sg_cc_val, stat=['mode'])['mode'] 
    return sv_ccf