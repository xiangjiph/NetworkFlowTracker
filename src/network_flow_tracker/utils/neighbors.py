import numpy as np
from scipy.sparse import coo_matrix
import skimage as ski
import scipy.ndimage as ndi
import warnings

from . import util 

def compute_voxel_offseted_linear_indices(pos_ind, mask_size, offset=1):
    pos_ind = np.array(pos_ind) if isinstance(pos_ind, (list, tuple)) else pos_ind
    mask_size = np.array(mask_size) if isinstance(mask_size, (list, tuple)) else mask_size
    if isinstance(offset, int):
        offset = np.repeat(offset, mask_size.size)
    elif isinstance(offset, (tuple, list)):
        offset = np.array(offset)

    pad_mask_size = mask_size + 2 * offset
    pos_sub = list(np.unravel_index(pos_ind, mask_size))
    for i in range(len(pos_sub)):
        pos_sub[i] += offset[i]
    return np.ravel_multi_index(pos_sub, pad_mask_size)

def construct_connectivity_array(dim, connectivity, remove_center_Q=False):
    if dim == 3: 
        cube = np.ones((3, 3, 3))
        if connectivity == 6: 
            max_dist = 1 + 1e-6
        elif connectivity == 18: 
            max_dist = np.sqrt(2) + 1e-6
        elif connectivity == 26:
            max_dist = np.sqrt(3) + 1e-6
    elif dim == 2: 
        cube = np.ones((3, 3))
        if connectivity == 4: 
            max_dist = 1 + 1e-6
        elif connectivity == 8:
            max_dist = np.sqrt(2) + 1e-6
    sub = np.where(cube)
    sub_array = np.vstack(sub).astype(np.float32)
    ctr_idx = cube.size // 2
    ctr = sub_array[:, ctr_idx]
    dist = np.sqrt(np.sum((sub_array - ctr[:, None]) ** 2, axis=0))
    dist_mask = dist < max_dist
    if remove_center_Q:
        dist_mask[ctr_idx] = False
    return dist_mask.reshape(cube.shape)

def get_connectivity_info(dim, connectivity, remove_center_Q=True):

    if dim == 3: 
        cube = np.ones((3, 3, 3))
        if connectivity == 6: 
            max_dist = 1 + 1e-6
        elif connectivity == 18: 
            max_dist = np.sqrt(2) + 1e-6
        elif connectivity == 26:
            max_dist = np.sqrt(3) + 1e-6
    elif dim == 2: 
        cube = np.ones((3, 3))
        if connectivity == 4: 
            max_dist = 1 + 1e-6
        elif connectivity == 8:
            max_dist = np.sqrt(2) + 1e-6

    sub = np.where(cube)
    sub_array = np.vstack(sub).astype(np.float32)
    ctr_idx = sub_array.shape[1] // 2
    ctr = sub_array[:, ctr_idx]
    dist = np.sqrt(np.sum((sub_array - ctr[:, None]) ** 2, axis=0))
    if remove_center_Q:
        dist[ctr_idx] = np.inf
    dist_mask = dist < max_dist
    masked_sub = [s[dist_mask] for s in sub]
    dist = dist[dist_mask]
    return masked_sub, dist

def generate_neighbor_indices(mask_size, connectivity=26):
    sub, dist = get_connectivity_info(3, connectivity, False)
    tmp1, tmp2, tmp3 = sub

    neighbor_indices = np.ravel_multi_index(
        (tmp1, tmp2, tmp3), mask_size)
    mid_idx = int(neighbor_indices.size // 2)
    neighbor_indices -= neighbor_indices[mid_idx]
    neighbor_indices = np.delete(neighbor_indices, mid_idx)  # Remove center voxel
    dist = np.delete(dist, mid_idx)
    return neighbor_indices, dist

def compute_kernel_offset_pos_and_ind_in_padded_mask(kernel, mask_size):
    mask_size = np.array(mask_size) if isinstance(mask_size, (tuple, list)) else mask_size
    ker_size = np.array(kernel.shape)
    ker_r = ker_size // 2
    sub = np.nonzero(kernel)
    # indices in the padded array
    ind = np.ravel_multi_index([s + ker_r[i] for i, s in enumerate(sub)], mask_size + 2 * ker_r)
    ctr_idx = ind.size // 2
    d_ind = ind - ind.flat[ctr_idx]
    d_sub = [s - s.flat[ctr_idx] for s in sub]
    return d_ind, d_sub

def construct_array_sparse_representation(voxel_list, mask_size, voxel_val=None):
    voxel_list = np.asarray(voxel_list)
    mask_size = np.asarray(mask_size)
    num_voxel = voxel_list.size 
    if voxel_val is None:
        voxel_val = np.arange(1, num_voxel + 1)
    else:
        voxel_val = np.asarray(voxel_val)
        assert voxel_val.size == num_voxel, 'voxel_val should have the same number of element as voxel_list'
        if np.any(voxel_val == 0):
            warnings.warn("voxel_val has elements being 0.")
    # The index must be 1-based for sparse matrix
    sp_skl = coo_matrix(
        (voxel_val, (np.zeros(num_voxel, dtype=int), voxel_list)),
        shape=(1, np.prod(mask_size))).tocsr()
    return sp_skl

def construct_padded_array_sparse_matrix_representation(voxel_list, mask_size, pad_r=1, voxel_val=None):
    mask_size = np.asarray(mask_size)
    # Generate 1D sparse matrix representation of the skeleton array
    voxel_ind_padded = compute_voxel_offseted_linear_indices(voxel_list, mask_size, offset=pad_r)
    sp_skl = construct_array_sparse_representation(voxel_ind_padded, mask_size + 2 * pad_r, voxel_val=voxel_val)
    return sp_skl, voxel_ind_padded

def bwconncomp(mask, connectivity=None, return_ind_Q=True, \
               return_prop_Q=False, return_labeled_array_Q=False):
    # Connectivity skimage | MATLAB
    # 2D: 1 | 4; 2 | 9
    # 3D: 1 | 6; 3 | 26
    if connectivity is None: 
        if mask.ndim == 3:
            connectivity = 3
        elif mask.ndim == 2:
            connectivity = 2
    labeled_array = ski.measure.label(mask, connectivity=connectivity)
    cc_prop = ski.measure.regionprops(labeled_array)
    result = {}
    result['image_size'] = mask.shape
    result['num_cc'] = len(cc_prop)
    cc_subs = [cc.coords for cc in cc_prop]
    result['pixel_sub'] = util.list_to_np_1d_obj_array(cc_subs)
    if return_ind_Q:
        pxl_idx = [np.ravel_multi_index([s[:, i] for i in range(s.shape[1])], mask.shape) 
                                            for s in result['pixel_sub']]
        result['pixel_indices'] = util.list_to_np_1d_obj_array(pxl_idx)

    result['num_pixel_per_cc'] = np.array([p.shape[0] for p in result['pixel_sub']])
    if return_prop_Q:
        result['cc_prop'] = cc_prop
    if return_labeled_array_Q:
        result['labeled_array'] = labeled_array
    return result

def bwareaopen(mask, min_size, connectivity=None):
    cc_info = bwconncomp(mask, connectivity=connectivity, return_ind_Q=True,
                         return_prop_Q=False, return_labeled_array_Q=False)
    selected_Q = list(np.nonzero(cc_info['num_pixel_per_cc'] > min_size)[0])
    new_mask = np.zeros(mask.shape, 'bool')
    cc_ind = np.concatenate(cc_info['pixel_indices'][selected_Q])
    new_mask.flat[cc_ind] = True
    return new_mask


def check_vxl_pair_26_connected(ind_vec_1, ind_vec_2, size):
    ind_vec_1 = np.asarray(ind_vec_1)
    ind_vec_2 = np.asarray(ind_vec_2)
    sub_1 = np.vstack(np.unravel_index(ind_vec_1, size)) # (3, n)
    sub_2 = np.vstack(np.unravel_index(ind_vec_2, size)) # (3, m)
    coor_dist = np.abs(sub_1[:, :, None] - sub_2[:, None, :]) # (3, n, 1) - (3, 1, m) -> (3, n, m)
    connected_Q = np.all(coor_dist <= 1, axis=0)
    return connected_Q


class NearestMaskVoxel:
    def __init__(self, data, save_data_Q=False, val_0_based_label_Q=False):
        # maybe useful if need to map from indices to value
        if save_data_Q:
            self.data = data.copy()
            if val_0_based_label_Q: 
                # when the data is the label array, automatically subject the value by 1 
                self.data -= 1

        self.mask_size = data.shape
        mask = (data != 0) if (data.dtype != 'bool') else data
        self.dt, self.nearest_pos = ndi.distance_transform_edt(1 - mask, return_indices=True)

    def ind_to_nearest_sub(self, ind):
        pos = [pos.flat[ind] for pos in self.nearest_pos]
        return tuple(pos)
    
    def ind_to_nearest_ind(self, ind):
        pos = self.ind_to_nearest_sub(ind)
        return np.ravel_multi_index(pos, self.mask_size)
    
    def sub_to_nearest_sub(self, sub):
        # sub: (3, n) ndarray
        new_sub = [pos[sub[0], sub[1], sub[2]] for pos in self.nearest_pos]
        return np.stack(new_sub, axis=0)
    
    def sub_to_nearest_ind(self, sub):
        if isinstance(sub, (list, tuple)):
            ind = np.ravel_multi_index(sub, self.mask_size)
        elif isinstance(sub, (np.ndarray)):
            assert sub.shape[0] == 3, f"sub should be a (3, *) numpy array"
            ind = np.ravel_multi_index((sub[0], sub[1], sub[2]), self.mask_size)
        return self.ind_to_nearest_ind(ind)
    
    def sub_to_nearest_val(self, sub):
        ind = self.sub_to_nearest_ind(sub)
        return self.data.flat[ind]
    
    def ind_to_nearest_val(self, ind):
        ind = self.ind_to_nearest_ind(ind)
        val = self.data.flat[ind]
        return val
    
    def ind_to_nearest_dist(self, ind):
        return self.dt.flat[ind]

    def zyx_to_nearest_ind(self, zyx):
        return self.sub_to_nearest_ind(np.round(zyx).astype(np.int32))
    
    def zyx_to_nearest_val(self, zyx):
        return self.sub_to_nearest_val(np.round(zyx))