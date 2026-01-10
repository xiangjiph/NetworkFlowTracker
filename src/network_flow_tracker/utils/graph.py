import warnings
import heapq
from collections import defaultdict

import numpy as np
import scipy as sp
from scipy.sparse import issparse
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from numba import njit
import networkx as nx

from . import neighbors as nb
from . import util as util

class GraphCC: 
    # A collection of connected components
    def __init__(self, pxl_ind_list, space_shape):
        self.cc_ind = util.list_to_np_1d_obj_array(pxl_ind_list) if not isinstance(pxl_ind_list, np.ndarray) else pxl_ind_list
        # self.cc_ind = np.array(pxl_ind_list, dtype=object) if not isinstance(pxl_ind_list, np.ndarray) else pxl_ind_list
        assert self.cc_ind.ndim == 1, "Input pxl_ind_list is not a 1d list"
        self.space_shape = np.array(space_shape) if isinstance(space_shape, (list, tuple)) else space_shape
        self._num_voxel_per_cc = None
        self._num_voxel = None
        self._pos_ind = None
        self._voxel_link_label = None
        self._map_ind_to_link_label = None
        self._center_sub = None
    
    def __getstate__(self):
        state = {}
        save_prop = ['cc_ind', 'space_shape']
        for p in save_prop:
            if hasattr(self.__dict__[p], 'copy'):
                state[p] = self.__dict__[p].copy()
            else:
                state[p] = self.__dict__[p]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    @property
    def num_cc(self):
        return len(self.cc_ind) # assume cc_ind is a 1d array
    
    @property
    def num_voxel_per_cc(self):
        if self._num_voxel_per_cc is None:
            self._num_voxel_per_cc = np.array([l.size for l in self.cc_ind]) if self.num_cc else np.array([], dtype=np.int64)
        return self._num_voxel_per_cc
    
    @property
    def pos_ind(self):
        if self._pos_ind is None: 
            self._pos_ind = np.concatenate(self.cc_ind) if self.num_cc else np.array([])
        return self._pos_ind
    
    @property
    def num_voxel(self):
        if self._num_voxel is None: 
            self._num_voxel = int(np.sum(self.num_voxel_per_cc))
        return self._num_voxel

    @property
    def label(self):
        # 0-based
        if self._voxel_link_label is None: 
            self._voxel_link_label = np.repeat(np.arange(self.num_cc), self.num_voxel_per_cc)
        return self._voxel_link_label
    
    @property
    def center_sub(self):
        if self._center_sub is None: 
            if self.num_cc > 0: 
                self._center_sub = np.vstack([self.compute_cc_com(ind, self.space_shape) for ind in self.cc_ind])
            else: 
                self._center_sub = np.empty((0, len(self.space_shape)), dtype=np.int64)
        return self._center_sub

    def ind_to_label(self, val):
        # Link label is used for query cc_ind
        # The tricky part is that sparse matrix elements should be all non-zero 
        if self._map_ind_to_link_label is None: 
            self._map_ind_to_link_label = nb.construct_array_sparse_representation(\
                self.pos_ind, self.space_shape, voxel_val=(self.label + 1))

        # val = [val] if isinstance(val, int) else val
        if isinstance(val, (int, np.ScalarType)):
            link_label = self._map_ind_to_link_label[0, val] - 1
        else:
            # np.array
            link_label = self._map_ind_to_link_label[0, val].toarray()[0] - 1
        # -1 means does not exist - how to implement? Also, is label uint? overflow? 
        return link_label

    @staticmethod
    def compute_cc_com(cc_ind, mask_size):
        # cc_ind: (N, ) ndarray
        cc_ind = np.asarray(cc_ind)
        sub = np.unravel_index(cc_ind, mask_size)
        if cc_ind.size == 1: 
            return np.asarray(sub).flatten()
        else:
            return np.vstack(sub).mean(axis=1).flatten() # (ndim, ) ndarray

    @staticmethod
    def compute_cc_com_ind(cc_ind, mask_size): 
        cc_com = GraphCC.compute_cc_com(cc_ind, mask_size).astype(np.int64)
        return np.ravel_multi_index([cc_com[i] for i in range(cc_com.size)], mask_size)        
        
    @staticmethod
    def compute_dist_to_endpoints(cc_ind, mask_size):
        cc_ind = np.asarray(cc_ind)
        sub = np.unravel_index(cc_ind, mask_size)
        sub = np.vstack(sub)
        adj_dist = np.sqrt(np.sum(np.diff(sub, axis=1) ** 2, axis=0)) 
        dist_forward = np.concatenate(([0], np.cumsum(adj_dist)))
        dist_backward = dist_forward[-1] - dist_forward
        return np.column_stack((dist_forward, dist_backward))
    
    def ind2sub(self, ind_vec):
        ind_vec = np.asarray(ind_vec)
        return np.unravel_index(ind_vec, self.space_shape)

    def sub2ind(self, sub):
        """
            sub: np.ndarray (ndim, N) or (ndim, )
        """
        sub = np.asarray(sub).astype(np.int64)
        return np.ravel_multi_index((sub[0], sub[1], sub[2]), self.space_shape)
    
    def get_cc_vxl_value_from_array(self, value_array):
        val = []
        if isinstance(value_array, np.ndarray): 
            vxl_value = value_array.flat[self.pos_ind]
        elif type(value_array) == sp.sparse._csr.csr_matrix: 
            vxl_value = value_array[:, self.pos_ind].toarray().flatten()
        else: 
            raise NotImplementedError
        
        start_idx = np.concatenate(([0], np.cumsum(self.num_voxel_per_cc, axis=0)))
        for i in range(self.num_cc):
            tmp_val = vxl_value[start_idx[i] : start_idx[i+1]]
            val.append(tmp_val)
        return val
    
    def get_cc_feature_from_voxel_array(self, value_array, stat=['mode'], finite_only_Q=True):
        cc_vxl_val = self.get_cc_vxl_value_from_array(value_array)
        cc_feature = {k : np.full(self.num_cc, np.nan) for k in stat}
        for i, val in enumerate(cc_vxl_val):
            if finite_only_Q: 
                val = val[np.isfinite(val)]
            for tmp_s in stat: 
                if val.size > 1: 
                    if tmp_s == 'mode':
                        val, tmp_count = np.unique(val, return_counts=True)
                        if val.size > 1: 
                            val = val[np.argmax(tmp_count)]
                    elif tmp_s == 'mean': 
                        val = np.mean(val)
                    elif tmp_s == 'median': 
                        val = np.median(val)
                    elif tmp_s == 'std': 
                        val = np.std(val)
                cc_feature[tmp_s][i] = val

        return cc_feature

    def get_cc_feature_from_vxl_val_vec(self, value_vec, stat=['mode'], finite_only_Q=True):

        start_idx = np.concatenate(([0], np.cumsum(self.num_voxel_per_cc, axis=0)))
        cc_feature = {k : np.full(self.num_cc, np.nan) for k in stat}
        for i in range(self.num_cc):
            val = value_vec[start_idx[i] : start_idx[i + 1]]

            if finite_only_Q: 
                val = val[np.isfinite(val)]
            for tmp_s in stat: 
                if val.size > 1: 
                    if tmp_s == 'mode':
                        val, tmp_count = np.unique(val, return_counts=True)
                        if val.size > 1: 
                            val = val[np.argmax(tmp_count)]
                    elif tmp_s == 'mean': 
                        val = np.nanmean(val)
                    elif tmp_s == 'median': 
                        val = np.nanmedian(val)
                    elif tmp_s == 'std': 
                        val = np.nanstd(val)
                cc_feature[tmp_s][i] = val

        return cc_feature

    
    def get_cc_vxl_value_from_cc_feature(self, feature_vec, concatenated_Q=False):
        assert feature_vec.size == self.num_cc, ValueError(f"feature_vec should have the same number of elements as the number of connected components")
        vxl_val = np.repeat(feature_vec.flatten(), self.num_voxel_per_cc)
        if concatenated_Q: 
            return vxl_val
        else: 
            val = []
            start_idx = np.cumsum(np.concatenate(([0], self.num_voxel_per_cc)))
            for i in range(self.num_cc):
                val.append(vxl_val[start_idx[i] : start_idx[i+1]])
            return val


class GraphNode(GraphCC): 
    def __init__(self, pxl_ind_list, space_shape):
        super().__init__(pxl_ind_list, space_shape)
        self._connected_edge_label = None
        self._degree = None

    def __getstate__(self):
        state = super().__getstate__()
        save_prop = ['_connected_edge_label']
        for p in save_prop:
            if hasattr(self.__dict__[p], 'copy'):
                state[p] = self.__dict__[p].copy
            else:
                 state[p] = self.__dict__[p]
        return state
    
    @property
    def connected_edge_label(self):
        if self._connected_edge_label is None: 
            # self._connected_edge_label = [[] for _ in range(len(self.cc_ind))]
            self._connected_edge_label = np.empty((self.num_cc, ), dtype=object)
        return self._connected_edge_label
    
    @property
    def degree(self):
        if self._degree is None: 
            self._degree = np.array([len(x) for x in self._connected_edge_label])
        return self._degree


class GraphEdge(GraphCC):
    def __init__(self, pxl_ind_list, space_shape):
        super().__init__(pxl_ind_list, space_shape)
        self._connected_node_label = - np.ones((len(self.cc_ind), 2), dtype=int)
        self._num_connected_node = np.zeros((len(self.cc_ind), ), dtype=int)
        self._endpoint_ind = None
        self._endpoint_indp_to_label_map = None
    
    def __getstate__(self):
        state = super().__getstate__()
        save_prop = ['_connected_node_label']
        for p in save_prop:
            if hasattr(self.__dict__[p], 'copy'):
                state[p] = self.__dict__[p].copy
            else:
                 state[p] = self.__dict__[p]
        return state

    @property
    def connected_node_label(self):
        return self._connected_node_label
    
    @property
    def num_connected_node(self):
        return self._num_connected_node

    @property
    def endpoint_ind(self):
        if self._endpoint_ind is None: 
            self._endpoint_ind = np.array([[c[0], c[-1]] for c in self.cc_ind], dtype=np.uint)
        return self._endpoint_ind

    def _construct_map_ep_indp_to_label(self):
        e_ep_ind = np.unique(self.endpoint_ind.flatten()) # need unique here? seems only need it if isolated point is counted as an edge
        e_ep_label = self.ind_to_label(e_ep_ind) + 1 # 0-based to 1-based for sparse matrix
        endpoint_indp_to_label_map, e_ep_ind_p = nb.construct_padded_array_sparse_matrix_representation(\
            e_ep_ind, self.space_shape, pad_r=1, voxel_val=e_ep_label)
        return endpoint_indp_to_label_map

    def endpoint_indp_to_label(self, ind_p):
        if self._endpoint_indp_to_label_map is None: 
            self._endpoint_indp_to_label_map = self._construct_map_ep_indp_to_label()
        label = self._endpoint_indp_to_label_map[0, ind_p].toarray().flatten() - 1
        return label
    
    @staticmethod
    def compute_adj_ind_dist(cc_ind, mask_size):
        tmp_sub = np.vstack(np.unravel_index(cc_ind, mask_size))
        # Add a 0 at the beginning? 
        # tmp_adj_dist = np.concatenate(([0], np.sqrt(np.sum(np.diff(tmp_sub, axis=1) ** 2, axis=0))))
        tmp_adj_dist = np.sqrt(np.sum(np.diff(tmp_sub, axis=1) ** 2, axis=0))
        return tmp_adj_dist
    
    @staticmethod
    def compute_edge_length(cc_ind_list, mask_size):
        if isinstance(cc_ind_list, np.ndarray):
            if np.issubdtype(cc_ind_list.dtype, np.number):
                cc_ind_list = [cc_ind_list]
            else:
                assert cc_ind_list.dtype.kind == 'O'
        # avg_voxel_dist_26 = 1.4164218926549295 # (1 * 6 + sqrt(2) * 12 + sqrt(3) * 8) / 26
        avg_voxel_dist_26 = 1
        length = np.zeros(len(cc_ind_list), dtype=np.float32)
        for i, ind in enumerate(cc_ind_list):
            tmp_sub = np.vstack(np.unravel_index(ind, mask_size))
            tmp_adj_dist = np.sqrt(np.sum(np.diff(tmp_sub, axis=1) ** 2, axis=0))
            length[i] = np.sum(tmp_adj_dist) + avg_voxel_dist_26 # 1 seems most reasonable? or some sort of average of 1, sqrt(2), sqrt(3) ? 
        return length
    
    @staticmethod
    def get_sorted_edge_ind_based_on_dist_to_ind(cc_ind, ctr_cc_ind, mask_size, verbose_Q=False):
        cc_ind = np.asarray(cc_ind)
        if cc_ind.size == 1:
            return cc_ind
        else:
            ep_sub = np.vstack(np.unravel_index(cc_ind[[0, -1]], mask_size)) # (3, m)
            ori_sub = np.mean(np.vstack(np.unravel_index(ctr_cc_ind, mask_size)), axis=1, keepdims=True) # (3, 1)
            coor_dist = np.sum((ori_sub - ep_sub) ** 2, axis=0) # (3, m)
            min_idx = np.argmin(coor_dist)
            if min_idx == 1:
                if verbose_Q:
                    print(f"Reverse sequence {cc_ind}")
                return cc_ind[::-1]
            else: 
                return cc_ind
    
    @staticmethod
    def get_edge_segment(cc_ind, ind_1, ind_2, return_dir_Q=False):
        idx1 = np.nonzero(cc_ind == ind_1)[0]
        idx2 = np.nonzero(cc_ind == ind_2)[0]
        assert idx1.size == 1 and idx2.size == 1
        idx1 = int(idx1)
        idx2 = int(idx2)
        if idx1 <= idx2: 
            ind = cc_ind[idx1 : idx2 + 1]
            dir = 1
        else: 
            ind = cc_ind[idx2 : idx1 + 1]
            dir = -1
        if return_dir_Q: 
            dir = np.repeat(dir, ind.size).astype(np.int8)
            return ind, dir
        else: 
            return ind

    @staticmethod
    def compute_geometric_features(cc_ind, mask_size, ep_vec_nb=5, voxel_size_um=1):

        def pca_first_component(X):
            """
            Return the first principal component (largest singular vector)
            of the centered point cloud X (N x 3).
            """
            X_centered = X - np.mean(X, axis=0)
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            # Rows of Vt are the principal directions; the first row is the largest principal component.
            return Vt[0]  # shape: (3,)

        voxel_size_um = np.asarray(voxel_size_um)
        if voxel_size_um.size == 1: 
            voxel_size_um = np.repeat(voxel_size_um, mask_size.size)
        else: 
            assert voxel_size_um.size == mask_size.size, 'Dimension mismatch'
        

        cc_ind = np.asarray(cc_ind, object)

        scalar_features = ['length', 'ep2ep_dist']
        vector_features = ['ep2ep_vec_n', 'ep1_dir_vec', 'ep2_dir_vec']
        
        num_cc = cc_ind.size
        lf = {}
        for k in scalar_features: 
            lf[k] = np.full(num_cc, np.nan)
        for k in vector_features: 
            lf[k] = np.full((num_cc, 3), np.nan)

        for i, tmp_ind in enumerate(cc_ind):
            tmp_num_vxl = tmp_ind.size
            tmp_sub = np.column_stack(np.unravel_index(tmp_ind, mask_size)) # (N, 3)
            if np.any(voxel_size_um != 1): 
                tmp_sub = tmp_sub * voxel_size_um[None, :]
            tmp_adj_dist = np.sqrt(np.sum(np.diff(tmp_sub, axis=0) ** 2, axis=1))
            lf['length'][i] = np.sum(tmp_adj_dist) + 1 # this length does not include the distance to the connected node 
            
            if tmp_num_vxl > 1: 
                ep2ep_vec = tmp_sub[-1] - tmp_sub[0]
                ep_dist = np.sqrt(np.sum(ep2ep_vec ** 2)) 
                lf['ep2ep_vec_n'][i] = ep2ep_vec / ep_dist
                lf['ep2ep_dist'][i] = ep_dist
            
            if tmp_num_vxl > 2: 
                tmp_1_sub = tmp_sub[:min(ep_vec_nb, tmp_num_vxl), :]
                ep_1_vec = pca_first_component(tmp_1_sub)
                tmp_seg_ep_1_vec = tmp_1_sub[0] - tmp_1_sub[-1]
                # Check alignment with ep1->ep2 direction
                if ep_dist > 0 and np.dot(tmp_seg_ep_1_vec, ep_1_vec) < 0:
                    lf['ep1_dir_vec'][i] = -ep_1_vec
                else:
                    lf['ep1_dir_vec'][i] = ep_1_vec

                
                tmp_2_sub = tmp_sub[max(0, tmp_num_vxl - min(ep_vec_nb, tmp_num_vxl)):, :]
                ep_2_vec = pca_first_component(tmp_2_sub)
                tmp_seg_ep_2_vec = tmp_2_sub[-1] - tmp_2_sub[0]
                # Check alignment with ep1->ep2 direction
                if ep_dist > 0 and np.dot(tmp_seg_ep_2_vec, ep_2_vec) < 0:
                    lf['ep2_dir_vec'][i] = -ep_2_vec
                else:
                    lf['ep2_dir_vec'][i] = ep_2_vec

        lf['straightness'] = lf['ep2ep_dist'] / lf['length']

        return lf

class SpatialGraph:
    def __init__(self, voxel_list, mask_size=None):
        """
        Initialize the SkeletonGraph object.
        Inputs: 
            voxel_list: numpy array, can be a 1D array of voxel indices or a 3D binary array.
            mask_size: tuple of ints, size of the mask (if voxel_list is a 1D array).
        """
        voxel_list, self.num = self._parse_inputs(voxel_list, mask_size)
        n_info = self._classify_voxels(voxel_list, self.num)
        link_cc, num_unvisited_points, voxel_unvisited = self._track_links(voxel_list, n_info)
        node_cc = SpatialGraph._find_cc_in_sparse_matrix(voxel_list[n_info['l_nd_idx']], self.num['mask_size'])
        self.edge = GraphEdge(link_cc['PixelIdxList'], self.num['mask_size'])
        self.node = GraphNode(node_cc['PixelIdxList'], self.num['mask_size'])
        if self.node.num_cc:
            self.edge, self.node = SpatialGraph._connect_nodes_and_edges(self.edge, self.node, self.num)
        else:
            print("This spatial graph does not contain a node connected component.")

    def __repr__(self):
        msg = f"Spatial graph with {self.edge.num_cc} edges and {self.node.num_cc} nodes."
        return msg
        
    @property
    def pos_ind(self):
        return np.concatenate((self.edge.pos_ind, self.node.pos_ind))
    
    def ind_to_vxl_idx(self, ind):
        if not hasattr(self, '_ind_to_vxl_idx'):
            self._ind_to_vxl_idx = nb.construct_array_sparse_representation(self.pos_ind, self.num['mask_size']) # 1-based 
        ind = np.asarray(ind)
        # Single point query is much faster than vectorized query of a single value
        if ind.size == 1: 
            idx = self._ind_to_vxl_idx[0, int(ind)] - 1
        else: 
            idx = self._ind_to_vxl_idx[0, ind].toarray().flatten() - 1
        return idx

    @staticmethod
    def _parse_inputs(voxel_list, mask_size=None):
        voxel_list = np.array(voxel_list) if isinstance(voxel_list, (list, tuple)) else voxel_list
        # Initialization
        if voxel_list.ndim != 1:
            # If the input is a 3D logical array, convert it to voxel list.
            if mask_size is None:
                mask_size = voxel_list.shape
            voxel_list = np.flatnonzero(voxel_list)
        else:
            if mask_size is None:
                raise ValueError("mask_size must be provided if voxel_list is a list")
        voxel_list = voxel_list.astype('int64')

        num = {}
        num['mask_size'] = np.array(mask_size) 
        num['mask_size_pad'] = num['mask_size'] + 2
        num['block_voxel'] = np.prod(num['mask_size'])
        num['block_voxel_pad'] = np.prod(num['mask_size_pad'])
        num['skeleton_voxel'] = len(voxel_list)
        num['neighbor_add'], _ = nb.generate_neighbor_indices(num['mask_size'])
        num['neighbor_add_pad'], _ = nb.generate_neighbor_indices(num['mask_size_pad'])
        return voxel_list, num

    @staticmethod
    def _classify_voxels(voxel_list, num):
        sp_skl, voxel_idx_padded = nb.construct_padded_array_sparse_matrix_representation(\
            voxel_list, num['mask_size'], pad_r=1, voxel_val=np.arange(1, voxel_list.size+1))

        nb_vxl_label = voxel_idx_padded[:, None] + num['neighbor_add_pad'][None, :] # shape: (26, num_vxls)
        del voxel_idx_padded
        # Label of the neighboring voxels. 1-based to 0-based
        nb_vxl_label = sp_skl[0, nb_vxl_label].reshape(nb_vxl_label.shape).tocsr().toarray() - 1
        del sp_skl
        # Somehow sorting along axis 0 and 1 (after transpose) takes similar amount of time
        nb_vxl_label = np.sort(nb_vxl_label, axis=1) # ascending
        l_num_neighbor = np.sum(nb_vxl_label >= 0, axis=1)
        n_info = {}
        n_info['l_voxel_neighbor_idx'] = nb_vxl_label
        n_info['num_neighbor'] = l_num_neighbor
        n_info['l_ep_idx'] = np.where(l_num_neighbor == 1)[0] # this indices is 0-based 
        n_info['l_nd_idx'] = np.where(l_num_neighbor > 2)[0] 
        n_info['l_isop_Q'] = (l_num_neighbor == 0)
        return n_info

    @staticmethod
    def _track_links(voxel_list, n_info):
        # Track along links
        num_voxel, num_neighbors = n_info['l_voxel_neighbor_idx'].shape
        # All the edge voxels in the 26-neighbor of the node voxels
        n_nb_vxl_idx = n_info['l_voxel_neighbor_idx'][n_info['l_nd_idx'], :]
        n_nb_vxl_idx = n_nb_vxl_idx[n_nb_vxl_idx >= 0]
        n_nb_vxl_idx = np.unique(n_nb_vxl_idx)
        n_nb_e_vxl_idx = np.setdiff1d(n_nb_vxl_idx, n_info['l_nd_idx'])

        e_start_vxl_idx = np.union1d(n_nb_e_vxl_idx, n_info['l_ep_idx'])
        voxel_unvisited = np.ones(num_voxel, dtype=bool)
        voxel_unvisited[n_info['l_nd_idx']] = False
        voxel_unvisited[n_info['l_isop_Q']] = False
        
        # Get link connected components
        # Does not really need to return voxel_unvisited as the array is passed by reference
        link_cc_ind, num_unvisited_points, voxel_unvisited = SpatialGraph._get_link_cc_numba(
            voxel_list, n_info['l_voxel_neighbor_idx'], e_start_vxl_idx, voxel_unvisited)
        link_cc = {'PixelIdxList': link_cc_ind, 'NumObjects': len(link_cc_ind), 'Connectivity': num_neighbors}
        return link_cc, num_unvisited_points, voxel_unvisited

    def _construct_graph(self, voxel_list, n_info):
        # Construct graph attributes
        link_cc, num_unvisited_points, voxel_unvisited = self._track_links(voxel_list, n_info)
        node_cc = SpatialGraph._find_cc_in_sparse_matrix(voxel_list[n_info['l_nd_idx']], self.num['mask_size'])
        edge = GraphEdge(link_cc['PixelIdxList'], self.num['mask_size'])
        node = GraphNode(node_cc['PixelIdxList'], self.num['mask_size'])
        if self.node.num_cc:
            self.edge, self.node = SpatialGraph._connect_nodes_and_edges(edge, node, self.num)

        self.isopoint = GraphCC([np.array([i]) for i in voxel_list[n_info['l_isop_Q']]], 
                                self.num['mask_size'])
        # TODO: Isolated loop
        if self.num_unvisited_points > 0:
            pass
            #     try:
            #         loop_cc, num_unvisited_points_1, _ = self._get_link_cc_numba(
            #             self.voxel_list, self.l_voxel_neighbor_idx, np.where(self.voxel_unvisited)[0],
            #             self.voxel_unvisited)
            #         assert num_unvisited_points_1 == 0, \
            #             'Still exist unvisited voxels after finding isolated loop connected component'
            #     except Exception as e:
            #         print('The new version of finding isolated loop connected components failed. Using the old version.')
            #         print('Error:', e)
            #         loop_cc = self._find_cc_in_sparse_matrix(self.voxel_list[self.voxel_unvisited], self.mask_size)
            # else:
            #     loop_cc = {'PixelIdxList': []}
            # self.isoloop = {}
            # self.isoloop['cc_ind'] = loop_cc['PixelIdxList']
            # self.isoloop['num_cc'] = len(self.isoloop['cc_ind'])
            # self.isoloop['pos_ind'] = (np.concatenate(self.isoloop['cc_ind'])
            #                            if self.isoloop['cc_ind'] else np.array([]))

    @staticmethod
    @njit(cache=True)
    def _get_link_cc_numba(voxel_list, vxl_nb_idx_list,
                           e_start_vxl_idx, voxel_unvisited_Q):
        # vxl_nb_idx_list.shape:  (num_vxl, 26)
        num_unvisited_points = np.count_nonzero(voxel_unvisited_Q)
        PixelIdxList = []

        num_start_vxl = len(e_start_vxl_idx)
        link_idx = np.zeros(2000, dtype=np.int64)
        start_idx = 0

        while (start_idx < num_start_vxl) and (num_unvisited_points > 0):
            keep_tracking = False
            # Find the starting voxel index for links
            for t_start_point_idx in range(start_idx, num_start_vxl):
                cc_size = 0
                t_current_id = e_start_vxl_idx[t_start_point_idx]
                if voxel_unvisited_Q[t_current_id]:
                    start_idx = t_start_point_idx + 1
                    keep_tracking = True
                    break
            if not keep_tracking:
                break
            while keep_tracking:
                keep_tracking = False
                # Add the current voxel to the connected component voxel list
                if cc_size >= link_idx.size:
                    # Increase the size of t_link_idx
                    link_idx = np.resize(link_idx, link_idx.size * 2)

                link_idx[cc_size] = t_current_id # 0-based indexing
                cc_size += 1
                voxel_unvisited_Q[t_current_id] = False
                # Get the neighbors of the current voxel and pick the one that hasn't been visited
                t_neighbor_idx = vxl_nb_idx_list[t_current_id] # ascending order, the last 2 elements are non-zero
                for tmp_id in t_neighbor_idx[-1:-3:-1]:
                    tmp_id = int(tmp_id)
                    if tmp_id >= 0:
                        if voxel_unvisited_Q[tmp_id]:
                            keep_tracking = True
                            t_current_id = tmp_id
                            break
                    else:
                        break
            if cc_size > 0:
                num_unvisited_points -= cc_size
                PixelIdxList.append(voxel_list[link_idx[:cc_size]].copy()) # t_link_idx is 0-based
        
        return PixelIdxList, num_unvisited_points, voxel_unvisited_Q

    @staticmethod
    def _find_cc_in_sparse_matrix(voxel_ind, mask_size):
        """
        This method collects the connected components in a mask.
        """
        # Initialization
        cc = {
            'PixelIdxList': [],
            'Connectivity': 26,
            'ImageSize': mask_size,
            'NumObjects': 0
        }
        if voxel_ind.size == 0:
            # Warning('The input idx_list is empty')
            return cc
        mask_size = np.array(mask_size) if isinstance(mask_size, (list, tuple)) else mask_size
        # Generate 26 neighbor indices addition coefficients
        neighbor_add_coeff, _ = nb.generate_neighbor_indices(mask_size + 2)

        # Build sparse matrix representation
        sparse_matrix, n_ind_p = nb.construct_padded_array_sparse_matrix_representation(
            voxel_ind, mask_size, pad_r=1, voxel_val=np.arange(1, voxel_ind.size+1))

        # Find the position of the 26 neighbors for each voxel
        nb_label = n_ind_p[:, None] + neighbor_add_coeff[None, :]
        nb_label = sparse_matrix[0, nb_label.flatten()].toarray().flatten().reshape(nb_label.shape) - 1

        # Find connected components
        PixelIdxList, NumObjects = SpatialGraph._find_cc_numba(voxel_ind, nb_label)

        cc['PixelIdxList'] = PixelIdxList
        cc['NumObjects'] = NumObjects

        return cc

    @staticmethod
    @njit(cache=True)
    def _find_cc_numba(idx_list, neighbor_voxel_idx):
        """
        This method collects the connected components in a sparse matrix representation.
        """
        num_voxel = len(idx_list)
        PixelIdxList = []

        # Initialization
        t_node_unvisited = np.ones(num_voxel, dtype=np.bool_)
        t_start_search_idx = 0  # Zero-based indexing
        t_num_unvisited_points = num_voxel
        queue = np.zeros(num_voxel, dtype=np.int64)
        t_cc_node_ind_list = np.zeros(num_voxel, dtype=np.int64)

        while t_num_unvisited_points > 0:
            # Find the next unvisited node to start a new connected component
            found = False
            for t_node_idx in range(t_start_search_idx, num_voxel):
                if t_node_unvisited[t_node_idx]:
                    t_current_id = t_node_idx
                    t_start_search_idx = t_node_idx + 1
                    t_node_unvisited[t_node_idx] = False
                    found = True
                    break
            if not found:
                break

            # Initialize BFS queue pointers
            queue_start_pointer = 0
            queue_end_pointer = 1
            queue[0] = t_current_id
            t_cc_node_ind_list[0] = idx_list[t_current_id]
            t_num_vol = 1  # Number of voxels in the current connected component

            # Perform BFS to find all connected voxels
            while queue_start_pointer < queue_end_pointer:
                t_current_id = queue[queue_start_pointer]
                queue_start_pointer += 1

                # Get neighbors of the current voxel
                new_nb_idx = neighbor_voxel_idx[t_current_id]
                new_nb_idx = new_nb_idx[new_nb_idx >= 0]  # Remove non-neighbors

                # Find unvisited neighbors
                unvisited_mask = t_node_unvisited[new_nb_idx]
                new_nb_idx = new_nb_idx[unvisited_mask]
                num_new_nb = len(new_nb_idx)

                if num_new_nb > 0:
                    # Mark neighbors as visited
                    t_node_unvisited[new_nb_idx] = False

                    # Add neighbors to the connected component list
                    end_vol = t_num_vol + num_new_nb
                    t_cc_node_ind_list[t_num_vol:end_vol] = idx_list[new_nb_idx]
                    t_num_vol = end_vol

                    # Add neighbors to the queue
                    queue[queue_end_pointer:queue_end_pointer + num_new_nb] = new_nb_idx
                    queue_end_pointer += num_new_nb

            # Store the connected component
            PixelIdxList.append(t_cc_node_ind_list[:t_num_vol].copy())
            t_num_unvisited_points -= t_num_vol

        return PixelIdxList, len(PixelIdxList)

    @staticmethod
    def _connect_nodes_and_edges(edge, node, num):
        msz = np.array(num['mask_size'])
        m_e_indp_2_label = edge._construct_map_ep_indp_to_label()
        n_ind_pad = nb.compute_voxel_offseted_linear_indices(node.pos_ind, msz)
        node_cc_range_idx = np.cumsum(np.concatenate(([0], node.num_voxel_per_cc)))
        # Compute neighboring edge label. It does not cost that much to convert to full matrix here as
        # computing the neighbor indices is alrealy a full matrix 
        # In the test dataset, querying the sparse matrix dominates the run time (0.176 s out of 0.195 s)
        n_nb_e_label = n_ind_pad[:, None] + num['neighbor_add_pad'][None, :] # (#node, 26)
        n_nb_e_label = m_e_indp_2_label[0, n_nb_e_label].reshape(n_nb_e_label.shape).tocsr().toarray()

        e_num_c_n, e_c_n_lb, n_c_el = SpatialGraph._connect_nodes_and_edges_compute_numba(node.num_cc, edge.num_cc, \
                                                                                   n_nb_e_label, node_cc_range_idx)
        
        edge._num_connected_node = e_num_c_n
        edge._connected_node_label = e_c_n_lb
        node._connected_edge_label = np.array(n_c_el, dtype=object) 
        # To do: n_nb_e_label can be used to sort the edge.cc_ind

        return edge, node

    def compute_edge_extended_endpoint_pos(self, correct_dir_Q=True):
        """
        Extended endpoint is the position (floating number subscript) where the 
        edges end if node is treated as an ideal point.  
        For edge with endpoints, the extended endpoint position is the endpoint position
        For edge with connected nodes, the extended endpoint position is the node center of mass
        As a by-product, sort the edge voxel sequence from the first node to the second node in place 

        Input: 
            correct_dir_Q: bool scalar. If true, make the edge cc_ind start from the first connected node. 
        Return:
            edge_ep_pos: (num_edge, 2, 3) np.array, arranged according to edge.connected_node_label
        """
        node_ctr = self.node.center_sub
        edge_ep_pos = np.full((self.edge.num_cc, 2, 3), np.nan)
        for e_label, e_ind in enumerate(self.edge.cc_ind):
            e_ind = np.asarray(e_ind) # e_ind has been copied at this point
            e_node_pair = self.edge.connected_node_label[e_label]
            for i, n_label in enumerate(e_node_pair):
                if n_label == -1:
                    # endpoint
                    ep_ind = e_ind[0] if i == 0 else e_ind[-1]
                    edge_ep_pos[e_label][i] = self.ind2sub(ep_ind)
                else:
                    assert n_label >= 0, "Invalid node label"
                    edge_ep_pos[e_label][i] = node_ctr[n_label]
                    if i == 0 and e_ind.size > 1:
                        ep_sub = np.vstack(self.ind2sub(e_ind[[0, -1]])) # (3, m)
                        ori_sub = edge_ep_pos[e_label][i].reshape((3, 1))
                        coor_dist = np.sum((ori_sub - ep_sub) ** 2, axis=0) # (3, m)
                        min_idx = np.argmin(coor_dist)
                        if min_idx == 1:
                            # print(f"Distance between the first node and the two endpoints: {coor_dist}. Reverse sequence.")
                            e_ind = e_ind[::-1] # this must be updated for the case where the first endpoint is connected while the second endpoint is free 
                            if correct_dir_Q: 
                                self.edge.cc_ind[e_label] = e_ind # reverse the voxel sequence
        return edge_ep_pos

    def compute_edge_effective_length(self):
        """
            Compute the effective length of the edge, including its distance to the connected 
            node centroid. If the edge has free endpoints, the endpoints are duplicated and
            added to the begining / end of the voxel coordiante array (extended_edge_cc_pos)
        """
        extended_ep_pos = self.compute_edge_extended_endpoint_pos(correct_dir_Q=True) # sort the edge ind in place
        length = np.zeros(self.edge.num_cc, dtype=np.float32)
        extended_edge_cc_pos = []
        for i, ind in enumerate(self.edge.cc_ind):
            e_start = extended_ep_pos[i][0]
            e_end = extended_ep_pos[i][1]
            tmp_sub = np.vstack(np.unravel_index(ind, self.num['mask_size'])) # (3, n)
            tmp_sub = np.hstack((e_start[:, None], tmp_sub, e_end[:, None]))
            tmp_adj_dist = np.sqrt(np.sum(np.diff(tmp_sub, axis=1) ** 2, axis=0))
            length[i] = np.sum(tmp_adj_dist)
            extended_edge_cc_pos.append(tmp_sub)
        return length, extended_ep_pos, extended_edge_cc_pos

    @staticmethod
    @njit(cache=True)
    def _connect_nodes_and_edges_compute_numba(num_node, num_edge, n_nb_e_label, node_cc_range_idx):
        # n_nb_e_label is a full matrix. 
        # In test dataset with 610 nodes, this function is about 5x faster than the 
        # function with the sparse matrix input (0.022 vs 0.122 s)
        # Numba compiling takes ~ 1 second; later run time ~ 0.002 second
        e_num_c_n = np.zeros((num_edge, ), dtype=np.uint8)
        e_c_n_lb = - np.ones((num_edge, 2), dtype=np.int32)
        n_c_e_l = []

        for t_n_label in range(num_node):
            t_n_nb_e_label = n_nb_e_label[node_cc_range_idx[t_n_label] : node_cc_range_idx[t_n_label+1]].flatten()
            t_n_nb_e_label = t_n_nb_e_label[t_n_nb_e_label > 0]
            t_n_nb_e_label = np.unique(t_n_nb_e_label) - 1
            
            n_c_e_l.append(t_n_nb_e_label)
            for e_label in t_n_nb_e_label:
                e_c_n_lb[e_label, e_num_c_n[e_label]] = t_n_label 
                e_num_c_n[e_label] += 1
                assert e_num_c_n[e_label] < 3, 'The link should be connected to less than 3 node voxels'

        return e_num_c_n, e_c_n_lb, n_c_e_l
    
    @staticmethod
    def _connect_nodes_and_edges_compute(num_node, num_edge, n_nb_e_label, node_cc_range_idx):
        # n_nb_e_label is a full matrix. 
        # In test dataset with 610 nodes, this function is about 5x faster than the 
        # function with the sparse matrix input (0.022 vs 0.122 s)
        e_num_c_n = np.zeros((num_edge, ), dtype=np.uint8)
        e_c_n_lb = - np.ones((num_edge, 2), dtype=np.int32)
        n_c_e_l = [[] for _ in range(num_node)]

        for t_n_label in range(num_node):
            t_n_nb_e_label = n_nb_e_label[node_cc_range_idx[t_n_label] : node_cc_range_idx[t_n_label+1]].flatten()
            t_n_nb_e_label = t_n_nb_e_label[t_n_nb_e_label > 0]
            t_n_nb_e_label = np.unique(t_n_nb_e_label) - 1

            n_c_e_l[t_n_label] = t_n_nb_e_label
            for e_label in t_n_nb_e_label:
                e_c_n_lb[e_label, e_num_c_n[e_label]] = t_n_label 
                e_num_c_n[e_label] += 1
                assert e_num_c_n[e_label] < 3, 'The link should be connected to less than 3 node voxels'
        return e_num_c_n, e_c_n_lb, n_c_e_l

#region Helper functions
    def ind2sub(self, ind_vec):
        ind_vec = np.asarray(ind_vec)
        return np.unravel_index(ind_vec, self.num['mask_size'])
    
    def get_edges_label_between_node_pair(self, node_1, node_2):
        edges_1 = self.node.connected_edge_label[node_1]
        edges_2 = self.node.connected_edge_label[node_2]
        shared_edge = np.intersect1d(edges_1, edges_2)
        return shared_edge
#endregion

#region Branch order
    def get_nearest_neighbor_edge_labels_of_an_edge(self, edge_label:int, output_dir_Q=False):
        """
        
        Outputs: 
            nb_edge_label: [np.array, np.array], list of connected, first order neighboring edges
                The order of the numpy arraies are the same as the order of the node labels
            nb_edge_dir: similar with nb_edge_label. 1 if the nb edge cc voxels are ordered 
                to start from the node that two edges are connected through the node; -1 otherwise. 
        """
        connected_node = self.edge.connected_node_label[edge_label]
        nb_edge_label = []
        if output_dir_Q: 
            nb_edge_dir = []
        for i, n_label in enumerate(connected_node):
            if n_label == -1:
                nb_edge = np.array([], np.int64)
                e_dir = np.array([], np.int8)
            else:
                nb_edge = self.node.connected_edge_label[n_label]
                nb_edge = nb_edge[nb_edge != edge_label]

                if output_dir_Q: 
                    e_dir = np.zeros(nb_edge.shape, np.int8)
                    for j, e in enumerate(nb_edge):
                        e_node = self.edge.connected_node_label[e]
                        if e_node[0] == n_label: 
                            e_dir[j] = 1
                        elif e_node[1] == n_label: 
                            e_dir[j] = -1
                        else: 
                            raise ValueError(f"Edge {e} is not connected to node {n_label}")

            nb_edge_label.append(nb_edge)
            if output_dir_Q: 
                nb_edge_dir.append(e_dir)
                
        if output_dir_Q: 
            return nb_edge_label, nb_edge_dir
        else: 
            return nb_edge_label

    def get_nearest_neighbor_edge_labels(self, edge_label):
        # merge 
        assert np.all(edge_label >= 0), 'Exist invalid edge label'
        connected_node = self.edge.connected_node_label[edge_label].flatten()
        connected_node = connected_node[connected_node != -1]
        nb_edge = self.node.connected_edge_label[connected_node]
        nb_edge = np.setdiff1d(np.concatenate(nb_edge), edge_label, assume_unique=False)
        return nb_edge
    
    def get_nearest_neighbor_node_labels(self, node_label):
        node_label = np.asarray(node_label)
        assert np.all(node_label >= 0), 'Exist invalid node label'
        connected_edge = self.node.connected_edge_label[node_label]
        if connected_edge.dtype == np.object_:
            connected_edge = np.concatenate(connected_edge)            
        connected_edge = connected_edge[connected_edge != -1]
        nb_node = self.edge.connected_node_label[connected_edge].flatten()
        nb_node = nb_node[nb_node != -1]
        nb_node = np.setdiff1d(nb_node, node_label, assume_unique=False)
        return nb_node
        
    def get_neighbor_edge_by_branch_order_of_an_edge(self, edge_label, max_branch_order=1):
        result = {}
        start_edge = edge_label.copy if isinstance(edge_label, (list, tuple, np.ndarray)) else edge_label
        labeled_edge = None
        bo = 1
        while bo <= max_branch_order:
            nb_edge = self.get_nearest_neighbor_edge_labels(start_edge)
            if labeled_edge is None: 
                labeled_edge = nb_edge
            else:
                nb_edge = np.setdiff1d(nb_edge, labeled_edge)
                labeled_edge = np.union1d(labeled_edge, nb_edge)
            if nb_edge.size:
                result[bo] = nb_edge
                start_edge = nb_edge
                bo += 1
            else:
                break
        return result
    
    def compute_edge_branch_order(self, source_label, drain_label=None):
        unlabeled_od = -1
        branch_order = np.full((self.edge.num_cc, ), unlabeled_od, dtype=np.int16)
        if drain_label is not None: 
            is_drain_Q = np.zeros(self.edge.num_cc, bool)
            is_drain_Q[np.asarray(drain_label)] = True
        curr_order = 0
        max_order = 1000
        while source_label.size > 0: 
            branch_order[source_label] = curr_order
            curr_order += 1
            next_candidate = self.get_nearest_neighbor_edge_labels(source_label)
            unlabeled_Q = (branch_order[next_candidate] == unlabeled_od)
            if drain_label is not None: 
                unlabeled_Q = np.logical_and(unlabeled_Q, ~is_drain_Q[next_candidate])
            source_label = next_candidate[unlabeled_Q]

            if curr_order >= max_order: 
                NotImplementedError(f"branch order gets greater than {max_order}. This is very unlikely. Please double check the data")
        return branch_order.astype(np.int16)
    
    def compute_node_branch_order(self, source_label, drain_label=None):
        unlabeled_od = -1
        branch_order = np.full((self.node.num_cc, ), unlabeled_od, dtype=np.int16)
        if drain_label is not None: 
            is_drain_Q = np.zeros(self.node.num_cc, bool)
            is_drain_Q[np.asarray(drain_label)] = True

        curr_order = 0
        max_order = 1000
        while source_label.size > 0: 
            branch_order[source_label] = curr_order
            curr_order += 1
            next_candidate = self.get_nearest_neighbor_node_labels(source_label)
            unlabeled_Q = (branch_order[next_candidate] == unlabeled_od)
            if drain_label is not None: 
                unlabeled_Q = np.logical_and(unlabeled_Q, ~is_drain_Q[next_candidate])
            source_label = next_candidate[unlabeled_Q]

            if curr_order >= max_order: 
                NotImplementedError(f"branch order gets greater than {max_order}. This is very unlikely. Please double check the data")
        return branch_order.astype(np.int16)
    
#endregion
    
#region VoxelGraph
class VoxelGraph():
    def __init__(self):
        pass

    @staticmethod
    def construct_voxel_connectivity_sparse_matrix(mask, connectivity=26):
        mask_size = np.array(mask.shape)
        voxel_list, num = SpatialGraph._parse_inputs(mask)
        nb_add_pad, nb_dist = nb.generate_neighbor_indices(mask_size + 2, connectivity=connectivity)
        sp_nb, voxel_idx_padded = nb.construct_padded_array_sparse_matrix_representation(
            voxel_list, mask_size, pad_r=1, voxel_val=np.arange(1, voxel_list.size+1))
        nb_vxl_label = voxel_idx_padded[:, None] + nb_add_pad[None, :] # shape: (connectivity, num_vxls)
        # Label of the neighboring voxels. 1-based to 0-based
        nb_vxl_label = sp_nb[0, nb_vxl_label].reshape(nb_vxl_label.shape).tocsr().toarray() - 1
        self_idx, nb_idx = np.where(nb_vxl_label > -1)
        nb_dist = nb_dist[nb_idx]
        nb_idx = nb_vxl_label[self_idx, nb_idx]
        sp_l = np.prod(mask_size)

        adj_graph = coo_matrix(
            (nb_dist, (voxel_list[self_idx], voxel_list[nb_idx])), 
            shape=(sp_l, sp_l)).tocsr()
        return adj_graph, voxel_list
    
    @staticmethod
    def compute_distance_in_mask(mask, pos_1, pos_2, connectivity=26, limit=np.inf):
        mask_size = np.array(mask.shape)
        ind_1 = np.ravel_multi_index(pos_1, mask_size)
        ind_2 = np.ravel_multi_index(pos_2, mask_size)
        adj_graph, voxel_list = VoxelGraph.construct_voxel_connectivity_sparse_matrix(mask, connectivity)
        dist_from_1 = dijkstra(adj_graph, directed=False, indices=ind_1, return_predecessors=False, limit=limit)
        return dist_from_1[ind_2]


    @staticmethod
    def test():
        test_array = np.ones((3, 3, 3), 'bool')
        adj_graph, voxel_list = VoxelGraph.construct_voxel_connectivity_sparse_matrix(test_array)
        test_dist = VoxelGraph.compute_distance_in_mask(adj_graph, np.unravel_index(voxel_list[0], test_array.shape),
                                            np.unravel_index(voxel_list[4], test_array.shape))
        assert (test_dist - np.sqrt(2)) < 1e-6
#endregion

#region AbstractGraph
class AbstractGraph:
    """
    Abstract graph derived from spatial graph for computing the distance between skeleton
    voxels efficiently. 
    Note: Node size is ignored so the length of the path that contains more than one 
    edges will be smaller than the length computed directly from the voxel graph above. 
    """
    def __init__(self, vsl_graph:SpatialGraph):
        """
        
            self.eff_ep_pos (np.ndarray): A 1D numpy object array, each object is a (2, 3) 
              numpy array (float) of the two endpoint / node positions the edge is connected
              to.
            self.edge_cc_pos (np.ndarray): A 1D numpy object array, each object is a (3, M)
              numpy array (float) where M is the number of voxels in the edge connected 
              components.
        """
        self.length, self.node_pair, self.eff_ep_pos, self.edge_cc_pos = \
            self._parse_spatial_graph(vsl_graph)
        self.spatial_graph = vsl_graph
        self.mask_size = vsl_graph.num['mask_size']

        self.edge_info = []
        for el, w in enumerate(self.length):
            tmp_e = (self.node_pair[el][0], self.node_pair[el][1], el, {'weight': w})
            self.edge_info.append(tmp_e)
        self.G = nx.MultiGraph()
        self.G.add_edges_from(self.edge_info)

    def prepare_split_edge_data(self, edge_label, split_ind, new_node_label=None):
        if type(edge_label) == np.ndarray and edge_label.size == 1: 
            edge_label = int(edge_label)
        assert edge_label >= 0, ValueError(f"Edge label {edge_label} should be a non-negative integer")
        rm_edge_info = self.edge_info[edge_label]
        rm_node_pair = self.node_pair[edge_label]

        if new_node_label is None: 
            new_node_label = self.G.number_of_nodes()
        
        # Use the extended length distance that account for the size of the node
        # [   0    ,    1   ,    2   ,    ,     i    , ..., n + 1  ]
        # [d(es, 0), d(0, 1), d(1, 2), ..., d(i-1, i), ... d(n, ee)]
        # edge_ind = self.edge_cc[edge_label] # voxels in each segment is sorted to start from the first node in the pair
        # e_start = self.eff_ep_pos[edge_label][0]
        # e_end = self.eff_ep_pos[edge_label][1]
        # split_idx = np.nonzero(edge_ind == split_ind)[0][0]
        # tmp_sub = np.vstack(np.unravel_index(edge_ind, self.mask_size)) # (n, 3)
        # tmp_sub = np.hstack((e_start[:, None], tmp_sub, e_end[:, None]))
        # tmp_adj_dist = np.sqrt(np.sum(np.diff(tmp_sub, axis=1) ** 2, axis=0)) 
        # new_edge_l_1 = np.sum(tmp_adj_dist[:split_idx+1])
        # new_edge_l_2 = np.sum(tmp_adj_dist[split_idx+1:])

        edge_pos = self.edge_cc_pos[edge_label]
        split_idx = self.find_ind_in_extended_sub_array(edge_pos, split_ind, self.mask_size)
        tmp_adj_dist = np.sqrt(np.sum(np.diff(edge_pos, axis=1) ** 2, axis=0)) 
        new_edge_l_1 = np.sum(tmp_adj_dist[:split_idx])
        new_edge_l_2 = np.sum(tmp_adj_dist[split_idx:])

        new_edge_1 = (rm_node_pair[0], new_node_label, 'pe1', {'weight': new_edge_l_1})
        new_edge_2 = (new_node_label, rm_node_pair[1], 'pe2', {'weight': new_edge_l_2}) 
        return [rm_edge_info], [new_edge_1, new_edge_2]
    
    def ind_to_node(self, ind, new_node_label=None):
        node_label_1 = self.spatial_graph.node.ind_to_label(ind)
        rm_edge = []
        add_edges = []
        if node_label_1 != -1:
            source_node = int(node_label_1)
        else:
            edge_label = self.spatial_graph.edge.ind_to_label(ind)
            if edge_label >= 0:
                if new_node_label is None:
                    new_node_label = self.G.number_of_nodes()
                source_node = int(new_node_label)
                rm_edge, add_edges = self.prepare_split_edge_data(\
                    edge_label, ind, source_node)
            else:
                source_node = []
                # print("ind is neither a node nor an edge voxel")
        return source_node, rm_edge, add_edges
        
    def compute_shortest_path_length_and_dir_within_one_edge(self, ind1, ind2):
        """
        Args: 
            ind1, ind2: non-negative integer scalar, skeleton voxel linear indices
        Outputs: 
            l: float scalar, length of the path if two indices are in the same edge 
            dir: element of {0, +1, -1}, 0 for not connected
            edge_label: scalar or 2-element numpy integer array, edge label
        """
        # somehow querying 1 label at a time is much faster than query 2 labels simulatneously...
        edge_label = self.spatial_graph.edge.ind_to_label(ind1)
        lable_2 = self.spatial_graph.edge.ind_to_label(ind2)
        if (edge_label == lable_2) and (edge_label >= 0):
            edge_sub = self.edge_cc_pos[edge_label].astype(np.int32)
            idx_1 = self.find_ind_in_extended_sub_array(edge_sub, ind1, self.mask_size)
            idx_2 = self.find_ind_in_extended_sub_array(edge_sub, ind2, self.mask_size)

            if idx_1 > idx_2:
                idx_1, idx_2 = idx_2, idx_1
                dir = -1
            else: 
                dir = 1
            # Note the index difference here, because of slicing before taking the difference
            seg_sub = edge_sub[:, idx_1 : idx_2+1]
            l = np.sum(np.sqrt(np.sum(np.diff(seg_sub, axis=1) ** 2, axis=0)))
        else: 
            l = np.inf
            dir = 0
            edge_label = np.array([edge_label, lable_2])

        return l, dir, edge_label
        
    def compute_shortest_path_length_between_two_voxel_indices(self, ind1, ind2, cached_Q=True, clear_cache_Q=False):
        if (cached_Q and not hasattr(self, '_path_length')) or clear_cache_Q:
            self._path_length = {}

        if ind1 == ind2: 
            return 0
        
        if cached_Q:
            node_pair = (ind1, ind2)
            conj_node_pair = (ind2, ind1)
            if node_pair in self._path_length:
                return self._path_length[node_pair]
            elif conj_node_pair in self._path_length:
                return self._path_length[conj_node_pair]
        # if in the same edge: 
        path_length, _, _ = self.compute_shortest_path_length_and_dir_within_one_edge(ind1, ind2)
        # if in different edges:
        if path_length == np.inf:
            new_node_label = self.G.number_of_nodes()
            source_node, rm_edge_1, add_edges_1 = self.ind_to_node(ind1, new_node_label)
            target_node, rm_edge_2, add_edges_2 = self.ind_to_node(ind2, new_node_label + 1)
            if isinstance(source_node, int) and isinstance(target_node, int):
                rm_edge = rm_edge_1 + rm_edge_2
                add_edge = add_edges_1 + add_edges_2
                # print(rm_edge, add_edge)
                # Copy at the moment for debuging            
                if len(rm_edge):
                    self.G.remove_edges_from(rm_edge)
                    self.G.add_edges_from(add_edge)
                try:
                    path_length = nx.shortest_path_length(self.G, source_node, target_node, weight="weight")
                except nx.NetworkXNoPath:
                    path_length = np.inf
                if len(rm_edge):
                    self.G.remove_edges_from(add_edge)
                    self.G.add_edges_from(rm_edge)
                    self.G.remove_nodes_from([new_node_label, new_node_label+1])
            else:
                path_length = np.inf

            if cached_Q:
                self._path_length[node_pair] = path_length

        return path_length

    def compute_shortest_path_between_two_voxel_indices(self, ind1, ind2, cached_Q=True, clear_cache_Q=False):
        """
        Compute the distance between two voxels in the weighted undirected graph of vessel skeleton. 

        ind1, ind2: integer scalars, indices of voxels in the skeleton (NOT node label in the network!)
        """

        if (cached_Q and not hasattr(self, '_path')) or clear_cache_Q:
            self._path = {}
            self._path_length = {}

        if cached_Q:
            node_pair = (ind1, ind2)
            conj_node_pair = (ind2, ind1)
            if node_pair in self._path and node_pair in self._path_length:
                return self._path[node_pair], self._path_length[node_pair]
            elif conj_node_pair in self._path and conj_node_pair in self._path_length:
                return self._path[conj_node_pair][::-1], self._path_length[conj_node_pair]            

        new_node_label = self.G.number_of_nodes()
        # if in the same edge: 
        path_length, dir, edge_label = self.compute_shortest_path_length_and_dir_within_one_edge(ind1, ind2)
        if dir == 1: 
            path = [new_node_label, new_node_label + 1]
        elif dir == -1: 
            path = [new_node_label + 1, new_node_label]
        else: 
            path = []
        # edge_path = np.asarray(edge_label) # Not sure if edge path should be computed here or not... 

        # if in different edges:
        if path_length == np.inf:            
            source_node, rm_edge_1, add_edges_1 = self.ind_to_node(ind1, new_node_label)
            target_node, rm_edge_2, add_edges_2 = self.ind_to_node(ind2, new_node_label + 1)
            if isinstance(source_node, int) and isinstance(target_node, int):
                rm_edge = rm_edge_1 + rm_edge_2
                add_edge = add_edges_1 + add_edges_2
                # print(rm_edge, add_edge)
                # Copy at the moment for debuging            
                if len(rm_edge):
                    self.G.remove_edges_from(rm_edge)
                    self.G.add_edges_from(add_edge)
                try:
                    path = nx.dijkstra_path(self.G, source_node, target_node, weight="weight")
                    path_length = np.sum([min(e['weight'] for e in self.G[u][v].values()) for u, v in zip(path[:-1], path[1:])])
                except nx.NetworkXNoPath:
                    path_length = np.inf
                if len(rm_edge):
                    self.G.remove_edges_from(add_edge)
                    self.G.add_edges_from(rm_edge)
                    self.G.remove_nodes_from([new_node_label, new_node_label+1])
            else:
                path_length = np.inf

        path = np.asarray(path)
        if cached_Q:
            self._path[node_pair] = path
            self._path_length[node_pair] = path_length

        return path, path_length

    def compute_shortest_path_length_between_point_pairs(self, source_list, target_list, cached_Q=False, clear_cache_Q=False):
        source_list = np.asarray(source_list)
        target_list = np.asarray(target_list)
        dist = np.repeat(np.inf, source_list.shape)
        for i in range(source_list.size):
            dist[i] = self.compute_shortest_path_length_between_two_voxel_indices(
                source_list[i], target_list[i], cached_Q=cached_Q, clear_cache_Q=clear_cache_Q)
        return dist
    
    def compute_shortest_path_between_point_pairs(self, source_list, target_list, cached_Q=False, clear_cache_Q=False):
        """
        Args: 
            source_list: 1D numeric array, indices of the source points 
            target_list: 1D numeric array, indices of the target points 
            cached_Q: logical scalar
            clear_cache_Q: logical scalar        
        """
        source_list = np.asarray(source_list)
        if source_list.size == 1 and target_list.size >= 1:
            source_list = np.repeat(source_list, target_list.size)
        if target_list.size == 1 and source_list.size >= 1:
            target_list = np.repeat(target_list, source_list.size)
        target_list = np.asarray(target_list)
        dist = np.repeat(np.inf, source_list.size)
        path = np.empty(source_list.size, dtype=object)
        for i in range(source_list.size):
            path[i], dist[i] = self.compute_shortest_path_between_two_voxel_indices(
                source_list[i], target_list[i], cached_Q=cached_Q, clear_cache_Q=clear_cache_Q)
        return path, dist

    def compute_path_dir_from_one_node(self, start_node_label, path, end_edge_label=-1, end_node_label=-1):
        """
        The direction from a node seems not well defined. 
        Based on edge node pair order? But this is arbitrary too! We don't know the flow direction initially. 
        Is it really useful to compute this? 
        """
        assert path[0] == start_node_label, "The path does not start with the specified node"
        if path.size > 2: 
            # use the first next node 
            tmp_next_node_label = path[1]
            end_edge_label = list(self.G.get_edge_data(start_node_label, tmp_next_node_label).keys())
            if len(end_edge_label) > 1:
                # Pick the first one at the moment 
                end_edge_label = end_edge_label[0]
            tmp_node_pair = self.node_pair[end_edge_label]
        elif path.size == 2: 
            # within 1 edge            
            if end_edge_label >= 0:
                tmp_node_pair = self.node_pair[end_edge_label]
            elif end_node_label >= 0:
                # Get edge direction 
                end_edge_label = list(self.G.get_edge_data(start_node_label, end_node_label).keys())
                if len(end_edge_label) > 1:
                    # Pick the first one at the moment 
                    end_edge_label = end_edge_label[0]
                tmp_node_pair = self.node_pair[end_edge_label]
            else: 
                raise NotImplementedError
        else: 
            raise NotImplementedError

        path_dir = 1 if tmp_node_pair.flatten()[0] == start_node_label else -1

        return path_dir

    def is_dummy_node_Q(self, node_label):
        return (node_label >= self.G.number_of_nodes())

    def get_edge_path_from_node_path(self, node_path, source_ind, target_ind):
        source_e_label = int(self.spatial_graph.edge.ind_to_label(source_ind))
        target_e_label = int(self.spatial_graph.edge.ind_to_label(target_ind))
        edge_list = []
        if (source_e_label == target_e_label) and (source_e_label >= 0): 
            assert node_path.size == 2, f"node_path contains more than 2 nodes"
            edge_list = [source_e_label]
        else: 
            for u, v in zip(node_path[:-1], node_path[1:]):
                if self.is_dummy_node_Q(u):
                    assert not self.is_dummy_node_Q(v), "Both nodes are dummy nodes"
                    tmp_edge = source_e_label
                elif self.is_dummy_node_Q(v):
                    assert not self.is_dummy_node_Q(u), "Both nodes are dummy nodes"
                    tmp_edge = target_e_label
                else:
                    tmp_dict = self.G.get_edge_data(u, v)
                    tmp_edge = None
                    tmp_edge_weight = np.inf
                    for k, v in tmp_dict.items():
                        if (tmp_edge is None) or (tmp_edge_weight < v['weight']):
                            tmp_edge = k
                            tmp_edge_weight = v['weight']
                assert tmp_edge >= 0, "Invalid edge label"
                edge_list.append(tmp_edge)
        
        return np.asarray(edge_list)
    
    def get_voxel_path_from_node_path(self, node_path, source_ind, target_ind, include_node_Q=False, return_edge_path_Q=False):
        edge_path = self.get_edge_path_from_node_path(node_path, source_ind, target_ind)
        if edge_path.size == 0:             
            # node_path = [] would suggests that the two voxels are not connected in the graph. This should be imposible for the 
            # current application of this function, as the points being linked all have finite graph geodesic distance. 
            assert (node_path.size == 1), f'Two voxels are in the same node connected component. Node path: {node_path, source_ind, target_ind}'
            path_ind = np.array([], dtype=np.int32)
            path_vxl_dir = np.array([], dtype=np.int8)
            path_edge_dir = np.array([], dtype=np.int8)
        elif edge_path.size == 1:
            tmp_e = int(edge_path)
            edge_ind = self.spatial_graph.edge.cc_ind[tmp_e]
            tmp_e_cnt_node = self.spatial_graph.edge.connected_node_label[tmp_e] 
            source_node_label = self.spatial_graph.node.ind_to_label(source_ind)
            if source_node_label >= 0:
                node_ind = self.spatial_graph.node.cc_ind[source_node_label]
                if tmp_e_cnt_node[0] == source_node_label: 
                    source_ind = node_ind[0] if include_node_Q else edge_ind[0]
                    edge_ind = np.concatenate((node_ind, edge_ind)) if include_node_Q else edge_ind
                else:
                    source_ind = node_ind[-1] if include_node_Q else edge_ind[-1]
                    edge_ind = np.concatenate((edge_ind, node_ind)) if include_node_Q else edge_ind

            target_node_label = self.spatial_graph.node.ind_to_label(target_ind)
            if target_node_label >= 0:
                node_ind = self.spatial_graph.node.cc_ind[target_node_label]
                if tmp_e_cnt_node[0] == target_node_label: 
                    target_ind = node_ind[0] if include_node_Q else edge_ind[0]
                    edge_ind = np.concatenate((node_ind, edge_ind)) if include_node_Q else edge_ind
                else:
                    target_ind = node_ind[-1] if include_node_Q else edge_ind[-1]
                    edge_ind = np.concatenate((edge_ind, node_ind)) if include_node_Q else edge_ind

            path_ind, path_vxl_dir = self.spatial_graph.edge.get_edge_segment(edge_ind, source_ind, target_ind, return_dir_Q=True)
            path_edge_dir = path_vxl_dir[[0]]
            # reverse direction 
            if path_vxl_dir[0] == -1:
                path_ind = path_ind[::-1]
        else:
            path_ind = []
            path_vxl_dir = []
            path_edge_dir = []
            for i, tmp_e in enumerate(edge_path):
                tmp_e_ind = self.spatial_graph.edge.cc_ind[tmp_e]
                tmp_e_cnt_node = self.spatial_graph.edge.connected_node_label[tmp_e] 

                if i == 0: 
                    partial_cc_Q = self.is_dummy_node_Q(node_path[0])
                    if tmp_e_cnt_node[0] == node_path[1]: 
                        tmp_ind = self.spatial_graph.edge.get_edge_segment(tmp_e_ind, tmp_e_ind[0], source_ind) if partial_cc_Q else tmp_e_ind
                        tmp_dir = -1
                    elif tmp_e_cnt_node[1] == node_path[1]:
                        tmp_ind = self.spatial_graph.edge.get_edge_segment(tmp_e_ind, source_ind, tmp_e_ind[-1]) if partial_cc_Q else tmp_e_ind
                        tmp_dir = 1
                    else: 
                        raise ValueError(f"Node {node_path[1]} is not connected to edge {edge_path[i]}")
                    
                    tmp_node_ind = self.spatial_graph.node.cc_ind[node_path[i]] if (not partial_cc_Q) else np.array([], dtype=tmp_ind.dtype)

                elif i < (edge_path.size - 1):
                    tmp_ind = tmp_e_ind
                    if tmp_e_cnt_node[0] == node_path[i]:
                        tmp_dir = 1
                    elif tmp_e_cnt_node[1] == node_path[i]:
                        tmp_dir = -1
                    else: 
                        raise ValueError(f"Node {node_path[i]} is not connected to edge {edge_path[i]}")
                    
                    tmp_node_ind = self.spatial_graph.node.cc_ind[node_path[i]]   

                elif i == (edge_path.size - 1):
                    partial_cc_Q = self.is_dummy_node_Q(node_path[-1])
                    if tmp_e_cnt_node[0] == node_path[-2]:
                        tmp_ind = self.spatial_graph.edge.get_edge_segment(tmp_e_ind, tmp_e_ind[0], target_ind) if partial_cc_Q else tmp_e_ind
                        tmp_dir = 1
                    elif tmp_e_cnt_node[1] == node_path[-2]:
                        tmp_ind = self.spatial_graph.edge.get_edge_segment(tmp_e_ind, target_ind, tmp_e_ind[-1]) if partial_cc_Q else tmp_e_ind
                        tmp_dir = -1
                    else: 
                        raise ValueError(f"Node {node_path[-2]} is not connected to edge {edge_path[i]}")
                    # Does not need to include the node voxel 
                    tmp_node_ind = self.spatial_graph.node.cc_ind[node_path[i]]   
                    tmp_final_node_ind = self.spatial_graph.node.cc_ind[node_path[i + 1]] if (not partial_cc_Q) else np.array([], dtype=tmp_ind.dtype)

                # Direction for the node voxels is ill-defined 
                if tmp_dir == -1: 
                    tmp_ind = tmp_ind[::tmp_dir]
                if include_node_Q:
                    if i < (edge_path.size - 1):
                        tmp_ind = np.concatenate((tmp_node_ind, tmp_ind))
                    else:
                        tmp_ind = np.concatenate((tmp_node_ind, tmp_ind, tmp_final_node_ind))
                
                path_edge_dir.append(tmp_dir)
                tmp_dir = np.repeat(tmp_dir, tmp_ind.size).astype(np.int8)
                path_ind.append(tmp_ind)
                path_vxl_dir.append(tmp_dir)
            
            path_ind = np.concatenate(path_ind)
            path_vxl_dir = np.concatenate(path_vxl_dir)
            path_edge_dir = np.asarray(path_edge_dir)
            
        if return_edge_path_Q: 
            return path_ind, path_vxl_dir, edge_path, path_edge_dir
        else: 
            return path_ind, path_vxl_dir

    @staticmethod
    def compute_path_dir_in_one_edge(left_node_label, path):
        path_dir = 0
        if path.size == 2:
            path_dir = 1 if (path[0] < path[1]) else -1
        elif path.size > 2:
            next_node = path[1]
            if next_node == left_node_label:
                path_dir = -1
            else: 
                path_dir = 1
        return path_dir

    @staticmethod
    def compute_path_dir_in_edges(left_node_labels, paths):
        left_node_labels = np.asarray(left_node_labels)
        paths = np.asarray(paths)
        if left_node_labels.size == 1 and paths.size >= 1:
            left_node_labels = np.repeat(left_node_labels, paths.size)
        assert left_node_labels.size == paths.size, "paths should have the same number of elements as left_node_labels"

        if paths.dtype == 'O':
            path_dir = np.zeros(paths.size)
            for i, p in enumerate(paths):
                path_dir[i] = AbstractGraph.compute_path_dir_in_one_edge(left_node_labels[i], p)
            return path_dir
        else: 
            return np.asarray(AbstractGraph.comptue_path_dir_in_one_edge(left_node_labels, paths))

    @staticmethod
    def _parse_spatial_graph(vsl_graph: SpatialGraph):
        """
        Args: 
            vsl_graph (graph.SpatialGraph): an instance of graph.SpatialGraph.

        Returns: 
            tuple: A tuple containing: 
                - eff_vsl_length (np.ndarray): A 1D numpy array of the effective length of each edge, including the distance to the connected nodes (if exist)
                - new_node_pair (np.ndarray): A 2D numpy array of shape (N, 2), where each row is the node label the edge connected to.
                - eff_ep_pos (np.ndarray): 
                - extended_edge_cc_pos (np.ndarray): (N + 2, 3), each row is the zyx coordinate of the skeleton voxel in the edge. The first and last rows are the coordiante of the two endpoint / connected node centroids. 

        """
        new_node_pair = vsl_graph.edge.connected_node_label.copy() # 0-based indexing. Endpoints are -1
        eff_vsl_length, eff_ep_pos, extended_edge_cc_pos = vsl_graph.compute_edge_effective_length() # return sorted edge_pos 

        # Get the map from new node lable 
        current_new_node_label = vsl_graph.node.num_cc # node label starts at 0
        for e_label in range(new_node_pair.shape[0]):
            e_node_pair = new_node_pair[e_label]
            # Make all cc_ind voxels start near node 1 and end near node 2
            if np.all(e_node_pair == -1):
                # Unconnected edge - does not need to sort
                new_node_pair[e_label][0] = current_new_node_label
                current_new_node_label += 1
                
                new_node_pair[e_label][1] = current_new_node_label
                current_new_node_label += 1
            elif np.any(e_node_pair == -1):
                # Edge with one unconnected endpoint
                assert e_node_pair[1] == -1, 'Assuming the first label is valid'
                new_node_pair[e_label][1] = current_new_node_label
                current_new_node_label += 1
        return eff_vsl_length, new_node_pair, eff_ep_pos, extended_edge_cc_pos
    
    @staticmethod
    def show_graph_differences(G1, G2):
        nodes_diff = set(G1.nodes()) ^ set(G2.nodes())
        edges_diff = set(G1.edges()) ^ set(G2.edges())
        
        print("Different nodes:", nodes_diff)
        print("Different edges:", edges_diff)
        return nodes_diff, edges_diff
    
    @staticmethod
    def find_ind_in_extended_sub_array(sub_array, ind:int, mask_size):
        """
        Input: 
            sub_array: extended edge subscript. [node_1/ep1, vxl1, ... vxln, node_2/ep2]
                If the edge contians endpoint(s), its subscript (3D) is duplicated. 
        
        """
        sub_array = np.asarray(sub_array, dtype=np.int16)
        edge_ind = np.ravel_multi_index((sub_array[0], sub_array[1], sub_array[2]), mask_size)
        edge_ind[[0, -1]] = -1
        i = find_first_idx_numba(edge_ind, ind)
        assert i>= 0, ValueError(f"Ind {ind} is not in the array")
        return i

        # sub = np.hstack(np.unravel_index(ind, mask_size))[:, None]
        # num_sub = sub_array.shape[1]
        # assert sub.shape == (3, 1), "ind should be a integer scalar"
        # assert sub_array.shape[0] == 3, "sub_array should be a (3, n) numpy integer array"
        # match_Q = np.all(sub_array == sub, axis=0)
        # idx = np.nonzero(match_Q)[0]
        # if idx.size == 1:
        #     return idx[0]
        # elif idx.size == 2:
        #     if idx[0] == 0:
        #         return idx[1]
        #     elif idx[-1] == (num_sub - 1):
        #         return idx[0]
        #     else:
        #         raise ValueError(f"Unexpected value {idx}")
        # else:
        #     raise ValueError(f"Unexpected value {idx}")

    @staticmethod
    def _construct_dictionary_graph(fg:SpatialGraph, weighted_Q=True): 
        e_l, n_pair, _, _ = AbstractGraph._parse_spatial_graph(fg)
        if not weighted_Q:
           # overwright
           e_l = np.ones(e_l.shape, np.uint16)

        dd_g = defaultdict(list)
        for idx, (u, v) in enumerate(list(n_pair)):
            w = e_l[idx]
            dd_g[u].append((v, w, idx))
            dd_g[v].append((u, w, idx))
        return dd_g, e_l, n_pair

    @staticmethod
    def _compute_shortest_path_length_skip_single_edge(dd_g, skip_idx, n_pair, weighted_Q=True, return_node_path_Q=False): 
        num_g_n = len(dd_g.keys())
        src, target = n_pair[skip_idx]

        heap = [(0, 0, 0, src)]
        dist = [float('inf')] * num_g_n
        dist[src] = 0

        num_node = [float('inf')] * num_g_n
        num_node[src] = 0

        visited = [False] * num_g_n
        if return_node_path_Q: 
            parent = [None] * num_g_n
        
        path = []
        final_ed = float('inf')
        final_gd = float('inf')
        while heap: 
            d, ed, gd, node = heapq.heappop(heap)
            if node == target: 
                if return_node_path_Q: 
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path = path[::-1]
                final_ed = ed
                final_gd = gd
                
            if visited[node]: 
                continue
            visited[node] = True

            for nei, weight, edge_idx in dd_g[node]:
                if edge_idx == skip_idx: 
                    continue
                if weighted_Q: 
                    new_dist = d + weight
                else: 
                    new_dist = d + 1

                if dist[nei] > new_dist: 
                    dist[nei] = new_dist
                    if return_node_path_Q: 
                        parent[nei] = node
                    heapq.heappush(heap, (dist[nei], ed + weight, gd + 1, nei))

        if return_node_path_Q: 
            return final_ed, final_gd, path
        else: 
            return final_ed, final_gd
        
    @staticmethod
    def compute_shortest_loop_length(fg, edge_list=None, weighted_Q=True):
        # Always construct a weighted graph
        dd_g, e_l, n_pair = AbstractGraph._construct_dictionary_graph(fg, weighted_Q=True)
        if edge_list is None: 
            edge_list = np.arange(e_l.size, dtype=np.uint32)
        else: 
            edge_list = np.asarray(edge_list)
        
        loop_e_l, loop_g_l = (np.full(edge_list.size, np.nan) for _ in range(2))
        for i, e in enumerate(list(edge_list)): 
            # tmp_l: shortest path from one node of an edge to another node without passing the edge
            # tmp_g: number of edges in the shortest path 
            tmp_l, tmp_g = AbstractGraph._compute_shortest_path_length_skip_single_edge(dd_g, e, n_pair, weighted_Q=weighted_Q, return_node_path_Q=False)
            if np.isfinite(tmp_l):
                loop_e_l[i] = tmp_l + e_l[e]
                loop_g_l[i] = tmp_g + 1
        
        return loop_e_l, loop_g_l
#region
    
@njit(cache=True)
def find_first_idx_numba(edge_ind, ind):
    for i in range(1, edge_ind.size - 1):
        if edge_ind[i] == ind:
            return i
    return -1  # Or raise error from Python wrapper