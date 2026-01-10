import logging, warnings
from collections import defaultdict, OrderedDict
import numpy as np
# import plotly.express as px

from .utils import graph
from .utils import neighbors as nb
from .utils import util
from .utils import vis

from network_flow_tracker.EdgeFlow import EdgeFlow

class FlowGraph(graph.SpatialGraph):
    def __init__(self, vsl_skl):
        """
        Initialize the FlowGraph object.
        Input: 
            vsl_skl: 1D numpy array, linear indices of the vessel skeleton 
            
        """
        super().__init__(vsl_skl)
        self.abs_g = graph.AbstractGraph(self)

    def init_nearest_skl_map(self, vsl_skl_mask=None, vsl_skl_labeled=None, vsl_vol_labeled=None): 
        if (vsl_skl_mask is not None) and (vsl_skl_labeled is None) and (vsl_vol_labeled is None): 
            self.nearest_map = nb.NearestMaskVoxel(vsl_skl_mask, save_data_Q=False, val_0_based_label_Q=False)
        elif (vsl_skl_labeled is not None) and (vsl_vol_labeled is not None): 
            self.nearest_map = NearestSkeletonVoxel(vsl_skl_labeled=vsl_skl_labeled, vsl_vol_labeled=vsl_vol_labeled)
        else: 
            raise NotImplementedError
    
    def skl_dist(self, X, Y):
        """
            X, Y: 3D subscripts of the skeleton voxels
        """
        e_dist = np.sqrt(np.sum((X - Y) ** 2, axis=0))
        X_ind = self.nearest_map.sub_to_nearest_ind(np.round(X).astype(np.int64))
        Y_ind = self.nearest_map.sub_to_nearest_ind(np.round(Y).astype(np.int64))
        if X_ind == Y_ind: 
            return e_dist
        else: 
            g_dist = float(self.abs_g.compute_shortest_path_length_between_two_voxel_indices(X_ind, Y_ind, 
                                                                                             cached_Q=True, clear_cache_Q=False))
            if g_dist < e_dist: 
                return e_dist
            else: 
                return g_dist
    
#region Prediction

    def init_velocity(self, edge_v_pxl, v_std_pxl=None, edge_track_frac=None):
        self.edge_v_pxl = edge_v_pxl
        self.edge_v_std_pxl = v_std_pxl
        self.edge_track_frac = edge_track_frac            
        self.ind_to_ep_dist = {}
        self.e_ep1_dist = []
        self.e_length = self.abs_g.length
        for i, e_edge_ext_zyx in enumerate(self.abs_g.edge_cc_pos): 
            # e_edge_ext_zyx = fg.abs_g.edge_cc_pos[p_e_l]            
            adj_dist = np.sqrt(np.sum(np.diff(e_edge_ext_zyx, axis=1) ** 2, axis=0)) 
            ep_dist = np.cumsum(adj_dist)
            e_ext_len = ep_dist[-1]
            ep_dist = ep_dist[:-1]
            # ep_dist = np.stack((ep_dist, e_ext_len - ep_dist), axis=1)
            e_edge_int_ind = self.edge.sub2ind(e_edge_ext_zyx[:, 1:-1])
            assert np.all(e_edge_int_ind == self.edge.cc_ind[i])
            assert (self.e_length[i] - e_ext_len) < 1e-2, f"Inconsistent edge length {self.e_length[i]} vs {e_ext_len}"
            for i, ind in enumerate(e_edge_int_ind):
                assert ind not in self.ind_to_ep_dist, f"skl_ind {ind} has been added"
                self.ind_to_ep_dist[ind] = ep_dist[i]
            
            self.e_ep1_dist.append(ep_dist)
        
    def move_particle_along_one_edge(self, init_skl_ind, move_dist=None, edge_label=None, move_by_time_Q=False, 
                                     v_pxl=np.nan, dt=np.nan, inferred_v_Q=False):

        if edge_label is None: 
            edge_label = self.edge.ind_to_label(init_skl_ind)
        
        assert edge_label >= 0, f"Skl voxel {init_skl_ind} is not an edge voxel"

        if move_by_time_Q: 
            assert (move_dist is None) and (np.isfinite(v_pxl) and (dt >= 0)), ValueError(f"Invaid input argument value.")
            move_dist = v_pxl * dt
        else: 
            assert move_dist is not None, ValueError(f"move_dist should be a numerical scalar")
            # ignore v_pxl and dt

        info = {'init_ind': init_skl_ind, 'exit_network_Q': False, 'ind': np.nan, 'sub': np.full((3, ), np.nan), 'edge_label': edge_label, 
                'move_dist': move_dist, 'v_pxl': v_pxl, 'move_dt_Q': move_by_time_Q, 'exit_ep_sub': np.full((3, ), np.nan), 
                'dist_to_exit_ep': np.nan, 'inferred_v_Q': inferred_v_Q}        

        to_exit_node = self.edge.connected_node_label[edge_label][0] if move_dist < 0 else self.edge.connected_node_label[edge_label][1]
        to_exit_ep_ind = self.edge.cc_ind[edge_label][0] if move_dist < 0 else self.edge.cc_ind[edge_label][-1]

        p_e_len = self.e_length[edge_label]
        p_dist_to_ep = self.ind_to_ep_dist[init_skl_ind]
        p_dist_to_ep1 = p_dist_to_ep + move_dist
        if p_dist_to_ep1 >= 0 and p_dist_to_ep1 <= p_e_len: 
            # ends within the same edge
            if move_dist != 0: 
                p_tp1_idx = np.argmin(np.abs(self.e_ep1_dist[edge_label] - p_dist_to_ep1))
                tp1_skl_ind = self.edge.cc_ind[edge_label][p_tp1_idx]
            else: 
                tp1_skl_ind = init_skl_ind
            info['ind'] = tp1_skl_ind
            info['sub'] = np.asarray(self.ind2sub(tp1_skl_ind))
            if to_exit_node < 0: 
                # Signed distance to the exit endpoint. > 0 if the point is still in the graph
                info['dist_to_exit_ep'] = p_dist_to_ep1 if move_dist < 0 else (p_e_len - p_dist_to_ep1)
            return [info]
        
        else: 
            # ends in another edge
            if p_dist_to_ep1 < 0: 
                remaining_dist = - p_dist_to_ep1
            elif p_dist_to_ep1 > p_e_len: 
                remaining_dist = p_dist_to_ep1 - p_e_len

            if to_exit_node >= 0: 
                if move_by_time_Q: 
                    remaining_time = np.abs(remaining_dist / v_pxl)
                    result = self.move_particle_from_one_node(to_exit_node, move_dist=None, previous_e_label=edge_label, 
                                                            move_by_time_Q=move_by_time_Q, prev_v_pxl=np.abs(v_pxl), 
                                                            dt=remaining_time, inferred_v_Q=inferred_v_Q)
                else: 
                    # move by distance
                    result = self.move_particle_from_one_node(to_exit_node, move_dist=remaining_dist, previous_e_label=edge_label, 
                                                            move_by_time_Q=move_by_time_Q, prev_v_pxl=np.nan, dt=np.nan, 
                                                            inferred_v_Q=inferred_v_Q)
                return result
            else: 
                # print("Particle exits the network")
                info['exit_network_Q'] = True
                info['exit_ep_sub'] = np.asarray(self.ind2sub(to_exit_ep_ind))
                info['dist_to_exit_ep'] = - remaining_dist # Signed distance to the exit endpoint. < 0 if the predicted position is outside the graph
                return [info]
            
    def move_particle_from_one_node(self, start_node, move_dist=None, previous_e_label=None, 
                                    move_by_time_Q=False, prev_v_pxl=np.nan, dt=np.nan, inferred_v_Q=False):
        
        if move_by_time_Q: 
            if np.isnan(dt): 
                assert (move_dist is not None) and (np.isfinite(prev_v_pxl)), ValueError(f"dt is not given. Need to provide move_dist and previous_v")
                dt = move_dist / np.abs(prev_v_pxl)
            else: 
                assert dt >= 0, ValueError("dt must be non-netagive")
                assert prev_v_pxl >=0, ValueError("prev_v_pxl should be non-negative")
                assert move_dist is None, ValueError("When moving by time, move_dist should be None. Will be inferred from the dt and velocity direclty.")
                move_dist = prev_v_pxl * dt
        else: 
            # move by a fix distance, independent of edge velocity
            assert move_dist >= 0, ValueError(f"move_dist should be a finite scalar when moving by distance. ")
            # assert dt is None and previous_v is None, ValueError
                
        result = []
        downstream_edge_label = self.node.connected_edge_label[start_node]
        if previous_e_label is not None: 
            downstream_edge_label = downstream_edge_label[downstream_edge_label != previous_e_label]

        for next_e_label in downstream_edge_label: 
            # Need to check if the downstream edge is inflow or outflow 
            next_edge_connected_node = self.edge.connected_node_label[next_e_label]
            if next_edge_connected_node[0] == start_node: 
                next_skl_ind = self.edge.cc_ind[next_e_label][0]
                start_ep1_Q = True
            elif next_edge_connected_node[1] == start_node: 
                next_skl_ind = self.edge.cc_ind[next_e_label][-1]
                start_ep1_Q = False
            else: 
                raise ValueError(f"The exit node {start_node} is not connected to edge {next_e_label}")

            if move_by_time_Q:
                next_edge_v = self.edge_v_pxl[next_e_label] if hasattr(self, 'edge_v_pxl') else np.nan
                if np.isfinite(next_edge_v): 
                # Use prior estimated edge velocity to estiamte the distance 
                # the particle can travel within one time step - overwrite 
                # move_dist
                    if (start_ep1_Q and next_edge_v >= 0) or (not start_ep1_Q and next_edge_v <= 0): 
                        # Use the remaining time and edge velocity to update the move_dist in the edge
                        # if the speed has been inferred previously, keep this log 
                        tmp_result = self.move_particle_along_one_edge(next_skl_ind, move_dist=None, edge_label=next_e_label,
                                            move_by_time_Q=move_by_time_Q, v_pxl=next_edge_v, dt=dt, inferred_v_Q=inferred_v_Q)
                    else: 
                        # Inflow edge, no need to compute
                        continue   
                else: 
                    # assume the same speed
                    next_edge_v = prev_v_pxl if start_ep1_Q else (- prev_v_pxl)
                    tmp_result = self.move_particle_along_one_edge(next_skl_ind, move_dist=None, edge_label=next_e_label,
                                                        move_by_time_Q=move_by_time_Q, v_pxl=next_edge_v, dt=dt, inferred_v_Q=True)
            else: 
                e_move_dist = move_dist if start_ep1_Q else (- move_dist)
                tmp_result = self.move_particle_along_one_edge(next_skl_ind, move_dist=e_move_dist, edge_label=next_e_label,
                                                        move_by_time_Q=move_by_time_Q, v_pxl=np.nan, dt=np.nan, inferred_v_Q=False)
            
            result.extend(tmp_result)

        return result

    def predict_single_particle_position(self, pos_zyx, dt=1, p_skl_ind=None, 
                                         edge_label=None, noise_cv=0.0):
        pos_zyx = np.asarray(pos_zyx)
        assert pos_zyx.shape == (3, ), 'pos_zyx must be a (3, ) numpy array'
        
        if p_skl_ind is None: 
            p_skl_ind = self.nearest_map.zyx_to_nearest_ind(pos_zyx)

        if edge_label is None: 
            edge_label = self.edge.ind_to_label(p_skl_ind)

        info = {'init_ind': p_skl_ind, 'exit_network_Q': False, 'ind': np.nan, 'sub': self.ind2sub(p_skl_ind), 
                'edge_label': edge_label, 'move_dist': None, 'v_pxl': np.nan, 'move_dt_Q': True, 'exit_ep_sub': np.full((3, ), np.nan), 
                'dist_to_exit_ep': np.nan, 'inferred_v_Q': False}

        result = []
        if edge_label >= 0: 
            p_e_v = self.edge_v_pxl[edge_label]
            if noise_cv > 0: 
                p_e_v = np.random.normal(p_e_v, scale=np.abs(p_e_v) * noise_cv)

            if np.isfinite(p_e_v): 
                result = self.move_particle_along_one_edge(p_skl_ind, move_dist=None, edge_label=edge_label, 
                                                                move_by_time_Q=True, v_pxl=p_e_v, dt=dt, inferred_v_Q=False)        
        else: 
            p_n_l = self.node.ind_to_label(p_skl_ind)
            if p_n_l >= 0: 
                result = self.move_particle_from_one_node(p_n_l, move_dist=None, previous_e_label=None, 
                                                                move_by_time_Q=True, prev_v_pxl=0, dt=dt, inferred_v_Q=False)

        result = [info] if len(result) == 0 else result                
        return result
    
    @staticmethod
    def parse_single_particle_next_frame_predictions(predict_result:list): 
        """Parse the result from predict_single_particle_position. 
        Combine predicted next-frame-position in the network with the 
        exit end point position. 
        Input: 
            predict_result: list, output of predict_single_particle_position
        Output: 
            result: dict. 
        """
        predict_sub = []
        last_known_v_pxl = []
        exit_extra_travel = []
        if len(predict_result) > 0: 
            for i, pred in enumerate(predict_result): 
                last_known_v_pxl.append(np.abs(pred['v_pxl'])) # last known velocity
                exit_extra_travel.append(pred['dist_to_exit_ep'])
                if np.all(np.isfinite(pred['sub'])):
                    predict_sub.append(pred['sub'])
                elif np.all(np.isfinite(pred['exit_ep_sub'])):
                    predict_sub.append(pred['exit_ep_sub'])
                else: 
                    raise ValueError(f"Both 'sub' and 'exit_ep_sub' are invalid: {pred}")
        
        result = {}
        result['sub'] = np.asarray(predict_sub)
        result['abs_v'] = np.asarray(last_known_v_pxl)
        result['dist_to_exit_ep'] = np.asarray(exit_extra_travel)
        return result

    def predict_particles_next_frame_skl_sub(self, pos_zyx, dt=1):
        """
        Input: 
            pos_zyx: (N, 3) np.array, coordinate of N particle in 3D
        Output: 
            result: dict
                num_pos: (N, ) np.array. num_pos[i] is the number of 
                    the predicted position for the i-th particle 
                sub: [i] (n, 3) np.array, where n == num_pos[i]


        
        """
        pos_zyx = np.atleast_2d(np.asarray(pos_zyx))
        assert pos_zyx.shape[1] == 3, f"pos_zyx must be (N, 3) np.ndarray"
        pos_tp1 = []
        abs_v = []
        extra_travel = []
        p_skl_ind = self.nearest_map.zyx_to_nearest_ind(pos_zyx.T)
        p_edge_label = self.edge.ind_to_label(p_skl_ind)

        for i, pos in enumerate(pos_zyx): 
            tmp_predict = self.predict_single_particle_position(pos, dt=dt, p_skl_ind=p_skl_ind[i], edge_label=p_edge_label[i])
            tmp_predict = self.parse_single_particle_next_frame_predictions(tmp_predict)
            pos_tp1.append(tmp_predict['sub'])
            abs_v.append(tmp_predict['abs_v'])
            extra_travel.append(tmp_predict['dist_to_exit_ep'])
        
        result = {}
        result['num_pos'] = np.array([x.shape[0] for x in pos_tp1]) # number of predicted position
        result['sub'] = np.vstack([p for p in pos_tp1 if p.shape[0] > 0])
        result['abs_v'] = np.concatenate([v for v in abs_v if v.size > 0])
        result['dist_to_exit_ep'] = np.concatenate([v for v in extra_travel if v.size > 0])
        result['idx'] = np.repeat(np.arange(result['num_pos'].size), result['num_pos'])
        # result['p_idx_end'] = np.cumsum(result['num_pos'])
        # result['p_idx_start'] = np.concatenate(([0], result['p_idx_end'][:-1]))
        return result
    
    def configure_trackpy_query(self, search_range, num_nb=10, max_exit_travel=0, predictQ=True, gdistQ=True, 
                                v_error_frac=0.5, v_error_min=5, compare_feature=[], feature_cost_max=10):
        self.tpq_config = {'search_range': search_range, 'num_nb': num_nb, 'max_exit_travel': max_exit_travel, 
                           'predictQ': predictQ, 'gdistQ': gdistQ, 
                           'v_error_frac': v_error_frac, 'v_error_min': v_error_min, 
                           'compare_feature': compare_feature, 'feature_cost_max': feature_cost_max}
        
        print(f"Finish configuring trackpy query function. ", self.tpq_config)

    def trackpy_query(self, source_hash, dest_hash, search_range, num_nb):
        config = self.tpq_config
        if config['predictQ']:
            dists, inds = self.trackpy_search_nb_with_prediction(source_hash, dest_hash, search_range=search_range, 
                                                                    num_nb=num_nb,  
                                                                    v_error_frac=config['v_error_frac'], 
                                                                    v_error_min=config['v_error_min'])
        else: 
            dists, inds = self.trackpy_search_nb(source_hash, dest_hash, search_range, max_neighbors=num_nb, 
                                                 on_graph_Q=config['gdistQ'])
            
        if len(config['compare_feature']) > 0: 
            # diff_score = 1 - similarity_score, where similarity_score is in [0, 1], i.e. probability. 
            diff_score = self.trackpy_compute_pair_similarity(source_hash, dest_hash, dists, inds, config['compare_feature'])
            dists, inds = self.trackpy_add_similarity_score(dists, inds, config['feature_cost_max'] * diff_score)

        return dists, inds

    def trackpy_search_nb_with_prediction(self, source_hash, dest_hash, search_range, num_nb=10, 
                                          v_error_frac=np.nan, v_error_min=np.nan):
        """
        Input: 
            source_hash, desh_hash: Trackpy object, property of Subnets.
            search_range: maximum distance between the predicted particle 
                position and the particles in the next frame 
                To do: enable adative search range. 
            num_nb: number of neighbors in the knn search 
            max_exit_travel: maximum extra exit travel for a exiting particle
                to be added to the predicted particle position array. 
            v_error_frac: prediction position error = v * v_error_frac. if not valid, use search_range
            v_error_min: minimum prediction position error, pixel
        Output: 
            p_dist: the [i, j] element is the distance between the j-th nearest neighbor in (t + 1) frame
                of the i-th particle before (t + 1) frame. Each row is sorted in ascending order. 
            p_inds: the [i, j] element is the list index of j-th nearest neighbor in (t + 1) frame
                of the i-th particle before (t + 1) frame. 
                
        """
        # Predict particle positions
        source_sub = source_hash.coords_mapped
        dest_sub = dest_hash.coords_mapped
        num_source = source_sub.shape[0]
        dest_no_match_ind = dest_hash.coords_mapped.shape[0]
        # one source particle might have more than 1 predicted positions
        source_predict = self.predict_particles_next_frame_skl_sub(source_sub)

        # Computed the geodesic distnace between the predicted positions and their nearest neighbors
        if v_error_frac >= 0 and v_error_min >= 0: # Compute maximum prediction error
            est_err = np.maximum(v_error_min, source_predict['abs_v'] * v_error_frac) # (num_predict, )
            est_err[np.isnan(est_err)] = search_range
        else: 
            est_err = np.full((source_predict['abs_v'].size, ), search_range)
        assert np.all(est_err > 0), ValueError("est_err contains invalid value") 

        # Find predicted positions that are (1) inside the network or (2) outside the network but within the search range (negative dist_to_exit_ep)
        need_query_Q = (source_predict['dist_to_exit_ep'] >= - (search_range)) 
        need_query_idx = np.where(need_query_Q)[0]

        source_query_sub = source_predict['sub'][need_query_Q, :]
        assert np.any(need_query_Q), "Predict no particle to infer. Debug"
        # Get k-nn next frame particles near each predicted positions (one source particle might have more than 1 predicted positions)
        dists, inds = dest_hash.query(source_query_sub, num_nb, rescale=False, search_range=search_range)

        s_num_nb = np.sum(np.isfinite(dists), axis=1)
        for i in range(need_query_idx.size): # for each predicted position, select neighbors within est_err
            i_source_idx = need_query_idx[i]
            i_dist_to_ep = source_predict['dist_to_exit_ep'][i_source_idx]
            i_exit_travel = i_dist_to_ep if i_dist_to_ep < 0 else 0
            tmp_g_dist = np.full((num_nb, ), np.inf)
            for j in range(s_num_nb[i]): 
                # Adding extra cost for the particles that exit the network
                tmp_g_dist[j] = np.abs(self.skl_dist(source_query_sub[i], dest_sub[inds[i, j]]) + i_exit_travel)
            
            tmp_in_range_Q = (tmp_g_dist <= est_err[i_source_idx])
            if (not np.any(tmp_in_range_Q)) and (est_err[i_source_idx] < search_range): 
                # relax the search range to search_range if no neighbor found within est_err
                tmp_in_range_Q = (tmp_g_dist <= search_range)            
            
            tmp_g_dist[~tmp_in_range_Q] = np.inf
            dists[i] = tmp_g_dist
            inds[i, ~tmp_in_range_Q] = dest_no_match_ind

            assert np.all(dists[i, s_num_nb[i]:] == np.inf), ValueError("Expect np.inf")

        # merge results        
        source_idx_to_ind = util.bin_data_to_idx_list(source_predict['idx'][need_query_Q], return_type='dict')
        s_idx_end = np.cumsum(source_predict['num_pos'])
        s_idx_start = np.concatenate(([0], s_idx_end[:-1]))
        p_dist = np.full((num_source, num_nb), np.inf)
        p_inds = np.full((num_source, num_nb), dest_no_match_ind, dtype=inds.dtype)

        for i_source in range(num_source):
            # keep the top num_nb neighbors for each source particle
            if i_source in source_idx_to_ind: 
                i_idx = source_idx_to_ind[i_source]
                tmp_p_g_dist = dists[i_idx] # always a 2d array (n, nb)
                tmp_inds = inds[i_idx]
                if i_idx.size > 1: # multiple predict positions might have shared nearest neighbors. Find the unique set. 
                    tmp_idx, tmp_nb_inds = util.bin_data_to_idx_list(tmp_inds.flatten())
                    tmp_unq_nb_inds = []
                    tmp_unq_nb_min_gdist = []        
                    for tmp_i, tmp_nb_i in zip(tmp_idx, tmp_nb_inds):
                        if tmp_nb_i != dest_no_match_ind:
                            tmp_min_g_dist = np.min(tmp_p_g_dist.flat[tmp_i])
                            # if tmp_min_g_dist <= search_range:
                            tmp_unq_nb_inds.append(tmp_nb_i)
                            tmp_unq_nb_min_gdist.append(tmp_min_g_dist)
                    
                    tmp_p_g_dist = np.asarray(tmp_unq_nb_min_gdist)
                    tmp_inds = np.asarray(tmp_unq_nb_inds)

                tmp_s_idx = np.argsort(tmp_p_g_dist)
                # select the top #nb candidates
                if tmp_s_idx.size > num_nb: 
                    tmp_s_idx = tmp_s_idx[:num_nb]
                
                p_dist[i_source][0:tmp_s_idx.size] = tmp_p_g_dist.flat[tmp_s_idx]
                p_inds[i_source][0:tmp_s_idx.size] = tmp_inds.flat[tmp_s_idx]

            
            i_idx_w = np.arange(s_idx_start[i_source], s_idx_end[i_source])
            i_dist_2_eep = source_predict['dist_to_exit_ep'][i_idx_w]
            i_sd2eep_valid_Q = (i_dist_2_eep <= est_err[i_idx_w])
            if np.any(i_sd2eep_valid_Q):
                # Add exit information to the source_hash points ['exit_info'] = (ep_ind, dist, exit_speed) 
                # 1. Predicted inside the network, but close to the endpoint 
                # 2. Predicted outside the netowrk - regardless of how large the extra travel is.                 
                i_idx_w = i_idx_w[i_sd2eep_valid_Q]
                i_speed = source_predict['abs_v'][i_idx_w]
                i_sub = source_predict['sub'][i_idx_w]
                i_exit_ep_ind = np.ravel_multi_index((i_sub[:, 0], i_sub[:, 1], i_sub[:, 2]), self.num['mask_size'])

                if i_exit_ep_ind.size == 1: 
                    # if only has one predicted position, log the exit info directly
                    sampled_idx = 0
                else:
                    # if has more than one predicted position, and some of them exit - ???
                    # Option 2: based on child segment velocity
                    if np.all(np.isfinite(i_speed)) and np.sum(i_speed) > 0: 
                        sampled_idx = np.random.choice(np.arange(i_exit_ep_ind.size), 1, p=i_speed/np.sum(i_speed))[0]
                    else: 
                        sampled_idx = np.random.choice(np.arange(i_exit_ep_ind.size), 1)[0]

                source_hash.points[i_source].extra_data['exit_info'] = (i_exit_ep_ind[sampled_idx], \
                                                        i_dist_2_eep[sampled_idx], i_speed[sampled_idx]) 

        return p_dist, p_inds

    def trackpy_search_nb(self, source_hash, dest_hash, search_range, max_neighbors=10, on_graph_Q=True): 
        dists, inds = dest_hash.query(source_hash.coords_mapped, max_neighbors, rescale=False, search_range=search_range)
        dest_no_match_ind = dest_hash.coords_mapped.shape[0]
        if on_graph_Q: 
            nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
            for i, p in enumerate(source_hash.points):
                for j in range(nn[i]):
                    p_d = dest_hash.points[inds[i, j]] # p_d.pos: (3, ) array, same order as the pos_columns
                    g_dist = self.skl_dist(p.pos, p_d.pos)
                    if g_dist <= search_range: 
                        dists[i, j] = g_dist 
                    else: 
                        dists[i, j] = np.inf
                        inds[i, j] = dest_no_match_ind
                # Sort dists and inds to make the output consistent 
                s_idx = np.argsort(dists[i])
                dists[i] = dists[i][s_idx]
                inds[i] = inds[i][s_idx]
        
        return dists, inds

    def trackpy_compute_pair_similarity(self, source_hash, dest_hash, dists, inds, features):
        
        s_para = {'peak_int': {'min_std': 1000, 'cv': 0.5}}

        s_num_nb = np.sum(np.isfinite(dists), axis=1)
        num_source = len(source_hash.points)
        source_features = {k: np.full((num_source, ), np.nan, dtype=np.float32) for k in features}
        for i, p in enumerate(source_hash.points):
            for k in features: 
                if k in p.extra_data: 
                    source_features[k][i] = p.extra_data[k]
        
        num_dest = len(dest_hash.points)
        dest_features = {k: np.full((num_dest, ), np.nan, dtype=np.float32) for k in features}
        for i, p in enumerate(dest_hash.points):
            for k in features: 
                if k in p.extra_data: 
                    dest_features[k][i] = p.extra_data[k]

        feature_diff = np.zeros(dists.shape, dtype=np.float32)
        # Can be completely vectorized if needed. 
        for i in range(feature_diff.shape[0]):
            tmp_similarity = 1
            for k in features: 
                tmp_s_val = source_features[k][i]
                tmp_d_val = dest_features[k][inds[i, :s_num_nb[i]]]
                if k == 'peak_int': 
                    tmp_para = s_para[k]
                    tmp_p = (tmp_s_val - tmp_d_val) / (np.sqrt(2) * np.maximum(tmp_para['min_std'], tmp_para['cv'] * tmp_s_val))
                    tmp_p = np.exp(- tmp_p ** 2)
                else: 
                    raise NotImplementedError
                    # tmp_diff += np.abs(tmp_s_val - tmp_d_val) / tmp_s_val

                tmp_similarity *= tmp_p
            tmp_diff = 1 - tmp_similarity
            feature_diff[i, :s_num_nb[i]] = tmp_diff
        
        assert np.all(np.isfinite(feature_diff)), "Exist invalid similarity score"

        return feature_diff
    
    def trackpy_add_similarity_score(self, dists, inds, feature_diff):
        # TODO: need to adjust cost for the exciting particles? 
        s_num_nb = np.sum(np.isfinite(dists), axis=1)
        for i in range(dists.shape[0]): 
            tmp_nb = s_num_nb[i]
            new_score = dists[i, :tmp_nb] + feature_diff[i, :tmp_nb]
            sorted_idx = np.argsort(new_score)
            dists[i, :tmp_nb] = new_score[sorted_idx]
            inds[i, :tmp_nb] = inds[i, sorted_idx] 
        
        return dists, inds
 
#endregion

#region Flow map 
    def construct_vxl_speed_map(self, trace_result, long_traces_ind, include_node_Q=False, return_type='dict', update_fg_Q=True, particle_key='particle'): 
        num_frame = trace_result.frame.max() + 1
        # Initialize as zeros first, convert to nan later
        vsl_speed_dict = {i : np.zeros(num_frame) for i in self.pos_ind}
        vsl_count_dict = {i : np.zeros(num_frame) for i in self.pos_ind}
        trace_result['v'] = np.nan
        v_col_idx = trace_result.columns.get_loc('v')
        e_passing_cell = defaultdict(list)
        for i_trace in range(long_traces_ind.size):
            p_t_idx = long_traces_ind[i_trace]
            p_table = trace_result.iloc[p_t_idx]
            p_num_frame = p_table.shape[0]

            p_pos = p_table[['x', 'y', 'z']].to_numpy()
            p_id = p_table[particle_key].to_numpy()
            p_ds = np.sqrt(np.sum(np.diff(p_pos, axis=0) ** 2, axis=1))
            skl_ind = p_table.skl_ind.values
            p_e_label = p_table.edge_label.values
            
            exit_v = None
            exit_ind = None
            if 'exit_ind' in p_table: 
                exit_ind = int(p_table.exit_ind.values[-1])
                if exit_ind < 0: 
                    exit_ind = None
                

            for tmp_idx in range(p_num_frame):
                tmp_t1 = p_table.frame.iloc[tmp_idx]
                tmp_ind_1 = skl_ind[tmp_idx]

                if tmp_idx < p_num_frame - 1: 
                    tmp_t2 = p_table.frame.iloc[tmp_idx + 1]
                    tmp_ind_2 = skl_ind[tmp_idx + 1]
                    tmp_dist = p_ds[tmp_idx]
                elif exit_ind is not None: 
                    tmp_t2 = tmp_t1 + 1
                    tmp_ind_2 = exit_ind
                    tmp_dist = 0
                    exit_v = p_table.exit_v.values[-1]
                else: 
                    continue

                tmp_num_tp = tmp_t2 - tmp_t1
                if tmp_ind_1 == tmp_ind_2: 
                    tmp_ind = np.asarray([tmp_ind_1])
                    tmp_speed = tmp_dist
                    tmp_v = np.asarray([tmp_speed])
                    tmp_edge_path = p_e_label[tmp_idx]
                    tmp_edge_path = tmp_edge_path[tmp_edge_path >= 0]
                    # The direction is ill-defined here. Can compute the angle between the velocity and the 
                    # local segment voxel displacement vector
                    tmp_edge_dir = [1] # this is an approximation. 
                else: 
                    tmp_path_node, tmp_path_len = self.abs_g.compute_shortest_path_between_two_voxel_indices(tmp_ind_1, tmp_ind_2)
                    tmp_ind, tmp_dir, tmp_edge_path, tmp_edge_dir = self.abs_g.get_voxel_path_from_node_path(tmp_path_node, tmp_ind_1, tmp_ind_2, include_node_Q=include_node_Q, 
                                                                                               return_edge_path_Q=True)
                    tmp_path_len = np.maximum(tmp_path_len, tmp_dist)

                    tmp_speed = tmp_path_len / tmp_num_tp

                    if tmp_idx == (p_num_frame - 1) and (exit_ind is not None): 
                        assert (exit_v is not None) and (np.isfinite(exit_v)), ValueError(f"Invalid exit speed value {exit_v}")
                        tmp_speed = np.maximum(exit_v, tmp_speed)

                    tmp_v = tmp_dir * tmp_speed
                
                if tmp_speed > 100: 
                    raise ValueError(f"Trace {i_trace} t_idx {tmp_idx} has unreasonably large velocity between vxl {tmp_ind_1} and {tmp_ind_2}. Debug...")

                trace_result.iat[p_t_idx[tmp_idx], v_col_idx] = tmp_speed
                for i, v in zip(tmp_ind, tmp_v):
                    # might need to assign the same v to multiple time points
                    if i in vsl_speed_dict: 
                        for tmp_t in range(tmp_t1, tmp_t2): 
                            vsl_speed_dict[i][tmp_t] += v 
                            vsl_count_dict[i][tmp_t] += 1
                
                for i_e, e in enumerate(tmp_edge_path): 
                    e_dir = tmp_edge_dir[i_e]
                    for tmp_t in range(tmp_t1, tmp_t2): 
                        e_passing_cell[e].append((tmp_t, p_table.did.iloc[tmp_idx], 
                                                  p_id[tmp_idx], tmp_speed * e_dir)) # (time, detection_id, pid, speed_pxl2frame)

            if i_trace % 10 == 0: 
                print(f"\rFinish constructing voxel speed map using ({i_trace+1} of {long_traces_ind.size}) traces. ", end='', flush=True)
        
        for k in vsl_speed_dict.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vsl_speed_dict[k] /= vsl_count_dict[k]
            vsl_speed_dict[k][~np.isfinite(vsl_speed_dict[k])] = np.nan

        vxl_speed_map = np.full((self.pos_ind.size, num_frame), np.nan)
        for k, v in vsl_speed_dict.items():
            i = self.ind_to_vxl_idx(k)
            vxl_speed_map[i] = v

        if update_fg_Q:
            self.vxl_speed_map = vxl_speed_map
            self.edge_passenger = e_passing_cell

        if return_type == 'dict':
            return vsl_speed_dict, e_passing_cell
        
        elif return_type == 'array':
            return vxl_speed_map, e_passing_cell
    
    def get_edge_vxl_speed_map(self, edge_label):
        """
        Inputs: 
            skl_speed_map: (N, T) np.array, where N is the number of 
            skeleton voxels and T is the number of time points. 
            edge_label: scalar integer
        """
        assert edge_label >= 0, "edge_label must be a non-negative integer"
        edge_ind = self.edge.cc_ind[edge_label]
        edge_skl_idx = self.ind_to_vxl_idx(edge_ind)
        return np.atleast_2d(self.vxl_speed_map[edge_skl_idx])

    def get_edgeflow_object(self, edge_label, detection_in_edge=None, vxl_speed_map=None, cell_passenger=None, num_frame=None): 
        """
        
            vsl_speed_map: (num_e_cc_vxl, T)
        """
        assert edge_label >= 0, ValueError("Invalid edge label")
        
        if (vxl_speed_map is None) and hasattr(self, 'vxl_speed_map'):
            vxl_speed_map = self.get_edge_vxl_speed_map(edge_label)
        
        if (cell_passenger is None) and hasattr(self, 'edge_passenger'):
            cell_passenger = self.edge_passenger[edge_label] # list of tuples (time, p_idx, did)
        
        if num_frame is None: 
            num_frame = vxl_speed_map.shape[1] if (vxl_speed_map is not None) else None

        ef = EdgeFlow(edge_label, self.edge.cc_ind[edge_label], self.num['mask_size'], 
                      self.edge.connected_node_label[edge_label], self.abs_g.edge_cc_pos[edge_label], 
                      self.abs_g.length[edge_label], detections=detection_in_edge, duration=num_frame,
                      vxl_speed_map=vxl_speed_map, cell_passenger=cell_passenger)
        return ef

    def analyze_edges_detection_map(self, detections, init_bin_size=2, init_dt=1, min_peak_corr=0.05, 
                                      min_num_div=3, max_num_div=15, re_est_min_tot_cor_r=3, max_dt=16, high_freq_ccf=0.0368):
        """
        Inputs: 
            detections: pandas table
            
        """
        edge_detection_dict = util.get_table_value_to_idx_dict(detections, 'edge_label', filter=lambda x: x>=0)
        dm_log = defaultdict(lambda : np.full(self.edge.num_cc, np.nan))
        for el, idx in edge_detection_dict.items():
            try: 
                tmp_edge_table = detections.iloc[idx]
                test_ef = self.get_edgeflow_object(el, detection_in_edge=tmp_edge_table)

                dm_info = test_ef.iterative_est_avg_velocity_from_detection_map(test_ef.detect_map, init_bin_size=init_bin_size, init_dt=init_dt, 
                                                                                min_peak_corr=min_peak_corr, min_num_div=min_num_div, 
                                                                                max_num_div=max_num_div, re_est_min_tot_cor_r=re_est_min_tot_cor_r,
                                                                                max_dt=max_dt, vis_Q=False)
                of_info = test_ef.analyze_detection_map(high_freq_ccf)
                dm_log['num_hc_vxl'][el] = of_info['num_hc_vxl']
                dm_log['num_detection'][el] = of_info['total_detection']
                if dm_info['validQ']: 
                    for k, v in dm_info.items():
                        dm_log[k][el] = dm_info[k]
            except Exception as e: 
                print(f"{el}, {idx}")
                raise e
            if (el % 100) == 0: 
                print(f"\rFinish processing edge {el}. ", end='', flush=True)
        print(f"\nFinish analyzing edge detection map.")
        return dm_log

    @staticmethod
    def show_edge_dm_properties(dm_log, edge_label, all_propQ=False): 
        e_stat = {}
        if all_propQ: 
            dm_copy_keys = dm_log.keys()
        else: 
            dm_copy_keys = ['num_hc_vxl', 'avg_corr', 'total_corr_ratio', 'avg_v', 'major_diff_cv', 'total_corr_ratio']
        for k in dm_copy_keys:
            e_stat[k] = dm_log[k][edge_label]
        return e_stat

    def vis_edges_with_mips(self, edge_labels, mips, edge_weight=None, figsize=(10, 10), legend_type='idx', show_legend_Q=True):
        edge_labels = np.atleast_1d(np.asarray(edge_labels)) 
        assert np.all(edge_labels >= 0), ValueError("Invalid edge label(s)")
        # edge_ind = np.concatenate(self.edge.cc_ind[edge_labels])
        # edge_sub = self.ind2sub(edge_ind)
        if legend_type == 'idx': 
            edge_sub = [self.ind2sub(self.edge.cc_ind[el]) for el in edge_labels]
        elif legend_type == 'label':
            edge_sub = {el: self.ind2sub(self.edge.cc_ind[el]) for el in edge_labels}
            if edge_weight is not None: 
                edge_weight = {e: w for e, w in zip(edge_labels, edge_weight)} 
        if edge_weight is None: 
            f = vis.vis_mips_w_ptl(mips, ptl_zyx=edge_sub, figsize=figsize, show_legend_Q=show_legend_Q)
        else: 
            f = vis.vis_mips_w_ptl_n_weight(mips, ptl_zyx=edge_sub, plt_weight=edge_weight, figsize=figsize, show_legend_Q=show_legend_Q)
        return f
    
    def vis_edge_groups_with_mips(self, edge_group:dict, mips, figsize=(10, 10), legend_type='idx', show_legend_Q=True):
        edge_sub = {}
        for group_label, group_el in edge_group.items():
            group_el = [group_el] if isinstance(group_el, int) else group_el
            tmp_sub = np.hstack([np.vstack(self.ind2sub(self.edge.cc_ind[el])) for el in group_el])
            edge_sub[group_label] = tmp_sub
        f = vis.vis_mips_w_ptl(mips, ptl_zyx=edge_sub, figsize=figsize, show_legend_Q=show_legend_Q, vis_type='dot')
        return f

    # def vis_interactive_edge_pos(self, width=1200, height=1000):
    #     vsl_el_array = np.full(self.num['mask_size'], 0, dtype=np.uint16)
    #     vsl_el_array.flat[self.edge.pos_ind] = self.edge.label + 1
    #     vsl_el_mip = np.max(vsl_el_array, axis=0)
    #     f = px.imshow(vsl_el_mip)
    #     f.update_layout(width=width, height=height)
    #     f.show()


#endregion

#region 2 edge detection map
    def get_node_connected_edge_flow_dir(self, node_label, edge_v, output_type='array'): 
        """
            1: out-flow from the node
            -1: in-flow into the node
            0: unknown
        """
        assert node_label >= 0, ValueError("Invalid node label")
        connected_edge = self.node.connected_edge_label[node_label]
        edge_dir = np.zeros(connected_edge.shape, np.int8)
        for i, e in enumerate(connected_edge):
            tmp_ev = edge_v[e]
            tmp_connected_nodes = self.edge.connected_node_label[e]
            if tmp_connected_nodes[0] == node_label: 
                tmp_edge_dir = 1
            elif tmp_connected_nodes[1] == node_label: 
                tmp_edge_dir = -1
            else: 
                raise ValueError(f"Node {node_label} is not connected with edge {e}")
            
            if tmp_edge_dir * tmp_ev > 0: 
                edge_dir[i] = 1
            elif tmp_edge_dir * tmp_ev < 0: 
                edge_dir[i] = -1
            else: 
                edge_dir[i] = 0
        if output_type == 'array': 
            return connected_edge, edge_dir
        elif output_type == 'dict':
            return {e : d for e, d in zip(connected_edge, edge_dir)}
    

    def estimate_v_using_multi_edge_detection_map(self, edge_label, trace_result, e_to_idx, edge_v_pxl=None,
                                                   min_peak_corr=0.05, min_tot_corr_r=2, min_major_corr=0.25, min_v2l=0.75, 
                                                   min_diag_corr_ratio=0.4, vis_Q=False): 
        if edge_v_pxl is None: 
            edge_v_pxl = self.edge_v_pxl

        est_result = []
        num_frame = trace_result.frame.max() + 1
        ef = self.get_edgeflow_object(edge_label, detection_in_edge=trace_result.iloc[e_to_idx[edge_label]], num_frame=num_frame)
        nb_node_label = self.edge.connected_node_label[edge_label]
        for node_label in nb_node_label: 
            if node_label >= 0: 
                # Make the connected node follow the voxels in the edge cc 
                if ef.connected_node_label[1] == node_label: 
                    ef_skl_dir = 1 
                elif ef.connected_node_label[0] == node_label: 
                    ef_skl_dir = -1
                else: 
                    ValueError(f"Node {node_label} is not connected to edge {edge_label}")             
                ef_1_dm = ef.detect_map[:, ::ef_skl_dir] if ef_skl_dir == -1 else ef.detect_map # (T, l)

                nb_e_list, nb_e_dir_list = self.get_node_connected_edge_flow_dir(node_label, edge_v_pxl, output_type='array')
                # exclude self and all the in-flowing edge
                all_outflow_Q = np.all(nb_e_dir_list[nb_e_list != edge_label] > 0)
                tmp_included_Q = np.logical_and((nb_e_list != edge_label), nb_e_dir_list >= 0) 
                nb_e_dir_list = nb_e_dir_list[tmp_included_Q]
                nb_e_list = nb_e_list[tmp_included_Q]

                if nb_e_list.size > 0 and all_outflow_Q: 
                    # All neighboring edges are out-flowing edges
                    # select the edge with the most event? Or, is the longest one better? 
                    # Selecting the edge with the most event assume no stationary points in the edge ... 
                    logging.debug(f"All connected edges are out-flowing. Select the one with the most detection events")
                    nb_num_detection = np.asarray([e_to_idx[e].size for e in nb_e_list])
                    # nb_len = np.asarray([self.abs_g.length[e] for e in nb_e_list])
                    tmp_idx = np.argmax(nb_num_detection)
                    nb_e_list = [nb_e_list[tmp_idx]]
                    nb_e_dir_list = [nb_e_dir_list[tmp_idx]]
                else: 
                    logging.debug(f"Some of the connected edges have unknown direction")
                
                for nb_e, nb_e_dir in zip(nb_e_list, nb_e_dir_list):
                    logging.debug(f"Neighboring edge {nb_e} direction {nb_e_dir}")
                    if nb_e in e_to_idx: # not self, not in-flowing edge
                        # flow in this edge is either unknown or out-flowing 
                        nb_ef = self.get_edgeflow_object(nb_e, detection_in_edge=trace_result.iloc[e_to_idx[nb_e]], num_frame=num_frame)

                        if nb_ef.connected_node_label[0] == node_label: 
                            nb_ef_skl_dir = 1 
                        elif nb_ef.connected_node_label[1] == node_label: 
                            nb_ef_skl_dir = -1
                        else: 
                            ValueError(f"Node {node_label} is not connected to edge {nb_e}") 
                        ef_2_dm = nb_ef.detect_map[:, ::nb_ef_skl_dir] if nb_ef_skl_dir == -1 else nb_ef.detect_map 

                        dm_info = EdgeFlow.analyze_concatenated_detection_map(ef_1_dm, ef_2_dm, min_corr=min_peak_corr,
                                                                                min_major_corr=min_major_corr, min_tot_corr_r=min_tot_corr_r, 
                                                                                vis_Q=vis_Q)
                        

                        if not np.isfinite(dm_info['avg_v']):
                            logging.debug(f"The estiamted velocity is invalid.")
                        elif (dm_info['diag_corr_ratio'] < min_diag_corr_ratio) and not all_outflow_Q: 
                            # If we know all other edges are out-flowing, we know the 
                            # edge of interest must be in-flowing
                            logging.debug(f"Offset diagonal sum in the upper right correlation matrix is too small comapred to in the whole matrix. ")
                        elif dm_info['avg_v_to_max_v_1'] > min_v2l: 
                            dm_info['edge_label'] = edge_label
                            dm_info['nb_edge'] = nb_e
                            dm_info['nb_edge_dir'] = nb_e_dir
                            dm_info['nb_node'] = node_label
                            dm_info['ef_skl_dir'] = ef_skl_dir
                            dm_info['all_nb_outflow_Q'] = all_outflow_Q
                            assert dm_info['avg_v'] > 0
                            # correct the flow direction 
                            dm_info['avg_v'] = dm_info['avg_v'] * ef_skl_dir
                            est_result.append(dm_info)
                        else: 
                            logging.debug(f"avg_v_to_max_v_1 {dm_info['avg_v_to_max_v_1']:.2f} is lower than expected. Discard")
            else: 
                logging.debug(f"Endpoint. Skip")
        
        return est_result
    
    @staticmethod
    def select_multi_edge_detection_map_results(results:list):
        """ Select one result out of the output of estimate_v_using_multi_edge_detection_map
        
        """
        if len(results) == 1: 
            results = results[0]
        elif len(results) > 1: 
            is_all_outflow_idx = np.nonzero(np.asarray([d['all_nb_outflow_Q'] for d in results]))[0] 
            if is_all_outflow_idx.size > 0: 
                assert is_all_outflow_idx.size == 1, ValueError("is_all_outflow_idx.size should be either 0 or 1")
                results = results[int(is_all_outflow_idx)]
            else: 
                tmp_diag_corr_ratio = np.asarray([d['diag_corr_ratio'] for d in results])
                tmp_major_total_corr = np.asarray([d['major_total_corr'] for d in results])
                tmp_cr_idx = np.argmax(tmp_diag_corr_ratio)
                tmp_mtc_idx = np.argmax(tmp_major_total_corr)
                if tmp_cr_idx == tmp_mtc_idx: 
                    results = results[tmp_cr_idx]
                else: 
                    tmp_d1 = results[tmp_cr_idx]
                    tmp_d2 = results[tmp_mtc_idx]
                    tmp_v_diff = np.abs(tmp_d1['avg_v'] - tmp_d2['avg_v'])
                    tmp_v_std = np.sqrt((tmp_d1['avg_v_std'] ** 2 + tmp_d2['avg_v_std'] ** 2) / 2)
                    if tmp_v_diff / tmp_v_std < 1: 
                        results = results[tmp_mtc_idx] # does not really matter which one 
                    else: 
                        results = None
                    # raise NotImplementedError
        else: 
            results = None

        return results
    

#endregion

#region flow configuration and inference
    def compute_node_flow_configuration(self, known_edge_v=None):
        if known_edge_v is None: 
            known_edge_v = self.edge_v_pxl
        result = {'n_num_in': np.zeros(self.node.num_cc, np.uint8), 
                  'n_num_out': np.zeros(self.node.num_cc, np.uint8), 
                  'n_num_unknown': np.zeros(self.node.num_cc, np.uint8)}
        
        for nl in range(self.node.num_cc):
            nb_e, nb_e_dir = self.get_node_connected_edge_flow_dir(nl, known_edge_v, output_type='array')
            nb_e_out = nb_e[nb_e_dir == 1]
            nb_e_in = nb_e[nb_e_dir == -1]
            nb_e_uk = nb_e[nb_e_dir == 0]
            result['n_num_out'][nl] = nb_e_out.size
            result['n_num_in'][nl] = nb_e_in.size
            result['n_num_unknown'][nl] = nb_e_uk.size        

        return result

    def infer_edge_velocity_by_node_flow_configuration(self, known_edge_v=None, know_edge_v_std=None, verbose_Q=False):
        if known_edge_v is None: 
            known_edge_v = self.edge_v_pxl
        if know_edge_v_std is None: 
            know_edge_v_std = self.edge_v_std_pxl
        result = {'v': known_edge_v.copy(), 
                  'std': know_edge_v_std.copy(), 
                  'n_num_in': np.zeros(self.node.num_cc, np.uint8), 
                  'n_num_out': np.zeros(self.node.num_cc, np.uint8), 
                  'n_num_unknown': np.zeros(self.node.num_cc, np.uint8), 
                  'inferred_edge': []}
        for nl in range(self.node.num_cc):
            nb_e, nb_e_dir = self.get_node_connected_edge_flow_dir(nl, known_edge_v, output_type='array')
            nb_e_out = nb_e[nb_e_dir == 1]
            nb_e_in = nb_e[nb_e_dir == -1]
            nb_e_uk = nb_e[nb_e_dir == 0]
            result['n_num_out'][nl] = nb_e_out.size
            result['n_num_in'][nl] = nb_e_in.size
            result['n_num_unknown'][nl] = nb_e_uk.size
            if nb_e_in.size == 0 and nb_e_uk.size == 1: 
                # uk should be in 
                nb_e_uk = int(nb_e_uk)
                nb_v = np.abs(known_edge_v[nb_e_out])
                max_idx = np.argmax(nb_v)
                uk_v = nb_v[max_idx] # be conservative? what else can we do - especially when the radius is not accurate
                # inflow w.r.t. the node -> positive direction means the node is the second node the edge connected to 
                uk_v = uk_v if (self.edge.connected_node_label[nb_e_uk][1] == nl) else -uk_v 
                
                result['v'][nb_e_uk] = uk_v
                result['std'][nb_e_uk] = know_edge_v_std[nb_e_out[max_idx]]
                result['n_num_in'][nl] = 1
                result['n_num_unknown'][nl] = 0
                result['inferred_edge'].append(nb_e_uk)
                if verbose_Q: 
                    print(f"Node {nl} has outflow edge {nb_e_out} and no known inflow edge. Infer edge {nb_e_uk} to have velocity {uk_v}")
            elif nb_e_out.size == 0 and nb_e_uk.size == 1: 
                # uk should be out
                nb_e_uk = int(nb_e_uk)
                tmp_nb_e_in_v = np.abs(known_edge_v[nb_e_in])
                max_idx = np.argmax(tmp_nb_e_in_v)
                uk_v = tmp_nb_e_in_v[max_idx]
                uk_v = uk_v if (self.edge.connected_node_label[nb_e_uk][0] == nl) else -uk_v 
                
                result['v'][nb_e_uk] = uk_v
                result['std'][nb_e_uk] = know_edge_v_std[nb_e_in[max_idx]]
                result['n_num_out'][nl] = 1
                result['n_num_unknown'][nl] = 0
                result['inferred_edge'].append(nb_e_uk)
                if verbose_Q: 
                    print(f"Node {nl} has inflow edge {nb_e_in} and no known outflow edge. Infer edge {nb_e_uk} to have velocity {uk_v}")    
        if verbose_Q: 
            print(f"Inferred flow velocity in {len(result['inferred_edge'])} edges")
        return result

    def get_nearest_downstream_edge_labels(self, edge_label, know_edge_v=None): 
        edge_label = int(edge_label)
        if know_edge_v is None: 
            know_edge_v = self.edge_v_pxl
        nb_e = self.get_nearest_neighbor_edge_labels_of_an_edge(edge_label)
        nb_e = nb_e[0] if know_edge_v[edge_label] < 0 else nb_e[1]

        nb_n = self.edge.connected_node_label[edge_label]
        nb_n = nb_n[0] if know_edge_v[edge_label] < 0 else nb_n[1]
        ds_e = []
        if nb_n >= 0: 
            for tmp_e in nb_e: 
                if know_edge_v[tmp_e] > 0: 
                    tmp_e_v_dir = 1
                elif know_edge_v[tmp_e] < 0: 
                    tmp_e_v_dir = -1
                else: 
                    tmp_e_v_dir = 0
                tmp_e_n = self.edge.connected_node_label[tmp_e]
                if tmp_e_n[0] == nb_n: 
                    tmp_e_dir = 1
                elif tmp_e_n[1] == nb_n: 
                    tmp_e_dir = -1
                else: 
                    tmp_e_dir = 0
                
                if tmp_e_dir * tmp_e_v_dir > 0: 
                    ds_e.append(tmp_e)
        return ds_e

    def get_nearest_upstream_edge_labels(self, edge_label, know_edge_v=None): 
        edge_label = int(edge_label)
        if know_edge_v is None: 
            know_edge_v = self.edge_v_pxl
        nb_e = self.get_nearest_neighbor_edge_labels_of_an_edge(edge_label)
        nb_n = self.edge.connected_node_label[edge_label]
        nb_e = nb_e[0] if know_edge_v[edge_label] > 0 else nb_e[1]
        nb_n = nb_n[0] if know_edge_v[edge_label] > 0 else nb_n[1]
        
        us_e = []
        if nb_n >= 0: 
            for tmp_e in nb_e: 
                tmp_e_v_dir = np.sign(know_edge_v[tmp_e])

                tmp_e_n = self.edge.connected_node_label[tmp_e]
                if tmp_e_n[0] == nb_n: 
                    tmp_e_dir = 1
                elif tmp_e_n[1] == nb_n: 
                    tmp_e_dir = -1
                else: 
                    tmp_e_dir = 0
                
                if tmp_e_dir * tmp_e_v_dir < 0: 
                    us_e.append(tmp_e)
        return us_e    


    def get_downstream_edges_in_tree(self, edge_label, know_edge_v=None, current_order=1, cutoff_order=np.inf): 
        
        edge_label = int(edge_label)
        if know_edge_v is None: 
            know_edge_v = self.edge_v_pxl
        tree = OrderedDict()
        nearest_ds_edge = self.get_nearest_downstream_edge_labels(edge_label, know_edge_v)
        if len(nearest_ds_edge) > 0: 
            # tree[edge_label] = (nearest_ds_edge, current_order)
            tree[edge_label] = nearest_ds_edge
            if current_order < cutoff_order: 
                for next_edge in nearest_ds_edge: 
                    child_tree = self.get_downstream_edges_in_tree(next_edge, know_edge_v, 
                                                                   current_order=current_order+1, cutoff_order=cutoff_order)
                    tree.update(child_tree)
        return tree
    
    def get_downstream_edges(self, edge_label, know_edge_v=None, cutoff_order=1, include_self_Q=False):
        tmp_el_ds_e_t = self.get_downstream_edges_in_tree(edge_label, know_edge_v=know_edge_v, cutoff_order=cutoff_order)
        tmp_el_ds_t = [edge_label] if include_self_Q else []
        for k, v in tmp_el_ds_e_t.items():
            tmp_el_ds_t += ([k] + v)
        tmp_el_ds_t = np.unique(tmp_el_ds_t)
        if not include_self_Q: 
            tmp_el_ds_t = tmp_el_ds_t[tmp_el_ds_t != edge_label]

        return tmp_el_ds_t

#endregion


class NearestSkeletonVoxel(nb.NearestMaskVoxel):
    def __init__(self, vsl_skl_labeled, vsl_vol_labeled, save_data_Q=False):
        """ Class for mapping voxel indices to the nearest vessel skeleton indices

        Inputs: 
            vsl_skl_labeled: 3D labeled vessel skeleton array. 1 for capillary, 2 for artery, 3 for vein 
            vsl_mask_label: 3D labeled vessel volume array (from which the skeleton was derived). Same 
                label meaning as vsl_skl_labeled. 
            save_data_Q: logical scalar, cache vsl_skl_labeled if true.         

        This class mainly solves the problem of assigning cells to skeleton voxels when artiers and 
        veins get very close to each other. In that case, the pure distance-transform approach might 
        incorrectly assing part of a large vessel to a nearby small vessel due to the absolute distance 
        difference. This algorithm assume arteries / veins themselves are sparse such that the distance-
        based assignment is valid when the other type is abscent in the mask. 

        This method appear to be computational expensive to initialize but efficient for multiple random 
        look-up later, as needed for the graph distance computation. Not sure if there's a better approach. 

        Class methods are implemented in nb.NearestMaskVoxel. 

        """
        assert np.issubdtype(vsl_skl_labeled.dtype, np.integer), 'vsl_skl_labeled should be an integer array'
        assert np.issubdtype(vsl_vol_labeled.dtype, np.integer), 'vsl_mask_label should be an integer array'

        mask_size = vsl_skl_labeled.shape

        # Make sure the labels are consistent 
        inconsistent_artery_Q = np.logical_and(vsl_skl_labeled == 2, vsl_vol_labeled == 1)
        if np.any(inconsistent_artery_Q): 
            print(f"{np.count_nonzero(inconsistent_artery_Q)} skeleton voxels are labeled as artery in vsl_skl_labeled, but as capillary in vsl_vol_labeled. Correct to capillary.")
            vsl_skl_labeled[inconsistent_artery_Q] = 1
        inconsistent_vein_Q = np.logical_and(vsl_skl_labeled == 3, vsl_vol_labeled == 1)
        if np.any(inconsistent_vein_Q): 
            print(f"{np.count_nonzero(inconsistent_vein_Q)} skeleton voxels are labeled as vein in vsl_skl_labeled, but as capillary in vsl_vol_labeled. Correct to capillary.")
            vsl_skl_labeled[inconsistent_vein_Q] = 1
        inconsistent_large_vessel_Q = np.logical_or(np.logical_and(vsl_skl_labeled == 2, vsl_vol_labeled == 3), 
                                                  np.logical_and(vsl_skl_labeled == 3, vsl_vol_labeled == 2))
        if np.any(inconsistent_large_vessel_Q): 
            raise ValueError("Inconsistent skeleton voxel artery/vein label")

        # Compute the nearest skeleton ind for all voxel in the (artery + capillary) mask 
        ac_skl_mask = np.logical_and(vsl_skl_labeled > 0, vsl_skl_labeled != 3)
        ac_mask = np.logical_and(vsl_vol_labeled > 0, vsl_vol_labeled != 3)
        ac_skl_dt, ac_skl_ind = nb.ndi.distance_transform_edt(1 - ac_skl_mask, return_indices=True)
        ac_skl_ind = np.ravel_multi_index([ac_skl_ind[i] for i in range(3)], mask_size)
        combined_dt_ind = np.full(mask_size, -1, dtype=ac_skl_ind.dtype)
        combined_dt_ind[ac_mask] = ac_skl_ind[ac_mask]
        del ac_skl_dt, ac_skl_ind

        # Compute the nearest skeleton ind for all voxel in the (vein + capillary) mask 
        vc_skl_mask = np.logical_and(vsl_skl_labeled > 0, vsl_skl_labeled != 2)
        vc_mask = np.logical_and(vsl_vol_labeled > 0, vsl_vol_labeled != 2)
        vc_skl_dt, vc_skl_ind = nb.ndi.distance_transform_edt(1 - vc_skl_mask, return_indices=True)
        vc_skl_ind = np.ravel_multi_index([vc_skl_ind[i] for i in range(3)], mask_size)
        combined_dt_ind[vc_mask] = vc_skl_ind[vc_mask]
        del vc_skl_dt, vc_skl_ind

        # For the background voxels, compute the nearest vessel mask voxel 
        # Use this mask voxel indice to find its nearest skeleton voxel found above
        bg_mask = 1 - (vsl_vol_labeled > 0)
        all_mask_dt, all_mask_ind = nb.ndi.distance_transform_edt(bg_mask, return_indices=True)
        all_mask_ind = np.ravel_multi_index([all_mask_ind[i] for i in range(3)], mask_size)

        bg_ind = np.flatnonzero(bg_mask)
        bg_nearest_skl_ind = combined_dt_ind.flat[all_mask_ind.flat[bg_ind]] # bg_ind -> nearest mask ind -> nearest skeleton ind
        combined_dt_ind.flat[bg_ind] = bg_nearest_skl_ind
        del all_mask_dt, all_mask_ind, bg_mask

        # Compute the distance from every point to its assigned "nearest" skeleton voxel directly
        tmp_sub = np.unravel_index(np.arange(np.prod(combined_dt_ind.shape)), combined_dt_ind.shape)
        tmp_nearest_sub = np.unravel_index(combined_dt_ind.flatten(), combined_dt_ind.shape)
        combined_dt = np.vstack([(tmp_sub[i] - tmp_nearest_sub[i]) ** 2 for i in range(len(tmp_sub))]) 
        del tmp_sub, tmp_nearest_sub

        combined_dt = np.sqrt(np.sum(combined_dt, axis=0))
        combined_dt = combined_dt.reshape(combined_dt_ind.shape).astype(np.float32)

        self.dt = combined_dt
        self.nearest_pos = np.unravel_index(combined_dt_ind, mask_size)
        self.mask_size = mask_size
        if save_data_Q: 
            self.data = vsl_skl_labeled.copy()
