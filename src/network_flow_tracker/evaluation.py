import numpy as np
import pandas as pd

import network_flow_tracker.FlowGraph as FG
from .utils import util

class TraceGenerator():
    def __init__(self, fg: FG.FlowGraph, inflow_el, in_flow_num_entr_per_frame, 
                 int_mean=1e4, int_std=2e3):
        self.fg = fg
        self.inflow_el = inflow_el
        self.para = {'int_mean': int_mean, 'int_std': int_std}

        self.entr_rate = in_flow_num_entr_per_frame
        self.entr_range_pxl = np.round(-fg.edge_v_pxl[inflow_el]).astype(np.int64)
    
    def sample_entrance_skl_ind(self, ind2nump={}):
        s_num_p = np.random.poisson(self.entr_rate)
        tmp_has_new_p_el = np.where(s_num_p > 0)[0]
        s_start_ind = []
        for i, tmp_idx in enumerate(tmp_has_new_p_el): 
            tmp_n = s_num_p[tmp_idx]
            tmp_v = self.entr_range_pxl[tmp_idx]
            tmp_sample_ind = self.fg.edge.cc_ind[self.inflow_el[tmp_idx]][-tmp_v:]
            if len(ind2nump) > 0:
                tmp_sample_ind = np.array([ind for ind in tmp_sample_ind if ind2nump.get(ind, 0) < 2])
            if tmp_n < tmp_sample_ind.size: 
                tmp_sample_ind = np.random.choice(tmp_sample_ind, size=tmp_n, replace=False)

            s_start_ind.append(tmp_sample_ind)
        s_start_ind = np.concatenate(s_start_ind)
        return s_start_ind

    def entr_skl_ind_to_pos(self, skl_ind, return_type='xyz'): 
        assert return_type in ['sub', 'xyz'], "return_type must be 'sub' or 'xyz'"
        skl_sub = np.column_stack(self.fg.edge.ind2sub(skl_ind))
        # p_xyz_0 = np.random.normal(p_sub_0[:, ::-1], scale=pos_std)
        # p_xyz_0 = np.clip(p_xyz_0, [0, 0, 0], np.array(mask_size)[::-1] - 1)
        # p_sub_0 = np.round(p_xyz_0[:, ::-1]).astype(np.int64)
        if return_type == 'sub':
            return skl_sub
        elif return_type == 'xyz':
            skl_xyz = skl_sub[:, ::-1] #add noise later 
            return skl_xyz

    def sample_entrance_particle_int(self, num_p, int_mean=1e4, int_std=2e3):
        p_int = np.random.normal(int_mean, scale=int_std, size=num_p)
        p_int = np.clip(p_int, a_min=1, a_max=None).astype(np.int64)
        return p_int
    
    def generate_new_particles(self, num_particle, int_mean=1e4, int_std=2e3, ind2nump={}):
        p_ind_0 = self.sample_entrance_skl_ind(ind2nump=ind2nump)
        p_sub_0 = self.entr_skl_ind_to_pos(p_ind_0, return_type='sub')
        p_int_0 = self.sample_entrance_particle_int(p_ind_0.size, int_mean, int_std)
        p_id = np.arange(num_particle, num_particle + p_ind_0.size, dtype=np.int64)
        p_exit_Q = np.zeros(p_ind_0.size, np.bool_)
        # need to distinguish real exit and artificial exit due to unknown speed
        s_data = {'peak_int': p_int_0,
                'sub': p_sub_0, 'p_id': p_id, 
                'exit_Q': p_exit_Q, 'exit_network_Q': p_exit_Q.copy()} 
        return s_data
    
    def _compute_edge_and_voxel_occupation_fraction(self, curr_det):
        curr_det_ind = self.fg.edge.sub2ind(curr_det['sub'].T)
        vxl_p_count = {}
        for ind in curr_det_ind:
            if ind in vxl_p_count: 
                vxl_p_count[ind] += 1
            else: 
                vxl_p_count[ind] = 1
        unique_det_ind = np.array(list(vxl_p_count.keys()))
        vxl_el = self.fg.edge.ind_to_label(unique_det_ind)
        tmp_vxl_idx, tmp_curr_el = util.bin_data_to_idx_list(vxl_el)
        curr_f_p = {}
        for tmp_idx, tmp_el in zip(tmp_vxl_idx, tmp_curr_el):
            tmp_sum = np.sum([vxl_p_count[ind] for ind in unique_det_ind[tmp_idx]])
            curr_f_p[tmp_el] = tmp_sum / self.fg.e_length[tmp_el]

        return curr_f_p, vxl_p_count

    def _compute_edge_occupation_fraction(self, curr_det):
        curr_det_ind = self.fg.edge.sub2ind(curr_det['sub'].T)
        curr_ind_el = self.fg.edge.ind_to_label(curr_det_ind)
        tmp_e_idx, el_u = util.bin_data_to_idx_list(curr_ind_el)
        curr_f_p = {k: v.size / self.fg.e_length[k] for k, v in zip(el_u, tmp_e_idx)}
        return curr_f_p

    @staticmethod
    def get_active_particles(prev_det):
        if prev_det is not None: 
            selected_Q = (~prev_det['exit_Q'].copy())
            curr_det = {k:v[selected_Q] for k, v in prev_det.items()}
        else: 
            curr_det = None
        return curr_det

    def move_existing_active_particles(self, curr_det, noise_cv=0.1, high_occ_th=1.0, phase_seperation_n=1.0): 
        if curr_det is not None: 
            curr_f_p = self._compute_edge_occupation_fraction(curr_det)
            for i, p_sub in enumerate(curr_det['sub']): 
                s_pos_tp1 = self.fg.predict_single_particle_position(p_sub, dt=1, noise_cv=noise_cv)
                if len(s_pos_tp1) > 1: 
                    tmp_pos_el = np.array([tmp['edge_label'] for tmp in s_pos_tp1])
                    tmp_f_p = np.array([curr_f_p.get(el, 0) for el in tmp_pos_el])
                    tmp_v_pxl = np.abs(np.array([tmp['v_pxl'] for tmp in s_pos_tp1]))                   
                    tmp_v_sum = np.nansum(tmp_v_pxl)
                    if tmp_v_sum > 0:
                        tmp_p = tmp_v_pxl / tmp_v_sum
                        tmp_p[np.isnan(tmp_p)] = 0
                        # if high occupation, do not enter
                        tmp_non_zero_idx = np.nonzero(tmp_p)[0]
                        if tmp_non_zero_idx.size > 1: 
                            is_high_occ_Q = tmp_f_p[tmp_non_zero_idx] > high_occ_th
                            tmp_p[tmp_non_zero_idx[is_high_occ_Q]] = 0
                        if np.nansum(tmp_p) > 0:
                            tmp_p = tmp_p ** phase_seperation_n / np.nansum(tmp_p ** phase_seperation_n)
                        else: 
                            print(f"all possible next edges are highly occupied. Randomly choose one")
                            tmp_p = np.ones_like(tmp_v_pxl) / len(tmp_v_pxl)
                    else: 
                        tmp_p = np.ones_like(tmp_v_pxl) / len(tmp_v_pxl)
                    if np.isnan(tmp_p).any(): 
                        print(i, tmp_p)
                        raise ValueError("nan in tmp_p")

                    tmp_idx = int(np.random.choice(len(s_pos_tp1), size=1, p=tmp_p))
                else: 
                    tmp_idx = 0
                s_pos_tp1 = s_pos_tp1[tmp_idx]
                if np.isnan(s_pos_tp1['sub']).any():
                    assert s_pos_tp1['exit_network_Q'], "predicted sub is nan but not exit network"
                    curr_det['sub'][i] = s_pos_tp1['exit_ep_sub']
                else: 
                    curr_det['sub'][i] = s_pos_tp1['sub']
                curr_det['exit_Q'][i] = s_pos_tp1['exit_network_Q']
                curr_det['exit_network_Q'][i] = s_pos_tp1['exit_network_Q']
                if isinstance(s_pos_tp1['sub'], tuple): 
                    # move into an edge with unknown speed. Treat as exit
                    assert np.isnan(s_pos_tp1['ind']), "predicted ind is not nan but sub is tuple"
                    curr_det['exit_Q'][i] = True 
                if s_pos_tp1['v_pxl'] == 0:
                    # move into an edge with zero speed. Treat as exit
                    curr_det['exit_Q'][i] = True
                # TODO: add noise to intensity 
        else: 
            curr_det = None
        
        return curr_det

    def generate_detection_sequence(self, num_frame, v_cv=0, high_occ_th=0.5, phase_seperation_n=1.0):

        num_particle = 0
        detection_list = []
        for t in range(num_frame):
            # evolve existing particles 
            prev_det = detection_list[-1] if len(detection_list) > 0 else None
            curr_det = TraceGenerator.get_active_particles(prev_det)
            curr_det = self.move_existing_active_particles(curr_det, noise_cv=v_cv, 
                                                           high_occ_th=high_occ_th, 
                                                           phase_seperation_n=phase_seperation_n)
            # inject new particles
            if curr_det is not None: 
                curr_ind = self.fg.edge.sub2ind(curr_det['sub'].T)
                tmp_count = {}
                for ind in curr_ind:
                    if ind in tmp_count: 
                        tmp_count[ind] += 1
                    else: 
                        tmp_count[ind] = 1
            else: 
                tmp_count = {}    
            s_data = self.generate_new_particles(num_particle, self.para['int_mean'],
                                                    self.para['int_std'], ind2nump=tmp_count)
            if curr_det is not None: 
                for k, v in curr_det.items(): 
                    s_data[k] = np.concatenate([v, s_data[k]])
            detection_list.append(s_data)
            num_particle += s_data['peak_int'].size
            print(f"Simulating frame {t + 1}/{num_frame}", end='\r')
        
        return detection_list
    
    @staticmethod
    def detection_list_to_dataframe(detection_list):
        detection_table = []
        num_detect_in_frame = []
        num_frame = len(detection_list)
        for i, tmp_data in enumerate(detection_list):
            tmp_table = pd.DataFrame({'peak_int': tmp_data['peak_int'],
                                    'x': tmp_data['sub'][:, 2],
                                    'y': tmp_data['sub'][:, 1],
                                    'z': tmp_data['sub'][:, 0],
                                    'pid': tmp_data['p_id'], 
                                    'exit_Q': tmp_data['exit_Q'], 
                                    'exit_network_Q': tmp_data['exit_network_Q']})
            num_detect_in_frame.append(tmp_table.shape[0])
            tmp_table['frame'] = i
            detection_table.append(tmp_table)

        detection_table = pd.concat(detection_table).reset_index(drop=True)

        pid_to_idx = util.bin_data_to_idx_list(detection_table.pid.values, return_type='dict')

        sim_exit_Q = detection_table.exit_Q.values.copy()
        sim_normal_exit_Q = detection_table.exit_network_Q.values.copy()
        # reset 
        detection_table['exit_Q'] = False
        detection_table['exit_network_Q'] = False
        for tmp_pid, tmp_idx in pid_to_idx.items():
            if tmp_idx.size > 1: 
                if sim_exit_Q[tmp_idx[-1]]: 
                    detection_table.loc[tmp_idx[-2], 'exit_Q'] = True
                if sim_normal_exit_Q[tmp_idx[-1]]: 
                    detection_table.loc[tmp_idx[-2], 'exit_network_Q'] = True  
            else: 
                assert detection_table.iloc[tmp_idx].frame.values == (num_frame - 1)
        # remove particles that exit the volume
        detection_table = detection_table.iloc[~sim_exit_Q].reset_index(drop=True)
        
        return detection_table
    
    @staticmethod
    def detection_table_to_link_idx(detection_table, init_id=-1, normal_exit_id=-2, abnormal_exit_id=-3): 
        
        if 'pid' in detection_table.columns:
            pid_val = detection_table.pid.values
        elif 'particle' in detection_table.columns:
            pid_val = detection_table.particle.values
        else:
            raise ValueError("detection table must have 'pid' or 'particle' column")

        p_to_ind = util.bin_data_to_idx_list(pid_val, return_type='dict')
        did = detection_table.did.values if 'did' in detection_table.columns else np.arange(detection_table.shape[0], dtype=np.int64)
        if 'exit_network_Q' not in detection_table.columns: 
            exit_network_Q = np.zeros(detection_table.shape[0], dtype=bool)
        else:
            exit_network_Q = detection_table.exit_network_Q.values
        
        if 'exit_Q' not in detection_table.columns:
            exit_Q = np.zeros(detection_table.shape[0], dtype=bool)
        else:
            exit_Q = detection_table.exit_Q.values
        

        link_target = np.full(detection_table.shape[0], init_id, dtype=np.int64)
        for tmp_g_idx, tmp_g_ind in p_to_ind.items():
            tmp_did = did[tmp_g_ind]
            tmp_exit_network_Q = exit_network_Q[tmp_g_ind]
            tmp_exit_Q = exit_Q[tmp_g_ind]
            if tmp_did.size > 1:
                for j, (didt, didtp1) in enumerate(zip(tmp_did[:-1], tmp_did[1:])):
                    link_target[didt] = didtp1
            if tmp_exit_network_Q[-1]: # normal exit
                link_target[tmp_did[-1]] = normal_exit_id
            elif tmp_exit_Q[-1]: # disappear - due to unknown edge speed
                link_target[tmp_did[-1]] = abnormal_exit_id
    
        return link_target
    
    @staticmethod
    def compare_link_targets(pred_link_target, gt_link_target, normal_exit_id=-2, abnormal_exit_id=-3, 
                             include_abnormal_exit_Q=True, pos_include_normal_exit_Q=False):
        # exclude the trailing -1 in gt_link_target
        valid_Q = (gt_link_target != -1)
        # assert (np.argmax(~valid_Q) / valid_Q.size) > 0.99, "too many -1 in gt_link_target"
        pred_link_target = pred_link_target[valid_Q]
        gt_link_target = gt_link_target[valid_Q]

        true_positive_Q = (pred_link_target >= 0) & (pred_link_target == gt_link_target)
        # gt has connection, but pred has wrong connection
        false_connection_Q = (pred_link_target >= 0) & (gt_link_target >= 0) & (pred_link_target != gt_link_target)
        # gt has no connection, but pred has connection
        false_positive_connection_Q = (pred_link_target >= 0) & (gt_link_target < 0)
        
        false_positive_Q = false_connection_Q | false_positive_connection_Q

        true_negative_Q = (pred_link_target < 0) & (gt_link_target < 0)
        false_negative_Q = (pred_link_target < 0) & (gt_link_target >= 0)

        true_normal_exit_Q = (pred_link_target == normal_exit_id) & (gt_link_target == normal_exit_id)
        false_normal_exit_Q = (pred_link_target == normal_exit_id) & (gt_link_target != normal_exit_id)

        true_abnormal_exit_Q = (pred_link_target == abnormal_exit_id) & (gt_link_target == abnormal_exit_id)
        false_abnormal_exit_Q = (pred_link_target == abnormal_exit_id) & (gt_link_target != abnormal_exit_id)

        if pos_include_normal_exit_Q:
            true_positive_Q = np.logical_or(true_positive_Q, true_normal_exit_Q)
            false_positive_Q = np.logical_or(false_positive_Q, false_normal_exit_Q)

            true_negative_Q = true_abnormal_exit_Q
            false_negative_Q = false_abnormal_exit_Q

        # check sum 
        covered_Q = (true_positive_Q | true_negative_Q | false_positive_Q | false_negative_Q)
        # print(np.nonzero(~covered_Q)[0])
        assert np.all(covered_Q), "some cases are not covered"

        result = {
                'true_positive_Q': true_positive_Q,
                'false_connection_Q': false_connection_Q,
                'false_positive_connection_Q': false_positive_connection_Q,
                'false_positive_Q': false_positive_Q,
                'true_negative_Q': true_negative_Q,
                'false_negative_Q': false_negative_Q,
                'true_normal_exit_Q': true_normal_exit_Q,
                'false_normal_exit_Q': false_normal_exit_Q,
                'true_abnormal_exit_Q': true_abnormal_exit_Q,
                'false_abnormal_exit_Q': false_abnormal_exit_Q}
        
        if not include_abnormal_exit_Q: 
            is_normal_Q = (gt_link_target != abnormal_exit_id)
            result = {k:v[is_normal_Q] for k, v in result.items()}
        # compute state
        stat = {k: np.mean(v) for k, v in result.items()}
        stat['num_links'] = len(pred_link_target)
        stat['precision'] = stat['true_positive_Q'] / (stat['true_positive_Q'] + stat['false_positive_Q'] + 1e-8)
        stat['recall'] = stat['true_positive_Q'] / (stat['true_positive_Q'] + stat['false_negative_Q'] + 1e-8)
        stat['accuracy'] = (stat['true_positive_Q'] + stat['true_negative_Q']) / (stat['true_positive_Q'] + stat['true_negative_Q'] + stat['false_positive_Q'] + stat['false_negative_Q'] + 1e-8)
        stat['f1'] = 2 * stat['precision'] * stat['recall'] / (stat['precision'] + stat['recall'] + 1e-8)
        result['stat'] = stat

        return result
    
    @staticmethod
    def analyze_tracking_result(trace_table, gt_link_target, edge_label_to_idx=None, edge_v_idx=None, 
                                init_default_id=-1, normal_exit_id=-2, abnormal_exit_id=-3):
        trace_table['exit_network_Q'] = (trace_table.exit_ind >= 0)
        trace_table['exit_Q'] = np.logical_or(np.isnan(trace_table.cost), trace_table['exit_network_Q'].values) 
        cv_link_target = TraceGenerator.detection_table_to_link_idx(trace_table, init_default_id, normal_exit_id, abnormal_exit_id)

        cv_stat = TraceGenerator.compare_link_targets(cv_link_target, gt_link_target, include_abnormal_exit_Q=True)
        if edge_label_to_idx is not None and edge_v_idx is not None:
            # Compute accuracy for each edge
            accurate_Q = cv_stat['true_positive_Q'] | (cv_stat['true_negative_Q'])
            num_edge = np.max(np.concatenate(edge_v_idx)) + 1
            edge_stat_key = ['precision', 'recall', 'accuracy']
            edge_stat = {k: np.full(num_edge, np.nan) for k in edge_stat_key}
            for el, el_idx in edge_label_to_idx.items():
                edge_stat['accuracy'][el] = np.mean(accurate_Q[el_idx])
                tmp_num_true_positive = np.sum(cv_stat['true_positive_Q'][el_idx])
                tmp_num_false_positive = np.sum(cv_stat['false_positive_Q'][el_idx])
                tmp_num_false_negative = np.sum(cv_stat['false_negative_Q'][el_idx])
                edge_stat['precision'][el] = tmp_num_true_positive / (tmp_num_true_positive + tmp_num_false_positive + 1e-8)
                edge_stat['recall'][el] = tmp_num_true_positive / (tmp_num_true_positive + tmp_num_false_negative + 1e-8)

            for k in edge_stat_key:
                edge_stat[f"{k}_ptrl"] = np.asarray(util.compute_stat_in_bin(edge_v_idx, edge_stat[k], fun=lambda x: np.percentile(x, [25, 50, 75]),
                                                    default_val=np.full(3, np.nan)))
                edge_stat[f"{k}_mean"] = np.asarray(util.compute_stat_in_bin(edge_v_idx, edge_stat[k], fun=lambda x: np.mean(x),
                                                    default_val=np.nan))
                

            cv_stat['branch'] = edge_stat
        return cv_stat, cv_link_target