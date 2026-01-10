import numpy as np
import pandas as pd


def mask_data_in_dict(data, mask, mask_priority=None):
    masked_data = {}
    assert isinstance(data, dict), "data should be a dictionary"
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            match_dim = np.nonzero(np.array(v.shape) == mask.size)[0]
            indexer = [slice(None)] * v.ndim
            if match_dim.size == 1:
                match_dim = int(match_dim)
            else:
                if mask_priority is None: 
                    raise "More than one dimension match"
                elif mask_priority == 'first':
                    match_dim = int(match_dim[0])
                else:
                    match_dim = int(match_dim[-1]) 
            indexer[match_dim] = mask
            v = v[tuple(indexer)]
        masked_data[k] = v
    return masked_data

def get_value_in_list_of_dict(data, key):
    result = []
    for d in data: 
        result.append(d[key])
    return result

def bin_data_to_idx_list(data, return_type='list'):
    data = np.asarray(data)
    sort_idx = np.argsort(data, kind='stable')
    unique_val, first_ind, counts = np.unique(data[sort_idx], return_index=True, return_counts=True)
    # must initialize the array first, and then fill 
    # If converting the list to numpy object array later and if 
    # each element in the list is a scalar, numpy will automatically 
    # convert each element into object type and form a 2d numpy object array 
    # there seems to be no way to avoid this at this point. 
    bin_idx_list = np.empty(first_ind.size, dtype=object) 
    for n, i0 in enumerate(first_ind):
        c = counts[n]
        tmp_idx = sort_idx[i0:(i0 + c)]
        bin_idx_list[n] = np.asarray(tmp_idx)
    if return_type == 'list': 
        return bin_idx_list, unique_val
    elif return_type == 'dict':
        return {k : v for k, v in zip(unique_val, bin_idx_list)}
    else: 
        raise ValueError(f"Unrecognized return_type. Options: 'list', 'dict'")

def find_ind_in_sub_array(sub_array, ind, mask_size, select='all'):
    sub = np.hstack(np.unravel_index(ind, mask_size))[:, None]
    num_sub = sub_array.shape[1]
    assert sub.shape == (3, 1), "ind should be a integer scalar"
    assert sub_array.shape[0] == 3, "sub_array should be a (3, n) numpy integer array"
    match_Q = np.all(sub_array == sub, axis=0)
    idx = np.nonzero(match_Q)[0]
    if select == 'all':
        return idx
    elif select == 'first':
        return idx[0]
    elif select == 'last':
        return idx[-1]
    elif select == 'shrink':
        if np.all(idx == np.array([0, 1])):
            return 1
        elif np.all(idx == np.array([num_sub-2, num_sub-1])):
            return num_sub-2
        else:
            return idx

def get_intervals_in_1d_binary_array(vec, prepend=0, append=0, including_end_Q=False):
    """
    
    Output: 
        intervals: (N, 2) np.ndarray
    """
    vec = np.asarray(vec)
    vec_diff = np.diff(vec, prepend=prepend, append=append)
    non_zeros = np.nonzero(vec_diff)[0]
    is_start_Q = (vec_diff[non_zeros] == 1)
    start_idx = non_zeros[is_start_Q]
    end_idx = non_zeros[~is_start_Q]
    assert np.all(start_idx < end_idx), f"{start_idx}, {end_idx}"
    if including_end_Q:
        end_idx -= 1
    intervals = np.stack((start_idx, end_idx), axis=1).astype(np.int64)
    return intervals

def get_continuous_interval_endpoint_idx_in_1d_integer_vector(vec, including_end_Q=True):

    vec = np.sort(np.asarray(vec))
    assert np.all(vec[1:] > vec[:-1]), 'vector elements are not strictly monotonically increasing'
    ep_mask = np.zeros(vec.shape, bool)
    diff = vec[1:] - vec[:-1]
    ep_mask[1:] = np.logical_or(ep_mask[1:], diff != 1)
    ep_mask[:-1] = np.logical_or(ep_mask[:-1], diff != 1)
    if including_end_Q: 
        ep_mask[0] = True
        ep_mask[-1] = True
    idx = np.nonzero(ep_mask)[0]
    return idx

def split_table_by_key(table:pd.DataFrame, split_key, output_type='list'):
    table = table.sort_values(by=split_key)
    f_val, f_idx, f_c = np.unique(table[split_key].values, return_index=True, return_counts=True)
    assert np.all(np.diff(f_val) >= 1), f"{split_key} is not sorted in ascending order"
    if output_type == 'list':
        t_num_frame = np.max(f_val) + 1
        result = [[] for _ in range(t_num_frame)]
        for f, i, c in zip(f_val, f_idx, f_c):
            result[f] = table.iloc[i : i + c].sort_index()
    elif output_type == 'dict':
        result = {}
        for f, i, c in zip(f_val, f_idx, f_c):
            result[f] = table.iloc[i : i + c].sort_index()
    else: 
        raise NotImplementedError

    return result

def get_table_value_to_idx_dict(table:pd.DataFrame, key, filter=None):
    idx, key_val = bin_data_to_idx_list(table[key].values)
    if filter is not None: 
        selected_Q = filter(key_val)
        idx = idx[selected_Q]
        key_val = key_val[selected_Q]
    v_to_idx_dict = {}
    for i, e in zip(idx, key_val):
        v_to_idx_dict[e] = i

    return v_to_idx_dict

def get_largest_interval_in_1d_array(vec, prepend=0, append=0):
    ints = get_intervals_in_1d_binary_array(vec, prepend, append)
    if ints.shape[0] > 0: 
        if ints.shape[0] > 1: 
            int_length = ints[:, 1] - ints[:, 0]
            max_int_idx = np.argmax(int_length)
            ints = ints[max_int_idx]
        else: 
            ints = ints[0]
        return int(ints[0]), int(ints[1])
    else: 
        return None, None

def ind_coordinate_transform_euclidean(c1_ind, c1_shape, vec_c1_to_c2, c2_shape):
    c1_ind = np.asarray(c1_ind)
    c1_sub = np.vstack(np.unravel_index(c1_ind, c1_shape))
    for i in range(c1_sub.shape[0]):
        c1_sub[i] += vec_c1_to_c2[i]
    c2_ind = np.ravel_multi_index(c1_sub, c2_shape)
    return c2_ind

def list_of_dict_to_dict_of_array(lod, to_numpy_Q=True): 
    result = {}

    for tmp_d in lod: 
        for k, v in tmp_d.items():
            if k not in result: 
                result[k] = []
            result[k].append(v)
    
    if to_numpy_Q: 
        for k, v in result.items():
            result[k] = np.asarray(v)
    
    return result

def dict_of_dict_to_dict_of_array(dod, to_numpy_Q=True):
    result = {}
    result['key'] = list(dod.keys())
    lod = [dod[k] for k in result['key']]
    result |= list_of_dict_to_dict_of_array(lod, to_numpy_Q=False)

    if to_numpy_Q: 
        for k, v in result.items():
            result[k] = np.asarray(v)
    return result

def compute_y_stat_in_x_bin(x, bin_edge, y, fun=np.mean, default_val=np.nan): 
    assert np.all(x.shape == y.shape), "x and y should have the same shape"
    bin_idx_l = get_x_idx_in_bin(x, bin_edge)
    bin_stat = compute_stat_in_bin(bin_idx_l, y, fun, default_val)
    return bin_stat

def get_x_idx_in_bin(x, bin_edge):
    bin_idx = np.digitize(x, bin_edge) - 1
    bin_idx_dict = bin_data_to_idx_list(bin_idx, return_type='dict')
    return [bin_idx_dict.get(i, np.array([], dtype=np.int64)) for i in range(len(bin_edge) - 1)]

def compute_stat_in_bin(bin_idx_l, data, fun=np.mean, default_val=np.nan):
    bin_stat = []
    for i, idx in enumerate(bin_idx_l):
        tmp_data = data[idx]
        tmp_data = tmp_data[~np.isnan(tmp_data)].astype(np.float32)
        if tmp_data.size > 0: 
            bin_stat.append(fun(tmp_data))
        else: 
            bin_stat.append(default_val)

    return bin_stat

def list_to_np_1d_obj_array(input_list):
    # Convert a list to a numpy 1d object array
    output_array = np.empty(len(input_list), dtype=object)
    output_array[:] = input_list
    return output_array