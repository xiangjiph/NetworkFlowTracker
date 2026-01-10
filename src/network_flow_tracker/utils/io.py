from scipy.io import savemat, loadmat
import os, sys, platform
import h5py, json, pickle
import tifffile as tiff
import SimpleITK as sitk
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


class DataManager:
    def __init__(self, hostname=None, data_group=None, data_name=None, use_scratch_Q=False):
        self.use_scratch_Q = use_scratch_Q
        self.data_group = data_group
        self.data_name = data_name

        if hostname is None:
            self.HOSTNAME = platform.node()
        else:
            self.HOSTNAME = hostname.lower()

        if self.HOSTNAME == 'XiangJi-PC'.lower():
            self.DATA_ROOT_PATH = 'D:\\data\\Vessel'
            self.SCRATCH_ROOT_PATH = self.DATA_ROOT_PATH
            self.SCRIPT_PATH = 'D:\\Github\\'
        elif self.HOSTNAME in ['bird', 'bird.dk.ucsd.edu']:
            self.DATA_ROOT_PATH = '/net/birdstore/Vessel'
            self.SCRIPT_PATH = '/home/xij072/Documents/Github'
            self.SCRATCH_ROOT_PATH = '/scratch/Vessel'
            self.SERVER_ROOT_PATH = '/net/birdstore'       

    def fp_root(self, use_scratch_Q=None):
        if use_scratch_Q is None: 
            use_scratch_Q = self.use_scratch_Q
        if use_scratch_Q:
            return self.SCRATCH_ROOT_PATH
        else:
            return self.DATA_ROOT_PATH
    
    @property
    def fpr_dataset(self):
        return os.path.join(self.data_group, self.data_name)
    
    def fp_dataset(self, scratchQ=None):
        if scratchQ is None: 
            scratchQ = self.use_scratch_Q
        if scratchQ:
            return os.path.join(self.SCRATCH_ROOT_PATH, self.fpr_dataset)
        else: 
            return os.path.join(self.DATA_ROOT_PATH, self.fpr_dataset)
    
    @property
    def fpr_processed_data_root(self):
        return os.path.join(self.fpr_dataset, 'processed_data')
    
    def fp_processed_data_root(self, scratchQ=None):
        return os.path.join(self.fp_dataset(scratchQ=scratchQ), 'processed_data')
    
    @property
    def fpr_raw_data(self):
        return os.path.join(self.fpr_dataset, 'raw_data')
    
    def fp_raw_data(self, scratchQ=None):
        return os.path.join(self.fp_dataset(scratchQ=scratchQ), 'raw_data')
    
    @property
    def fpr_visualization_folder(self):
        return os.path.join(self.fpr_dataset, 'visualization')
    
    def fp_visualization_folder(self, scratchQ=None):
        return os.path.join(self.fp_dataset(scratchQ), 'visualization')
    
    @property
    def fpr_mask(self):
        return self.fpr_process_data_folders('mask')
    
    def fpr_process_data_folders(self, data_type):
        return os.path.join(self.fpr_processed_data_root, data_type)
    
    @staticmethod
    def save_data(fp, data, verboseQ=False):
        return save_data(fp, data, verboseQ)
    
    @staticmethod
    def load_data(fp):
        return load_data(fp)
        

def save_data(fp, data, verboseQ=False):
    folder, fn = os.path.split(fp)
    fn, ext = os.path.splitext(fn)
    os.makedirs(folder, exist_ok=True)
    if ext == '.mat':
        if isinstance(data, dict):
            save_dict_as_mat_file(fp, data)
    elif ext == '.h5':
        save_h5(fp, data)
    elif ext == '.json':
        write_json(fp, data)
    elif ext == '.pickle':
        save_pkl(fp, data)
    elif ext == '.txt':
        write_text(fp, data)
    elif ext == '.tif':
        tiff.imwrite(fp, data)
    elif ext == '.nii':
        write_nii(fp, data)
    else:
        raise "Unrecognized file type"
    if verboseQ:
        print(f"Finish writing file {fp}")

def load_data(fp, arg=None):
    fn, ext = os.path.splitext(fp)
    if ext == '.pickle':
        return load_pickle(fp)
    elif ext == '.h5':
        return load_h5(fp)
    elif ext == '.json':
        return load_json(fp)
    elif ext in ('.tiff', '.tif'):
        return load_tiff(fp, arg)
    elif ext == '.mat':
        return loadmat(fp, squeeze_me=False)
    elif ext == '.nii':
        return load_nii(fp)
    elif ext == '.gz':
        if fp.endswith('.nii.gz'):
            return load_nii(fp)
        else:
            raise "Unrecognized file type"
    else: 
        raise "Unrecognized file type"
    
#region Saving data
# def save_dict_as_h5(fp, data_dict):
#     with h5py.File(fp, "w") as h5file:
#     # Function to recursively save dictionary to HDF5
#         def save_dict_to_h5(group, path, dic):
#             for key, value in dic.items():
#                 if isinstance(value, dict):  # If value is a dictionary, create a group
#                     sub_path = path + '/' + key
#                     subgroup = group.create_group(sub_path)
#                     save_dict_to_h5(subgroup, sub_path, value)
#                 elif isinstance(value, (np.ndarray, list)):  # If value is an array or list
#                     group.create_dataset(key, data=value)
#                 elif isinstance(value, (int, float, str)):  # If value is scalar
#                     group.attrs[key] = value  # Store as an attribute
#                 else:
#                     raise TypeError(f"Unsupported data type for key: {key}")

#         # Save the dictionary
#         save_dict_to_h5(h5file, data)

def save_dict_as_mat_file(fp, data):
    savemat(fp, data, appendmat=True, format='5',
            long_field_names=True, do_compression=False, oned_as='column')
    
def save_pkl(fp, data):
    parent_dir, tmp  = os.path.split(fp)
    os.makedirs(parent_dir, exist_ok=True)
    with open(fp, 'wb') as file:
        pickle.dump(data, file)

def save_h5(fp: str, data):
    if isinstance(data, dict):
        save_dict_as_h5(fp, data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def save_dict_as_h5(fp: str, data: dict):
    """
    Saves a nested dictionary to an HDF5 file using recursive calls. The function
    handles dictionaries, numpy arrays, lists, tuples, and basic scalars. It stores
    arrays and lists as HDF5 datasets and scalars as attributes of their parent group.

    Args:
    fp (str): The file path where the HDF5 file will be saved.
    data_dict (dict): The dictionary to save. This dictionary can contain nested
                      dictionaries, numpy arrays, lists, tuples, and scalars.

    Raises:
    TypeError: If a value in the dictionary is of an unsupported type.

    Example:
    >>> data = {
        'group1': {
            'dataset1': np.array([1, 2, 3]),
            'attribute1': 45
        },
        'group2': {
            'dataset2': [4, 5, 6]
        },
        'attribute2': 'example'
    }
    >>> save_dict_as_h5('data.h5', data)
    """
    with h5py.File(fp, "w") as h5file:
        def save_dict_to_h5(path, dic):
            """Recursively save nested dictionary to HDF5 groups and datasets."""
            for key, value in dic.items():
                if isinstance(value, dict):  # Handle nested dictionary
                    subgroup = path.create_group(key)
                    save_dict_to_h5(subgroup, value)
                elif isinstance(value, (np.ndarray, list, tuple)):  # Handle arrays and lists
                    path.create_dataset(key, data=value)
                elif isinstance(value, (int, float, str, np.integer, np.floating)):  # Handle scalars
                    path.attrs[key] = value  # Store as an attribute
                else:
                    raise TypeError(f"Unsupported data type for key: {key}")

        # Save the dictionary
        save_dict_to_h5(h5file, data)

def sparse_mat_to_dict(mat):
    result = {
        'type': mat.__class__.__name__, 'data': mat.data, 'indices': mat.indices, 
        'indptr': mat.indptr, 'shape': mat.shape
    }
    return result

def dict_to_sparse_mat(d):
    import scipy.sparse as sps
    if d['type'] == 'csr_matrix':
        mat = sps.csr_matrix((d['data'], d['indices'], d['indptr']), shape=d['shape'])
    elif d['type'] == 'csc_matrix':
        mat = sps.csc_matrix((d['data'], d['indices'], d['indptr']), shape=d['shape'])
    else:
        raise TypeError(f"Unrecognize sparse matrix type {d['type']}")
    return mat

def write_text(fp, txt):
    f = open(fp, "w")
    f.write(txt)
    f.close()

def write_json(fp, data):
    assert isinstance(data, dict), 'Only support writing diction into a json file'
    with open(fp, 'w') as file:
        json.dump(data, file)

def write_nii(fp, data):
    sitk_data = sitk.GetImageFromArray(data)
    sitk_data.SetOrigin((0.0, 0.0, 0.0))
    sitk_data.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(sitk_data, fp)
#endregion 

####################
#region Loading data 
####################
def load_pickle(fp):
    with open(fp, 'rb') as file:
        return pickle.load(file)

def load_json(fp):
    with open(fp) as f:
        return json.load(f)
    
def load_tiff(fp, sec_range=None):
    if sec_range is None: 
        return tiff.imread(fp)
    elif len(sec_range) == 2:
        with tiff.TiffFile(fp) as t:
            selected_pages = t.pages[sec_range[0] : sec_range[1]]
            tmp = [page.asarray() for page in selected_pages]
            return np.stack(tmp)
        
def load_nii(fp, cvrt2nyQ=True):
    data = sitk.ReadImage(fp)
    if cvrt2nyQ:
        data = sitk.GetArrayFromImage(data)
    return data

def load_h5(fp):
    try:
        with h5py.File(fp, 'r') as hdf:
            def _read_hdf5_to_dict(hdf_group):
                """
                Recursively reads HDF5 groups and datasets back into a dictionary.
                """
                result = {}
                for attr_name, attr_val in hdf_group.attrs.items():
                    result[attr_name] = attr_val

                for key, item in hdf_group.items():
                    if isinstance(item, h5py.Dataset):
                        data = item[()] # Use [()] to load all data
                        # More processing if needed
                        result[key] = data
                    elif isinstance(item, h5py.Group):
                        result[key] = _read_hdf5_to_dict(item)  # Recurse into groups
                return result
            
            data = _read_hdf5_to_dict(hdf)
            return data
    except Exception as e:
        print(f"Failed to read file {fp}: {e}")
        return {}
    
def read_mat_file_as_h5(filename, reorder_dimensions=False):
    """
    Reads a MATLAB v7.3 .mat file using h5py and returns a nested Python dictionary
    that mirrors the MATLAB data structure.

    Parameters
    ----------
    filename : str
        Path to the .mat file (v7.3).
    reorder_dimensions : bool, optional
        If True, re-order (transpose) arrays from MATLAB (column-major) to Python 
        (row-major) to match typical MATLAB shape conventions in Python. 
        Default is False.

    Returns
    -------
    dict
        A nested dictionary of Python objects representing the MATLAB data.

    Examples
    --------
    data_dict = read_mat_file('example_v7.3.mat', reorder_dimensions=True)
    """
    with h5py.File(filename, 'r') as f:
        # Top-level file may have multiple variables/groups
        out = {}
        for key in f.keys():
            out[key] = _read_h5_item(f[key], reorder_dimensions)
    return out


def _read_h5_item(item, reorder_dimensions):
    """
    Recursively read an h5py Group or Dataset, interpreting MATLAB 
    structures, cells, numeric/logical arrays, char arrays, strings, etc.
    """
    if isinstance(item, h5py.Group):
        # Check for MATLAB class attribute
        matlab_class = item.attrs.get('MATLAB_class', None)
        if matlab_class is not None:
            matlab_class = matlab_class.decode('utf-8', errors='ignore')

        if matlab_class == 'struct':
            # MATLAB struct -> dictionary
            return _read_h5_structure(item, reorder_dimensions)
        elif matlab_class == 'cell':
            # MATLAB cell -> list
            return _read_h5_cell(item, reorder_dimensions)
        else:
            # If not recognized specifically, parse as group of sub-items
            return {
                sub_key: _read_h5_item(sub_item, reorder_dimensions)
                for sub_key, sub_item in item.items()
            }

    elif isinstance(item, h5py.Dataset):
        return _read_dataset(item, reorder_dimensions)

    # Fallback: just return None if unrecognized
    return None


def _read_dataset(dset, reorder_dimensions):
    """
    Read a MATLAB dataset from the HDF5 file and convert to 
    Python/numpy types where appropriate.
    """
    # Read the raw data
    data = dset[()]

    # Check MATLAB_class
    matlab_class = dset.attrs.get('MATLAB_class', None)
    if matlab_class is not None:
        matlab_class = matlab_class.decode('utf-8', errors='ignore')

    # Handle char arrays (stored as uint16 or uint8 of shape (n, ))
    if matlab_class == 'char':
        # MATLAB char arrays often stored as numeric arrays of ASCII/Unicode codes
        if data.ndim == 1:
            # 1D char array
            data = ''.join(chr(x) for x in data)
        else:
            # If multi-dimensional, each column can be a char "row" in MATLAB
            # Flatten and decode:
            data = ''.join(chr(x) for x in data.flatten(order='F'))
        return data

    # Handle string arrays (R2016b+ style), can also appear as vlen object arrays
    if matlab_class == 'string':
        # If it's a vlen string dataset, can read directly asstr()
        if data.dtype.type is np.string_ or data.dtype.type is np.bytes_:
            # Convert raw bytes to Python string
            # If it's multiple strings, shape might be > 1
            data = data.astype('U')  # convert to Unicode
            if data.size == 1:
                data = data.item()     # single string
            else:
                data = data.tolist()   # array of strings
        return data

    # Handle logical
    if matlab_class == 'logical':
        data = data.astype(bool)

    # By default, treat as numeric or otherwise.
    # Reorder dimensions if requested and array has more than 1 dimension.
    if reorder_dimensions and data.ndim > 1:
        # MATLAB uses Fortran order, Python uses C order, so transpose all axes
        # e.g. if shape is (m, n, p), we do np.transpose to reorder them
        # to get consistent indexing as in MATLAB. 
        data = np.transpose(data, axes=range(data.ndim)[::-1])

    return data


def _read_h5_structure(group, reorder_dimensions):
    """
    Read a MATLAB struct group. Each field can be another dataset, cell, struct, etc.
    The structure is typically:
      group -> { field1, field2, ... }
    """
    # Each field name is a subgroup or dataset
    out_dict = {}
    for field_name, field_item in group.items():
        out_dict[field_name] = _read_h5_item(field_item, reorder_dimensions)
    return out_dict


def _read_h5_cell(group, reorder_dimensions):
    """
    Read a MATLAB cell array from an h5py.Group.  In a v7.3 file, 
    the cell array data might be stored as subgroups or datasets with references.
    A typical layout is that each cell element is a separate dataset or group 
    within the cell Group.  The shape of the cell array is often stored in the 
    'dims' attribute.  
    """
    # The shape can be found in the 'dims' attribute if present
    cell_dims = group.attrs.get('dims', None)
    if cell_dims is not None:
        cell_dims = tuple(cell_dims.astype(int)[::-1])  # reverse for Python

    # Gather child items.  Each key might look like '0', '1', etc. for each cell element
    # In some files, there's also a '#refs#' or similar internal group.
    keys = sorted([k for k in group.keys() if not k.startswith('#')], key=lambda x: int(x))
    elements = [_read_h5_item(group[k], reorder_dimensions) for k in keys]

    # If we know the shape, try to reshape
    if cell_dims is not None and np.prod(cell_dims) == len(elements):
        elements = np.array(elements, dtype=object).reshape(cell_dims)
        return elements.tolist()
    else:
        # Fallback: just return as a flat list
        return elements


####################
#endregion
####################

#####################
#region Visualization
#####################
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def print_image(fig_hdl, fig_fp):
    folder_path = os.path.dirname(fig_fp)
    if folder_path and (not os.path.isdir(folder_path)): 
        os.makedirs(folder_path)

    if fig_fp.endswith('eps'):
        # Change the font for editable eps file? 
        fig_hdl.savefig(fig_fp, format='eps', bbox_inches='tight')
    elif fig_fp.endswith('png'):
        fig_hdl.savefig(fig_fp, format='png', dpi=300, bbox_inches='tight')
    elif fig_fp.endswith('pickle'):
        with open(fig_fp, 'wb') as f:
            pickle.dump(fig_hdl, f)
    elif fig_fp.endswith('pdf'): 
        fig_hdl.savefig(fig_fp, dpi=300, bbox_inches='tight')
    else:
        fig_hdl.savefig(fig_fp)

def print_image_in_several_formats(fig_hdl, fig_fp, format_list=['.pdf', '.pickle', '.png']):
    fn, ext = os.path.splitext(fig_fp)
    if ext not in format_list:
        format_list.append(ext)
    for ie in format_list:
        ifn = fn + ie
        print_image(fig_hdl, ifn)
        print('Finish saving figure as {:s}'.format(ifn))

def load_matplotlib_pickle_file(fp):
    with open(fp, 'rb') as f:
        fig = pickle.load(f)

    new_fig = plt.figure()
    new_manager = new_fig.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    fig.show()
    return fig
#####################
#endregion
#####################

#################
#region Utilities
#################
def check_folder_existence(fp, createQ=True):
    parent_dir, dir_name = os.path.split(fp)
    parent_dir_exist_Q = os.path.isdir(parent_dir)
    if createQ and (not parent_dir_exist_Q):
        os.makedirs(parent_dir, exist_ok=True)
    return parent_dir_exist_Q

def filepath_correct_filesep(fp):
    if platform.system() in {'Linux', 'Darwin', 'Unix'}:
        return fp.replace("\\", "/")
    elif platform.system() == 'Windows':
        return fp.replace("/", "\\")
#################
#endregion
#################