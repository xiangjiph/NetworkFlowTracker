import numpy as np
from scipy import ndimage as ndi

def normalized_integer_data(data):
    # This function return a new array and does not alter the input outside the function
    assert isinstance(
        data, np.ndarray), 'The input data should be a numpy array'
    data_type = data.dtype
    if np.issubdtype(data_type, np.integer):
        type_info = np.iinfo(data_type)
        data = data.astype(np.float32) / type_info.max
    return data

def im_int16_to_uint16(data, method='truncate'):
    # Set all negative values to 0
    assert data.dtype == np.int16, 'The input data is suppoosed to be an int16 array'
    if method == 'truncate':
        data = np.maximum(0, data).astype(np.uint16)
    elif method == 'shift':
        data = (data.astype(np.float32) +
                np.iinfo(np.int16).max).astype(np.uint16)
    return data
     
def downsampling_by_stepping(im, step):
    assert im.ndim == np.size(step), 'Dimension mismatch'
    if im.ndim == 2:
        return im[::step[0], ::step[1]]
    elif im.ndim == 3:
        return im[::step[0], ::step[1], ::step[2]]

def find_cc_1d(mask_1d):
    # Find connected component in 1D logical array 
    if np.any(mask_1d):
        outlier_ep_Q = np.concatenate(([False], mask_1d, [False]), axis=None)
        outlier_ep_Q = np.diff(outlier_ep_Q)
        outlier_ep_idx = np.nonzero(outlier_ep_Q)[0]
        outlier_ep_idx = outlier_ep_idx.reshape((-1, 2))
        return outlier_ep_idx
    else:
        return None

def imresize3(vol, target_shape):
    target_shape = np.array(target_shape) if isinstance(target_shape, (list, tuple)) else target_shape
    input_shape = np.array(vol.shape)
    dt = vol.dtype
    if dt == 'bool':
        itp_order = 0
    else:
        itp_order = 1
    vol = ndi.zoom(vol, target_shape / input_shape, order=itp_order)
    return vol

#region Image enhancement 
def stretch_contrast(input_image, saturate_ptl_low=0.5, saturate_ptl_high=99.95, image_class=None, ignore_zeroQ=False, gamma=1.0):
    """
    Stretch the contrast of an image array of arbitrary dimension by saturating the specified percentile of pixels
    and linear transformation.

    Parameters:
    - input_image: numpy.ndarray - numerical array of arbitrary dimension
    - saturate_ptl_low: float - lower percentile for saturation (default 0.005)
    - saturate_ptl_high: float - higher percentile for saturation (default 0.9995)
    - image_class: str - data type of output image; if not provided, use input image data type
    - ignore_zeroQ: bool - if True, ignore zeros in percentile calculations

    Returns:
    - out_image: numpy.ndarray - contrast-stretched image array
    """
    if image_class is None:
        image_class = input_image.dtype

    if ignore_zeroQ:
        valid_pixels = input_image[input_image != 0]
    else:
        valid_pixels = input_image.flatten()

    if valid_pixels.size > 0:
        low_limit = np.percentile(valid_pixels, saturate_ptl_low)
        high_limit = np.percentile(valid_pixels, saturate_ptl_high)

        if low_limit != high_limit:
            out_image = np.clip(input_image, low_limit, high_limit)
            out_image = (out_image - low_limit) / (high_limit - low_limit)
            if np.abs(gamma - 1) > 0.1: 
                out_image = out_image ** gamma
        else:
            out_image = input_image
    else:
        out_image = input_image

    if image_class in ['uint8', 'uint16', 'int16']:
        if image_class == 'uint8':
            out_image = (out_image * 255).astype(np.uint8)
        elif image_class == 'uint16':
            out_image = (out_image * 65535).astype(np.uint16)
        elif image_class == 'int16':
            out_image = (out_image * 65535 - 32768).astype(np.int16)
    elif image_class in ['single', 'double', 'float']:
        if image_class == 'float':
            out_image = out_image.astype(np.float32)
        else:
            out_image = out_image.astype(np.float64)

    return out_image



#endregion

#region Morphological operations


#endregion Morphological operations