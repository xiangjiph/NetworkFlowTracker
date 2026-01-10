import numpy as np
import scipy as sp
import SimpleITK as sitk
import skimage.registration




class Registration:

    def __init__(self):
        pass


    
def TranslationRegITK(fixed_image, moving_image):
    # Convert to ITK object
    if isinstance(fixed_image, np.ndarray):
        fixed_image = sitk.GetImageFromArray(np.float32(fixed_image))
    if isinstance(moving_image, np.ndarray):
        moving_image = sitk.GetImageFromArray(np.float32(moving_image))
    
    # Setup registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, 
                                                                 minStep=1e-4, 
                                                                 numberOfIterations=200,
                                                                 gradientMagnitudeTolerance=1e-8)
    # Execute registration 
    final_transform = registration_method.Execute(fixed_image, moving_image)
    return final_transform.GetParameters()

def TranslationRegSP(fixed_image, moving_image, shift_image_Q=False):
    reg_result = skimage.registration.phase_cross_correlation(fixed_image, moving_image)
    if shift_image_Q:
        mov_im = sp.ndimage.shift(moving_image, reg_result[0])
        return reg_result[0], mov_im
    else:
        return reg_result[0]