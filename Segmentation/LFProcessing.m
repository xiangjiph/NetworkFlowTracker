classdef LFProcessing < handle
    
methods(Static)
    function im = apply_circ_mask(im, ctr_yx, r)
        im_class = class(im);
        if ~isfloat(im)
            im = single(im);
        end
        im_size = size(im);
        [s1, s2] = ndgrid(0.5 : im_size(1), 0.5 : im_size(2));
        mask = ((s1 - ctr_yx(1)) .^ 2 + (s2 - ctr_yx(2)) .^ 2) <= r ^ 2;
        im = im .* mask;
        im = cast(im, im_class);
    end
    
    function vol = preprocess_im(vol, opts)
        arguments
            vol
            opts.ctr_yx = [];
            opts.cm_r = [];
            opts.valid_bbox_mmxx = [53, 63, 21, 452, 462, 100]
            opts.voxel_size_um = [2, 2, 2.5];
            opts.target_voxel_size_um = 2;
            opts.est_max_bg_int = 250;
        end
        vol = vol(opts.valid_bbox_mmxx(1) : opts.valid_bbox_mmxx(4), ...
            opts.valid_bbox_mmxx(2) : opts.valid_bbox_mmxx(5), ...
            opts.valid_bbox_mmxx(3) : opts.valid_bbox_mmxx(6));
        vol_size = size(vol);
        vol = LFProcessing.apply_circ_mask(vol, vol_size(1:2) ./ 2, min(vol_size(1:2)) / 2);
        vol = medfilt3(vol);
        vol = max(opts.est_max_bg_int, vol);
        vol = uint16(rescale(single(vol)) .^ (1/2) * (2^16 - 1));
        vol_size_target = round(vol_size .* opts.voxel_size_um / opts.target_voxel_size_um);
        vol = imresize3(vol, vol_size_target);
    end
end
    
    
end