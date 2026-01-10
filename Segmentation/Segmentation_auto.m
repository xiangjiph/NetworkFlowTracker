clc;clear;close all;
%%
DataManager = FileManager;
dataset_name = 'Lightfield';
stack = 'Zhang2020';
segmentation_version = 'auto';
raw_data_folder = DataManager.fp_raw_data_folder(dataset_name, stack);
mask_folder = DataManager.fp_mask_folder(dataset_name, stack, segmentation_version);
graph_folder = fullfile(DataManager.fp_graph_folder(dataset_name, stack), segmentation_version);
info_fp = fullfile(DataManager.fp_processed_data(dataset_name, stack), 'info.mat');
%%
if ~isfile(info_fp )
    info = struct;
    info.dataset = dataset_name;
    info.stack = stack;
    info.raw_data_folder = raw_data_folder;
    info.vol_folders = sort(arrayfun(@(x) x.name, dir(fullfile(raw_data_folder, 'z*')), ...
        'UniformOutput', false));
    info.mip_folder = fullfile(DataManager.fp_processed_data(dataset_name, stack), 'mip');
    info.valid_bbox_mmxx = [53, 63, 21, 452, 462, 80];
    info.raw_voxel_size_um = [2, 2, 2.5];
    info.target_voxel_size_um = 2;
    info.fp = fullfile(DataManager.fp_processed_data(dataset_name, stack), 'info.mat');
    DataManager.write_data(info.fp, info);    
else
    info = DataManager.load_data(info_fp);
end

voxel_size_um = info.raw_voxel_size_um;
target_voxel_size_um = info.target_voxel_size_um;
valid_bbox_mmxx = info.valid_bbox_mmxx;
est_max_bg_int = 250;

gr_para.internal_offset = 0;
gr_para.pruning_max_length = 4;
gr_para.max_bilink_loop_length = 10;
gr_para.max_self_loop_length = 15;
%%
vol_folder_list = info.vol_folders;
% mip_folder = info.mip_folder;
mip_folder = 'D:\data\Vessel\Lightfield\Zhang2020\processed_data\mip';

%%
for test_vol_idx = 2 : numel(vol_folder_list)
%     test_vol_idx = 1;
    test_folder_name = vol_folder_list{test_vol_idx};
    test_mip = DataManager.load_single_tiff(fullfile(mip_folder, sprintf('%s_mip.tif', test_folder_name)));
    test_mip = test_mip(:, :, 21:end-1);
    test_mip = test_mip(valid_bbox_mmxx(1) : valid_bbox_mmxx(4), ...
        valid_bbox_mmxx(2) : valid_bbox_mmxx(5), :);
    vol_size = size(test_mip);
    
    test_mip_e = LFProcessing.apply_circ_mask(test_mip, vol_size(1:2) ./ 2, min(vol_size(1:2)) / 2);
    test_mip_e = medfilt3(test_mip_e);
    test_mip_e = max(est_max_bg_int, test_mip_e);
    test_mip_e = uint16(rescale(single(test_mip_e)) .^ (1/2) * (2^16 - 1));
    %% Overall stack
    vol_size_target = round(vol_size .* voxel_size_um / target_voxel_size_um);
    test_mip_i = imresize3(test_mip_e, vol_size_target);
    %% Segmentation
    seg_parameters = struct;
    seg_parameters.voxel_length_um = 2;
    seg_parameters.data_type = 'uint16';
    seg_parameters.rod_filter_radius_um = 2;
    seg_parameters.rod_filter_length_um = round(6*seg_parameters.rod_filter_radius_um + 1);
    seg_parameters.rod_filter_num_omega = 6;
    seg_parameters.vesselness.DoG_scale_list_um = [1, 2, 4] ./ seg_parameters.voxel_length_um;
    seg_parameters.vesselness_th = 0.01;
    seg_parameters.adp_th_scale_1_um = 8;
    seg_parameters.adp_th_scale_2_um = 16;
    seg_parameters.morp_min_cc_size = 27;
    seg_parameters.max_pool_size = 8;
    seg_parameters.min_bg_std = 100;
    seg_parameters.add_vxl_min_int = 1e4;
    
    vsl_mask = fun_mouselight_segmentation_1um_cube(test_mip_i, seg_parameters);
    vessel_mask = imfill(vsl_mask, 'holes');
    vessel_mask = imclose(vessel_mask, strel('cube', 3));
    %% Graph construction
    vsl_skl = bwskel(vessel_mask);
    mask_dt = bwdist(~vessel_mask);
    vsl_skl = fun_skeleton_recentering_within_mask(vsl_skl, vessel_mask, ...
        mask_dt > 0);
    vsl_graph = fun_skeleton_to_graph(vsl_skl);
    
    [vessel_graph, pruning_info_str_1] = fun_graph_delete_hairs_and_short_loops(...
        vsl_graph, test_mip_i, gr_para);
    vessel_graph = fun_graph_add_radius(vessel_graph, mask_dt, 1);
    %% Save info
    result = struct;
    result.image = test_mip_i;
    result.raw_valid_bbox = valid_bbox_mmxx;
    result.mask_size = vol_size_target;
    result.graph = vessel_graph;
    result.voxel_size_um = target_voxel_size_um;
    result.mask = vessel_mask;
    result.seg_para = seg_parameters;
    result.note = 'Remove top 20 sections and the last section before resizing';
    result.fp = fullfile(mask_folder, sprintf('%s.mat', test_folder_name));
    save(result.fp, '-struct', 'result');
end
%% Visualization
vis_mask = uint8(vessel_mask);
vis_mask(~vessel_mask & vsl_mask) = 2;
vis_mask(vessel_mask & ~vsl_mask) = 3;
itk_folder = DataManager.fp_itksnap_data_folder(dataset_name, stack, 'vol');
itk_f_str = fullfile(itk_folder, sprintf('%s_%s', test_folder_name, datestr(now, 'YYYYmmDD_HHMMSS')));
DataManager.visualize_itksnap(test_mip_i, vis_mask, itk_f_str, true);