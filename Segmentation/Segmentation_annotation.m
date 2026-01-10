clc;clear;close all;
%%
DataManager = FileManager;
dataset_name = 'Lightfield';
stack = 'Zhang2020';
segmentation_version = 'auto';
raw_data_folder = DataManager.fp_raw_data_folder(dataset_name, stack);
mask_folder = DataManager.fp_mask_folder(dataset_name, stack, segmentation_version);
graph_folder = fullfile(DataManager.fp_graph_folder(dataset_name, stack), segmentation_version);
itk_folder = DataManager.fp_itksnap_data_folder(dataset_name, stack, segmentation_version);
info_fp = fullfile(DataManager.fp_processed_data(dataset_name, stack), 'info.mat');
info = DataManager.load_data(info_fp);
vol_folder_list = info.vol_folders;
mip_folder = info.mip_folder;
%%
z_idx = 6;
tmp_folder = vol_folder_list{z_idx};
tmp_fp = fullfile(mask_folder, sprintf('%s.mat', tmp_folder));
tmp_result = load(tmp_fp);
%% Generate annotation files 
vis_im = tmp_result.image;
vis_mask = uint8(tmp_result.mask);
itk_f_str = fullfile(itk_folder, sprintf('%s_%s', tmp_folder, datestr(now, 'YYYYmmDD_HHMMSS')));
DataManager.visualize_itksnap(vis_im, vis_mask, itk_f_str, true);
%% Visualize blood flow 
valid_bbox_mmxx = info.valid_bbox_mmxx;
raw_folder = fullfile(raw_data_folder, tmp_folder);
load_range = [1, 200];
data = cell(load_range(2), 1);
for i = 1 : load_range(2)
    tmp_fn = fullfile(raw_folder, sprintf('%05d.tif', i));
    tmp_data = DataManager.load_single_tiff(tmp_fn);
    tmp_data = tmp_data(:, :, 21:end-1);
    tmp_data = tmp_data(valid_bbox_mmxx(1) : valid_bbox_mmxx(4), ...
                        valid_bbox_mmxx(2) : valid_bbox_mmxx(5), :);
    data{i} = tmp_data;
    fprintf('Finish reading file %d\n', i);
end
    %%
vis_z = 18;
vis_z_range = vis_z + [-10 : 10];
vis_mip = max(vis_im(:, :, vis_z_range), [], 3);
vis_z_video = cell(load_range(2), 1);
for i = 1 : load_range(2)
    tmp_im = fun_stretch_contrast(max(data{i}(:, :, vis_z_range), [], 3));
    tmp_rbg = repmat(vis_mip, 1, 1, 3) * 0.5;
    tmp_rbg(:, :, 1) = tmp_im;
    tmp_rbg = im2uint8(tmp_rbg);
    vis_z_video{i} = tmp_rbg;
end
vis_z_video = cat(4, vis_z_video{:});
implay(permute(vis_z_video, [2, 1, 3, 4]));
%% Try multi-color skeletonization 
itk_file = dir(fullfile(itk_folder, '*mask.nii.gz'));
if numel(itk_file) > 1
   % select by time?  
end
mask_a = DataManager.load_data(fullfile(itk_file.folder, itk_file.name));
%%
% Seems to work, though this is probably not an efficient way to do it.
min_cc_size = 5 ^ 3;
mask_art = (mask_a == 1 | mask_a == 2);
mask_art = bwareaopen(mask_art, min_cc_size);
mask_vein = (mask_a == 1 | mask_a == 3);
mask_vein = bwareaopen(mask_vein, min_cc_size);

skl_a = bwskel(imfill(mask_art, 'holes'));
skl_v = bwskel(imfill(mask_vein, 'holes'));
% check consistency 
f = figure();
a = axes(f);
imshowpair(max(skl_a, [], 3), max(skl_v, [], 3));
%%
skl_m = skl_a | skl_v;
vsl_skl = skl_m;
vsl_graph = fun_skeleton_to_graph(vsl_skl);

gr_para.internal_offset = 0;
gr_para.pruning_max_length = 4;
gr_para.max_bilink_loop_length = 10;
gr_para.max_self_loop_length = 15;
[vessel_graph, pruning_info_str_1] = fun_graph_delete_hairs_and_short_loops(...
    vsl_graph, vis_im, gr_para);
mask_dt = bwdist(~mask_a);
vessel_graph = fun_graph_add_radius(vessel_graph, mask_dt, 1);

recon_mask = fun_graph_reconstruction(vessel_graph, 1);
recon_ind = cat(1, vessel_graph.node.cc_ind{:}, vessel_graph.link.cc_ind{:});
recon_label = mask_a(recon_ind);
recon_skl = zeros(vessel_graph.num.mask_size, 'uint8');
recon_skl(recon_ind) = recon_label;
recon_skl_mip = max(recon_skl, [], 3);
% Compare final skeleton with the initial skeleton 
f = figure;
a = axes(f);
imshowpair(recon_skl_mip > 0,  max(skl_m, [], 3));
a.DataAspectRatio = [1,1,1];
% Reconstruct network with labels 
link_label = cellfun(@(x) max(mask_a(x)), vessel_graph.link.cc_ind);
link_vxl_label = repelem(link_label, vessel_graph.link.num_voxel_per_cc);
node_label = cellfun(@(x) max(mask_a(x)), vessel_graph.node.cc_ind);
node_vxl_label = repelem(node_label, vessel_graph.node.num_voxel_per_cc);
recon_r = mask_dt(recon_ind);
recon_label = cat(1, node_vxl_label, link_vxl_label);
recon_label_array = fun_skeleton_reconstruction_label(recon_ind, recon_r, ...
    recon_label, vessel_graph.num.mask_size);

recon_itk_folder = strrep(itk_folder, segmentation_version, 'recon');
recon_itk_f_str = fullfile(recon_itk_folder, sprintf('%s_%s', tmp_folder, datestr(now, 'YYYYmmDD_HHMMSS')));
DataManager.visualize_itksnap(vis_im, recon_label_array, recon_itk_f_str, true);
%% Save data
result = struct;
result.data_group = dataset_name;
result.date_name = stack;
result.version = 'annotated';
result.image = vis_im;
result.label_skl = recon_skl_mip;
result.label_mask = mask_a;
result.fp = fullfile(DataManager.fp_mask_folder(dataset_name, stack, result.version), ...
    sprintf('%s_%s_%s_%s.mat', dataset_name, stack, tmp_folder, result.version));
DataManager.write_data(result.fp, result);
