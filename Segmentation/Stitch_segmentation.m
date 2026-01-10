clc;clear;close all;
%%
DataManager = FileManager;
dataset_name = 'Lightfield';
stack = 'Zhang2020';
segmentation_version = 'auto';
raw_data_folder = DataManager.fp_raw_data_folder(dataset_name, stack);
processed_data_folder = DataManager.fp_processed_data(dataset_name, stack);
mask_folder = fullfile(processed_data_folder, 'mask', segmentation_version);
graph_folder = fullfile(processed_data_folder, 'graph', segmentation_version);
itk_folder = fullfile(processed_data_folder, 'itk', segmentation_version);
info_fp = fullfile(processed_data_folder, 'info.mat');
info = DataManager.load_data(info_fp);
vol_folder_list = info.vol_folders;
mip_folder = info.mip_folder;
data_info = DataManager.load_data(strrep(info_fp, 'info', 'data_info'));
%% Load ITK annotation 
stitch_z_list = 1 : 5;
num_vol = numel(stitch_z_list);
itk_mask_files = dir(fullfile(itk_folder, "z*.gz"));
itk_mask_files = arrayfun(@(x) x.name, itk_mask_files, 'UniformOutput', false);
itk_mask_files = sort(itk_mask_files);
[im, itk_mask] = deal(cell(num_vol, 1));
valid_sec = 1 : 75;
min_cc_size = 9;
for z_idx = 1 : num_vol
    tmp_folder = vol_folder_list{z_idx};
    tmp_fp = fullfile(mask_folder, sprintf('%s.mat', tmp_folder));
    tmp_result = load(tmp_fp);
    tmp_itk_file = fullfile(itk_folder, itk_mask_files{z_idx});
    tmp_itk_label = DataManager.load_data(tmp_itk_file);
    tmp_itk_mask = imfill(tmp_itk_label > 0, 'holes');
    tmp_itk_mask = bwareaopen(tmp_itk_mask, min_cc_size);
    tmp_itk_label(~tmp_itk_mask) = 0;
    im{z_idx} = tmp_result.image;
    itk_mask{z_idx} = tmp_itk_label;
end
%%
vol_size = size(im{1});
z_0_um = ((1 : num_vol) - 1) * 100;
vxl_size_um = double(data_info.target_voxel_size_um);
z_0_pxl = z_0_um ./ vxl_size_um + 1;
z_1_pxl = z_0_pxl + vol_size(3) - 1;

disp_vec = zeros(3, num_vol);
reg_corr = zeros(num_vol, 1);
int_th = 1e4;
% debug_Q = true;
for z_idx = 2 : num_vol
    fix_im = im{z_idx - 1}(:, :, 51 : 75);
    mov_im = im{z_idx}(:, :,  1 : 25);
    fix_im(fix_im < int_th) = 0;
    mov_im(mov_im < int_th) = 0;
    [tmp_t_xyz, tmp_c, ~] = MaskedTranslationRegistration(fix_im, mov_im, fix_im > int_th, ...
        mov_im > int_th, [2, 2, 5]);
    disp_vec(:, z_idx) = tmp_t_xyz([2, 1, 3]);
    reg_corr(z_idx) = tmp_c;
end
disp_vec(:, reg_corr < 0.2) = 0;
disp_vec_c = cumsum(disp_vec, 2);
%% Check pairwise registration 
% Look for missing segments 
im_reg_adj = cell(num_vol, 1);
im_reg_adj_mip = cell(3, num_vol);
for i = 1 : num_vol
    tmp_im = imtranslate(im{i}, disp_vec_c([2, 1, 3], i));
    im_reg_adj{i} = tmp_im;    
    for j = 1 : 3
        im_reg_adj_mip{j, i} = squeeze(max(tmp_im, [], j));
    end    
end
%%
num_pair = num_vol - 1;
f = figure;
for vis_idx = 2 : num_vol
    vis_im_1 = im_reg_adj{vis_idx - 1}(:, :, 51 : 75);
    vis_im_2 = im_reg_adj{vis_idx}(:, :, 1 : 25);
    vis_mip = cell(3, 2);
    for i = 1 : 3
        vis_mip{i, 1} = squeeze(max(vis_im_1, [], i));
        vis_mip{i, 2} = squeeze(max(vis_im_2, [], i));
    end
    tmp_a = nexttile();
    imshowpair(vis_mip{3, 1}, vis_mip{3, 2});
    tmp_a.Title.String = sprintf('%d vs %d', vis_idx - 1, vis_idx);
end
%% Stitch images and annotations 
bbox_mm = ones(3, num_vol);
bbox_xx = repmat(vol_size', 1, num_vol);
bbox_mm(3, :) =  z_0_pxl;
bbox_xx(3, :) = z_1_pxl;

bbox_mm_r = bbox_mm + disp_vec_c;
bbox_xx_r = bbox_xx + disp_vec_c;
g_bbox_mm_r = min(bbox_mm_r, [], 2);
g_bbox_xx_r = max(bbox_xx_r, [], 2);
g_bbox_ll = g_bbox_xx_r - g_bbox_mm_r + 1;
bbox_mm_r_1 = bbox_mm_r - g_bbox_mm_r + 1;
bbox_xx_r_1 = bbox_xx_r - g_bbox_mm_r + 1;

whole_im = zeros(g_bbox_ll', 'like', im{1});
whole_itk_mask = zeros(g_bbox_ll', 'like', itk_mask{1});
extra_blank_z_sec = 10;
for i = 1 : num_vol
    add_vol = im{i};
    if i > 0
        add_vol = single(add_vol);
        if i == 1
            gf_sigma = 10;
        else
            gf_sigma = 3;
        end
        add_vol = rescale(max(0, add_vol - imgaussfilt(add_vol, gf_sigma))); 
        add_vol = im2uint16(add_vol);
    end    
    add_vol = add_vol(:, :, valid_sec);
    add_itk_mask = itk_mask{i}(:, :, valid_sec);
    
    whole_im(bbox_mm_r_1(1, i) : bbox_xx_r_1(1, i), ...
             bbox_mm_r_1(2, i) : bbox_xx_r_1(2, i), ...
             bbox_mm_r_1(3, i) : bbox_xx_r_1(3, i)) = max(add_vol, ...
                 whole_im(bbox_mm_r_1(1, i) : bbox_xx_r_1(1, i), ...
                          bbox_mm_r_1(2, i) : bbox_xx_r_1(2, i), ...
                          bbox_mm_r_1(3, i) : bbox_xx_r_1(3, i)));  
                      
    whole_itk_mask(bbox_mm_r_1(1, i) : bbox_xx_r_1(1, i), ...
             bbox_mm_r_1(2, i) : bbox_xx_r_1(2, i), ...
             bbox_mm_r_1(3, i) : bbox_xx_r_1(3, i)) = max(add_itk_mask, ...
                 whole_itk_mask(bbox_mm_r_1(1, i) : bbox_xx_r_1(1, i), ...
                          bbox_mm_r_1(2, i) : bbox_xx_r_1(2, i), ...
                          bbox_mm_r_1(3, i) : bbox_xx_r_1(3, i)));  
end
% Cross the volume using the top sub-volume bounding box
whole_im = whole_im(bbox_mm_r_1(1, 1) : bbox_xx_r_1(1, 1), ...
                    bbox_mm_r_1(2, 1) : bbox_xx_r_1(2, 1), :);
whole_itk_mask = whole_itk_mask(bbox_mm_r_1(1, 1) : bbox_xx_r_1(1, 1), ...
                    bbox_mm_r_1(2, 1) : bbox_xx_r_1(2, 1), :);
implay(whole_im);
%%
mask_cc = bwconncomp(whole_itk_mask > 0);
mask_cc_size = cellfun(@numel, mask_cc.PixelIdxList);
small_cc_Q = mask_cc_size <= 250;
whole_itk_mask(cat(1, mask_cc.PixelIdxList{small_cc_Q})) = 0;
%%
itk_f_name = 'whole';
itk_f_str = fullfile(itk_folder, sprintf('%s_%s', itk_f_name, datestr(now, 'YYYYmmDD_HHMMSS')));
DataManager.visualize_itksnap(whole_im, uint8(whole_itk_mask), itk_f_str, true);
%% Registration result 
result = struct('dataset_name', dataset_name, 'stack', stack, ...
    'segmentation_version', segmentation_version, 'disp_vec', disp_vec, ...
    'disp_vec_c', disp_vec_c, 'itk_f_name', itk_f_name, 'stitched_im', ...
    whole_im, 'stitched_mask', uint8(whole_itk_mask));
[itk_folder, itk_fn, ~] = fileparts(itk_f_str);
result.fp = fullfile(itk_folder, sprintf('%s_data.mat', itk_fn));
save(result.fp, 'result');
%% Load annotation result and convert to graph 
data_fp = fullfile(itk_folder, 'whole_20250106_114041_data.mat');
result = DataManager.load_data(data_fp);
itk_f_str = fullfile(itk_folder, 'whole_20250105_114153_mask.nii.gz');
vol_label = DataManager.load_data(itk_f_str);
vsl_im = result.stitched_im;

min_cc_size = 5 ^ 3;
imc_r = 1;

vol_mask = imfill(vol_label > 0, 'holes');
vol_mask = bwareaopen(vol_mask, min_cc_size);
vol_mask_dt = bwdist(~vol_mask);

mask_art = (vol_label == 1 | vol_label == 2);
mask_art = imfill(mask_art, 'holes');
mask_art = bwareaopen(mask_art, min_cc_size);

mask_vein = (vol_label == 1 | vol_label == 3);
mask_vein = imfill(mask_vein, 'holes');
mask_vein = bwareaopen(mask_vein, min_cc_size);

mask_dt_a = bwdist(~mask_art);
mask_dt_v = bwdist(~mask_vein);

skl_a = bwskel(mask_art);
skl_a = fun_skeleton_recentering_within_mask(skl_a, mask_dt_a, mask_art);
skl_v = bwskel(mask_vein);
skl_v  = fun_skeleton_recentering_within_mask(skl_v, mask_dt_v, mask_vein);
% check consistency 
f = figure();
for i = 1 : 3
    im_a = squeeze(max(skl_a, [], i));
    im_v = squeeze(max(skl_v, [], i));
    if i ~= 3
        im_a = im_a.';
        im_v = im_v.';
    end
    a = subplot(1, 3, i);
    imshowpair(im_a, im_v);
end
%%
% Use distance map for recentering 
skl_m = bwskel(skl_a | skl_v);
vsl_graph = fun_skeleton_to_graph(skl_m);

gr_para.internal_offset = 0;
gr_para.pruning_max_length = 3;
gr_para.max_bilink_loop_length = 10;
gr_para.max_self_loop_length = 15;
continue_Q = true;
vessel_graph = vsl_graph;
% TODO: Add the network reconstruction and re-skeletonization into graph refinement. 
while continue_Q
    [vessel_graph, pruning_info_str_1] = fun_graph_delete_hairs_and_short_loops(...
        vessel_graph, vsl_im, gr_para);
    
    if pruning_info_str_1.num_bilink_loop > 0 || pruning_info_str_1.num_self_loop > 0 || ...
            pruning_info_str_1.num_short_link_ep1 > 0
        recon_ind = cat(1, vessel_graph.node.cc_ind{:}, vessel_graph.link.cc_ind{:});
        recon_skl_mask = false(vessel_graph.num.mask_size);
        recon_skl_mask(recon_ind) = true;
        recon_skl_mask_rs = bwskel(recon_skl_mask);
        if any(~recon_skl_mask_rs(:) & recon_skl_mask(:))
            vessel_graph = fun_skeleton_to_graph(recon_skl_mask_rs);
            fprintf("Need an additional round of graph refinement."); 
            disp(pruning_info_str_1);
            continue_Q = true;
        else
            continue_Q = false;
        end
    else
        continue_Q = false;
    end
end

mask_dt = max(mask_dt_a, mask_dt_v);
vessel_graph = fun_graph_add_radius(vessel_graph, mask_dt, 1);
%%
recon_mask = fun_graph_reconstruction(vessel_graph, 1);
recon_ind = cat(1, vessel_graph.node.cc_ind{:}, vessel_graph.link.cc_ind{:});
recon_label = vol_label(recon_ind);
recon_skl = zeros(vessel_graph.num.mask_size, 'uint8');
recon_skl(recon_ind) = recon_label;
recon_skl_mip = max(recon_skl, [], 3);
% Compare final skeleton with the initial skeleton 
f = figure;
a = axes(f);
imshowpair(recon_skl_mip > 0,  max(skl_m, [], 3));
a.DataAspectRatio = [1,1,1];

link_label = cellfun(@(x) mode(vol_label(x)), vessel_graph.link.cc_ind);
link_vxl_label = repelem(link_label, vessel_graph.link.num_voxel_per_cc);
node_label = cellfun(@(x) mode(vol_label(x)), vessel_graph.node.cc_ind);
node_vxl_label = repelem(node_label, vessel_graph.node.num_voxel_per_cc);
recon_r = mask_dt(recon_ind);
recon_label = cat(1, node_vxl_label, link_vxl_label);
recon_label_array = fun_skeleton_reconstruction_label(recon_ind, recon_r, ...
    recon_label, vessel_graph.num.mask_size);

itk_f_name = 'whole';
recon_itk_folder = strrep(itk_folder, segmentation_version, 'recon');
recon_itk_f_str = fullfile(recon_itk_folder, sprintf('%s_%s', itk_f_name, datestr(now, 'YYYYmmDD_HHMMSS')));
DataManager.visualize_itksnap(vsl_im, recon_label_array, recon_itk_f_str, true);
%% cc label map 
link_vxl_label = repelem(1 : vessel_graph.link.num_cc, vessel_graph.link.num_voxel_per_cc).';
node_vxl_label = -repelem(1 : vessel_graph.node.num_cc, vessel_graph.node.num_voxel_per_cc).';
cc_recon_label = cat(1, node_vxl_label, link_vxl_label);
recon_cc_label_array = fun_skeleton_reconstruction_label(recon_ind, recon_r, ...
    cc_recon_label, vessel_graph.num.mask_size);
%% 
data = result;
data.mask_size = vessel_graph.num.mask_size;
data.label_array_annotated = vol_label;
data.skl_sub = fun_ind2sub(data.mask_size, recon_ind);
data.skl_label = recon_label;
data.skl_r_pxl = recon_r;
data.label_array_recon = recon_label_array;
data.cc_label_array = recon_cc_label_array;
data.fp = sprintf('%s_data.mat', recon_itk_f_str);
save(data.fp, '-struct', 'data', '-v7.3');
%%
vessel_graph.link.map_ind_2_label(sub2ind(vessel_graph.num.mask_size, 67, 144, 52))

