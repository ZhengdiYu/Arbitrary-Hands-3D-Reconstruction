import torch
import torch.nn as nn
import numpy as np
from acr.config import args
from acr.utils import rot6D_to_angular

class ResultParser(nn.Module):
    def __init__(self):
        super(ResultParser,self).__init__()
        self.map_size = args().centermap_size
        self.part_name = ['cam', 'global_orient', 'hand_pose', 'betas']
        self.part_idx = [args().cam_dim, args().rot_dim,  (args().mano_theta_num-1)*args().rot_dim,       10] # [cam_dim:3, rot_dim:3/6, pose:15*3/6, beta:10]
        self.kps_num = 21 # + 21*2
        self.params_num = np.array(self.part_idx).sum()

        self.centermap_parser = CenterMap()
        if args().prior_mode == 'merge':
            self.fusion_fc_end = nn.Linear(106*4,106*2)

    @torch.no_grad()
    def parse(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)

        if 'params_pred' in outputs:
            idx_list, params_dict = [0], {}
            for i,  (idx, name) in enumerate(zip(self.part_idx,self.part_name)):
                idx_list.append(idx_list[i] + idx)
                params_dict[name] = outputs['params_pred'][:, idx_list[i]: idx_list[i+1]].contiguous()

            if args().Rot_type=='6D':
                params_dict['hand_pose'] = rot6D_to_angular(params_dict['hand_pose'])
                params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])

            params_dict['poses'] = torch.cat([params_dict['global_orient'], params_dict['hand_pose']], 1) # 3 + 45
            L, R = outputs['left_hand_num'], outputs['right_hand_num']
            outputs['output_hand_type'] = torch.cat((torch.zeros(L), torch.ones(R))).cuda().to(torch.int32)

        outputs['params_dict'] = params_dict

        return outputs, meta_data

    def determine_coeff(self, l_centers, r_centers, l_priors, r_priors):
        # only for testing stage.
        diff = torch.sqrt((l_centers[0,0]-r_centers[0,0])**2+(l_centers[0,1]-r_centers[0,1])**2)
        if diff > 32:
            l_priors, r_priors = 0,0
        return l_priors, r_priors

    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        #device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
            results = maps[batch_ids,flat_inds].contiguous()
        else:
            results = maps[batch_ids,:,flat_inds].contiguous()
        return results

    def reorganize_gts(self, meta_data, key_list, batch_ids):
        for key in key_list:
            if key in meta_data:
                if isinstance(meta_data[key], torch.Tensor):
                    meta_data[key] = meta_data[key][batch_ids]
                elif isinstance(meta_data[key], list):
                    meta_data[key] = np.array(meta_data[key])[batch_ids.cpu().numpy()]
        return meta_data

    def reorganize_data(self, outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids):
        exclude_keys += gt_keys
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = []
        for key, item in meta_data.items():
            if key not in exclude_keys:
                info_vis.append(key)
        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        for gt_key in gt_keys:
            if gt_key in meta_data:
                try:
                    meta_data[gt_key] = meta_data[gt_key][batch_ids,person_ids]
                except Exception as error:
                    raise ValueError
        return outputs,meta_data

    @torch.no_grad()
    def parse_maps(self,outputs, meta_data, cfg):

        #######################
        # process centers
        #######################
        l_center_preds_info = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['l_center_map'])
        l_batch_ids, l_flat_inds, l_cyxs, l_top_score = l_center_preds_info
        r_center_preds_info = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['r_center_map'])
        r_batch_ids, r_flat_inds, r_cyxs, r_top_score = r_center_preds_info

        batch_ids = torch.cat((l_batch_ids, r_batch_ids)) # initial judge
        detection_flag = []

        #######################
        # parameter sampling
        #######################
        if 'params_pred' not in outputs and 'l_params_maps' in outputs and 'r_params_maps' in outputs:
            if len(l_batch_ids) != 0:
                for i in range(len(l_batch_ids)):
                    detection_flag.append(True)
                outputs['l_params_pred'] = self.parameter_sampling(outputs['l_params_maps'], l_batch_ids, l_flat_inds, use_transform=True)
            else:
                detection_flag.append(False)
                l_batch_ids = torch.tensor([0]).cuda()
                l_flat_inds = torch.tensor([0]).cuda()
                outputs['l_params_pred'] = self.parameter_sampling(outputs['l_params_maps'], torch.tensor([0]), torch.tensor([0]), use_transform=True)

            if len(r_batch_ids) != 0:
                for i in range(len(r_batch_ids)):
                    detection_flag.append(True)
                outputs['r_params_pred'] = self.parameter_sampling(outputs['r_params_maps'], r_batch_ids, r_flat_inds, use_transform=True)
            else:
                detection_flag.append(False)
                r_batch_ids = torch.tensor([0]).cuda()
                r_flat_inds = torch.tensor([0]).cuda()
                outputs['r_params_pred'] = self.parameter_sampling(outputs['r_params_maps'], torch.tensor([0]), torch.tensor([0]), use_transform=True)

        #######################
        # process prior map
        #######################
        if args().inter_prior and args().dataset != 'FreiHand':
            l_cat_r, counts = torch.cat((l_batch_ids, r_batch_ids)).unique(return_counts=True)
            all_hand_valid_batch_ids = l_cat_r[torch.where(counts.gt(1))]
            if args().dataset == 'FreiHand':
                raise ValueError

            if len(all_hand_valid_batch_ids) > 0 and sum(detection_flag) == len(detection_flag):
                l_corr_idx = torch.nonzero(l_batch_ids[..., None] == all_hand_valid_batch_ids)
                r_corr_idx = torch.nonzero(r_batch_ids[..., None] == all_hand_valid_batch_ids)
                assert (l_corr_idx[:, 1] == r_corr_idx[:, 1]).all()
                l_corr_idx = l_corr_idx[:, 0]
                r_corr_idx = r_corr_idx[:, 0]
                # assert (all_hand_valid_batch_ids == l_batch_ids[l_corr_idx]).all()
                # assert (all_hand_valid_batch_ids == r_batch_ids[r_corr_idx]).all()

                if args().prior_mode == 'cross':
                    l_prior = self.parameter_sampling(outputs['l_prior_maps'], all_hand_valid_batch_ids, r_flat_inds[r_corr_idx], use_transform=True) # [N, 106]
                    r_prior = self.parameter_sampling(outputs['r_prior_maps'], all_hand_valid_batch_ids, l_flat_inds[l_corr_idx], use_transform=True) # [N, 106]
                    l_prior, r_prior = self.determine_coeff(l_cyxs, r_cyxs, l_prior, r_prior)
                    outputs['l_params_pred'][l_corr_idx, 3:] += l_prior
                    outputs['r_params_pred'][r_corr_idx, 3:] += r_prior

                elif args().prior_mode == 'merge':
                    l_prior = self.parameter_sampling(outputs['l_prior_maps'], all_hand_valid_batch_ids, r_flat_inds[r_corr_idx], use_transform=True) # [N, 106]
                    r_prior = self.parameter_sampling(outputs['r_prior_maps'], all_hand_valid_batch_ids, l_flat_inds[l_corr_idx], use_transform=True) # [N, 106]
                    x = torch.cat((outputs['l_params_pred'][l_corr_idx, 3:], l_prior, outputs['r_params_pred'][r_corr_idx, 3:], r_prior), dim=1)

                    merged_outputs = self.fusion_fc_end(x).float()
                    outputs['l_params_pred'][l_corr_idx, 3:] = merged_outputs[:, 0:106]
                    outputs['r_params_pred'][r_corr_idx, 3:] = merged_outputs[:, 106:212]

                else:
                    raise ValueError

        elif not args().inter_prior and args().prior_mode == 'none':
            print(' no prior!!!!!!!!!!!!!')
        elif args().inter_prior and args().dataset == 'FreiHand':
            raise ValueError
        else:
            raise ValueError

        # again
        outputs['detection_flag'] = torch.Tensor(detection_flag).cuda()
        outputs['params_pred'] = torch.cat((outputs['l_params_pred'], outputs['r_params_pred']))
        batch_ids = torch.cat((l_batch_ids, r_batch_ids))
        flat_inds = torch.cat((l_flat_inds, r_flat_inds))

        if 'centers_pred' not in outputs and len(batch_ids) != 0:
            outputs['l_centers_pred'] = torch.stack([l_flat_inds%args().centermap_size, torch.div(l_flat_inds, args().centermap_size, rounding_mode='floor')], 1)
            outputs['r_centers_pred'] = torch.stack([r_flat_inds%args().centermap_size, torch.div(r_flat_inds, args().centermap_size, rounding_mode='floor')], 1)
            outputs['l_centers_conf'] = self.parameter_sampling(outputs['l_center_map'], l_batch_ids, l_flat_inds, use_transform=True)
            outputs['r_centers_conf'] = self.parameter_sampling(outputs['r_center_map'], r_batch_ids, r_flat_inds, use_transform=True)

        outputs['left_hand_num'] = torch.tensor([len(outputs['l_params_pred'])]).cuda()
        outputs['right_hand_num'] = torch.tensor([len(outputs['r_params_pred'])]).cuda()

        #######################
        # fuse maps and reorgnize
        #######################
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]

        info_vis = ['image', 'offsets','imgpath']
        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        outputs['detection_flag_cache'] = outputs['detection_flag'].bool()

        return outputs,meta_data

#######################
# CenterMap
#######################
class CenterMap(object):
    def __init__(self,style='heatmap_adaptive_scale'):
        self.style=style
        self.size = args().centermap_size
        self.max_hand = args().max_hand
        self.shrink_scale = float(args().input_size//self.size)
        self.dims = 1
        self.sigma = 1
        self.conf_thresh= args().centermap_conf_thresh
        print('Confidence:', self.conf_thresh)
        self.gk_group, self.pool_group = self.generate_kernels(args().kernel_sizes)

    def generate_kernels(self, kernel_size_list):
        gk_group, pool_group = {}, {}
        for kernel_size in set(kernel_size_list):
            x = np.arange(0, kernel_size, 1, float)
            y = x[:, np.newaxis]
            x0, y0 = (kernel_size-1)//2,(kernel_size-1)//2
            gaussian_distribution = - ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
            gk_group[kernel_size] = np.exp(gaussian_distribution)
            pool_group[kernel_size] = torch.nn.MaxPool2d(kernel_size, 1, (kernel_size-1)//2)
        return gk_group, pool_group

    def parse_centermap_heatmap_adaptive_scale_batch(self, center_maps, train_flag=False):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[args().kernel_sizes[-1]])
        b, c, h, w = center_map_nms.shape
        if train_flag:
            K = self.max_hand
        else:
            K = 1

        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)

        topk_ys = torch.div(topk_inds, w, rounding_mode='floor').int().float()
        topk_xs = (topk_inds % w).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = torch.div(index, K, rounding_mode='floor').int()
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)

        mask = topk_score>self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_yxs = torch.stack([topk_ys[mask], topk_xs[mask]]).permute((1,0))
        return batch_ids, topk_inds[mask], center_yxs, topk_score[mask]

def nms(det, pool_func=None):
    maxm = pool_func(det)
    maxm = torch.eq(maxm, det).float()
    det = det * maxm
    return det

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap
