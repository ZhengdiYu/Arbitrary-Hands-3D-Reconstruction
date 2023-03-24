import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg')
import math

from acr.config import args
from mano.manolayer import ManoLayer
from acr.utils import process_idx
from collections import OrderedDict

default_cfg = {'save_dir':None, 'vids':None, 'settings':[]} # 'put_org'

class Visualizer(object):
    def __init__(self, resolution=(2048, 2048), renderer_type=None):

        # constants
        self.resolution = resolution
        self.MANO_SKELETON = load_skeleton('mano/skeleton.txt', 21)
        self.MANO_RGB_DICT = get_keypoint_rgb(self.MANO_SKELETON)
        self.mano2interhand_mapper = np.array([4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9,  16, 15, 14, 13, 20, 19, 18, 17, 0])

        if renderer_type is not None: 
            if renderer_type == 'pyrender':
                from acr.renderer.renderer_pyrd import get_renderer
                self.renderer = get_renderer(resolution=self.resolution, perps=True)
            elif renderer_type == 'pytorch3d':
                from acr.renderer.renderer_pt3d import get_renderer
                self.renderer = get_renderer(resolution=self.resolution, perps=True)
            else:
                raise NotImplementedError

        self.vis_size = resolution
        self.skeleton_3D_ploter = Plotter3dPoses()
        self.mano_layer=torch.nn.ModuleDict({
            '1':ManoLayer(
                    ncomps=45,
                    center_idx=args().align_idx if args().mano_mesh_root_align else None,#9, # TODO: 1. wrist align? root align ? 0 or 9?
                    side='right',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                ),
            '0':ManoLayer(
                    ncomps=45,
                    center_idx=args().align_idx if args().mano_mesh_root_align else None,#9, # TODO: 1. wrist align? root align ? 0 or 9?
                    side='left',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                ),
            
            'right':ManoLayer(
                    ncomps=45,
                    center_idx=args().align_idx if args().mano_mesh_root_align else None,#9, # TODO: 1. wrist align? root align ? 0 or 9?
                    side='right',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                ),
            'left':ManoLayer(
                    ncomps=45,
                    center_idx=args().align_idx if args().mano_mesh_root_align else None,#9, # TODO: 1. wrist align? root align ? 0 or 9?
                    side='left',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                ),
        })

    def visualize_renderer_verts_list(self, verts_list, j3d_list, hand_type=None, hand_type_list=None, faces_list=None, images=None, cam_params=None,\
                                            colors=None, trans=None, thresh=0., pre_colors=np.array([[0.46, 0.59, 0.64], [0.94, 0.71, 0.53]])):#pre_colors=np.array([[.8, .0, .0], [0, .0, .8]])):
        verts_list = [verts.contiguous() for verts in verts_list]

        if faces_list is None and hand_type is not None:
            faces_list = [self.mano_layer[str(hand_type[0])].th_faces.repeat(len(verts), 1, 1).to(verts.device) for verts in verts_list]

        if hand_type is not None and hand_type_list is None:

            rendered_imgs = []
            for ind, (verts, faces) in enumerate(zip(verts_list, faces_list)):
                if trans is not None:
                    verts += trans[ind].unsqueeze(1)

                if isinstance(colors, list):
                    color = colors[ind]  
                else:
                    color = np.array([pre_colors[x] for x in hand_type])

                rendered_img = self.renderer(verts, faces, colors=color, focal_length=args().focal_length, cam_params=cam_params)
                rendered_imgs.append(rendered_img)
            if len(rendered_imgs)>0:
                if isinstance(rendered_imgs[0],torch.Tensor):
                    rendered_imgs = torch.cat(rendered_imgs, 0).cpu().numpy()

            rendered_imgs = np.array(rendered_imgs)
            if rendered_imgs.shape[-1]==4:
                transparent = rendered_imgs[:,:, :, -1]
                rendered_imgs = rendered_imgs[:,:,:,:-1]

        elif hand_type is None and hand_type_list is not None:

            rendered_imgs = []
            for ind, (verts, j3d, hand_type) in enumerate(zip(verts_list, j3d_list, hand_type_list)):
                if trans is not None:
                    j3d += trans[ind].unsqueeze(1)
                    verts += trans[ind].unsqueeze(1)

                if isinstance(colors, list):
                    color = colors[ind]  
                else:
                    color = np.array([pre_colors[x] for x in hand_type])

                all_face = []
                for single_hand_type in hand_type:
                    all_face.append(self.mano_layer[str(int(single_hand_type))].th_faces)
                all_face = torch.stack(all_face, dim=0)
                '''
                if args().vis_otherview:
                    theta = -60
                    theta = 3.14159 / 180 * theta
                    R = [[math.cos(theta), 0, math.sin(theta)],
                        [0, 1, 0],
                        [-math.sin(theta), 0, math.cos(theta)]]
                    R = torch.tensor(R).float().cuda()
                    
                    ori_root = j3d[0, 9].clone()
                    j3d = torch.matmul(j3d, R)
                    new_root = j3d[0, 9]
                    trans_root = new_root - ori_root
                    trans_root = trans_root.unsqueeze(0).unsqueeze(0)

                    verts = torch.matmul(verts, R)
                    
                    print(verts.size(), trans_root.size())
                    verts = verts - trans_root
                    print(f'rending other view {theta}/{trans_root}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11')
                '''
                rendered_img = self.renderer(verts, all_face, colors=color, focal_length=args().focal_length, cam_params=cam_params)
                rendered_imgs.append(rendered_img)

            if len(rendered_imgs)>0:
                if isinstance(rendered_imgs[0],torch.Tensor):
                    rendered_imgs = torch.cat(rendered_imgs, 0).cpu().numpy()

            rendered_imgs = np.array(rendered_imgs)
            if rendered_imgs.shape[-1]==4:
                transparent = rendered_imgs[:,:, :, -1]
                rendered_imgs = rendered_imgs[:,:,:,:-1]
        else:
            raise NotImplemented

        visible_weight = 0.9
        if images is not None:
            if not (images.shape == rendered_imgs.shape):
                scale = rendered_imgs.shape[2] / images.shape[2]
                assert (scale == 4) or (scale == 1/4)
                images = F.interpolate(torch.from_numpy(images).permute(0, 3, 1, 2).float(),scale_factor=(scale,scale),mode='bilinear').permute(0, 2, 3, 1).numpy()

            valid_mask = (transparent > thresh)[:,:, :,np.newaxis]
            rendered_imgs = rendered_imgs * valid_mask * visible_weight + images * valid_mask * (1-visible_weight) + (1 - valid_mask) * images
        return rendered_imgs.astype(np.uint8)

    def draw_skeleton(self, image, pts, **kwargs):
        return draw_skeleton(image, pts, **kwargs)

    def draw_skeleton_multiperson(self, image, pts, **kwargs):
        return draw_skeleton_multiperson(image, pts, **kwargs)

    def visulize_result_live(self, outputs, frame_img, meta_data, show_items=['mesh'], vis_cfg=default_cfg, **kwargs):
        vis_cfg = dict(default_cfg, **vis_cfg)
        used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'], vids=vis_cfg['vids'])

        img_inds_org = [inds[0] for inds in per_img_inds]
        img_names = np.array(meta_data['imgpath'])[img_inds_org]
        org_imgs = meta_data['image'][outputs['detection_flag_cache']].cpu().numpy().astype(np.uint8)[img_inds_org]

        plot_dict = OrderedDict()
        for vis_name in show_items:
            if vis_name == 'org_img':
                plot_dict['org_img'] = {'figs':org_imgs, 'type':'image'}

            if vis_name == 'mesh' and outputs['detection_flag']:
                per_img_verts_list = [outputs['verts'][outputs['detection_flag_cache']][inds].detach() for inds in per_img_inds]
                per_img_j3d_list = [outputs['j3d'][outputs['detection_flag_cache']][inds].detach() for inds in per_img_inds]
                mesh_trans = [outputs['cam_trans'][outputs['detection_flag_cache']][inds].detach() for inds in per_img_inds]
                hand_type = [outputs['output_hand_type'][outputs['detection_flag_cache']][inds].detach() for inds in per_img_inds]

                rendered_imgs = self.visualize_renderer_verts_list(per_img_verts_list, per_img_j3d_list, hand_type_list=hand_type, images=org_imgs.copy(), trans=mesh_trans)
                #rendered_imgs = self.visualize_renderer_verts_list(per_img_verts_list, per_img_j3d_list, hand_type_list=hand_type, trans=mesh_trans)

                if 'put_org' in vis_cfg['settings']:
                    offsets = meta_data['offsets'].cpu().numpy().astype(np.int)[img_inds_org]
                    img_pad_size, crop_trbl, pad_trbl = offsets[:,:2], offsets[:,2:6], offsets[:,6:10]
                    rendering_onorg_images = []
                    for inds, j in enumerate(used_org_inds):
                        org_imge = frame_img

                        # 1
                        (ih, iw), (ph,pw) = org_imge.shape[:2], img_pad_size[inds]
                        if args().render_size > 1000:
                            ih, iw, ph, pw = ih*4, iw*4, ph*4, pw*4
                            org_imge = F.interpolate(torch.from_numpy(org_imge).unsqueeze(0).permute(0, 3, 1, 2).float(),scale_factor=(4,4),mode='bilinear').permute(0, 2, 3, 1).numpy()[0]
                            #org_imge = cv2.resize(org_imge, (iw, ih))

                        # 2
                        resized_images = cv2.resize(rendered_imgs[inds], (ph+1, pw+1), interpolation = cv2.INTER_CUBIC)
                        (ct, cr, cb, cl), (pt, pr, pb, pl) = crop_trbl[inds], pad_trbl[inds]

                        # 3
                        if args().render_size > 1000:
                            ct, cr, cb, cl, pt, pr, pb, pl = ct*4, cr*4, cb*4, cl*4, pt*4, pr*4, pb*4, pl*4
                        org_imge[ct:ih-cb, cl:iw-cr] = resized_images[pt:ph-pb, pl:pw-pr]
                        rendering_onorg_images.append(org_imge)

                    plot_dict['mesh_rendering_orgimgs'] = {'figs':rendering_onorg_images, 'type':'image'}

            if vis_name == 'j3d' and outputs['detection_flag']:
                real_aligned, pred_aligned, pos3d_vis_mask, joint3d_bones = kwargs['kp3ds']
                real_3ds = (real_aligned*pos3d_vis_mask.unsqueeze(-1)).cpu().numpy()
                predicts = (pred_aligned*pos3d_vis_mask.unsqueeze(-1)).detach().cpu().numpy()
                skeleton_3ds = []
                for inds in per_img_inds:
                    for real_pose_3d, pred_pose_3d in zip(real_3ds[inds], predicts[inds]):
                        skeleton_3d = self.skeleton_3D_ploter.encircle_plot([real_pose_3d, pred_pose_3d], \
                            joint3d_bones, colors=[(255, 0, 0), (0, 255, 255)])
                        skeleton_3ds.append(skeleton_3d)
                plot_dict['j3d'] = {'figs':np.array(skeleton_3ds), 'type':'skeleton'}

            if vis_name == 'pj2d' and outputs['detection_flag']:
                kp_imgs = []
                for img_id, inds_list in enumerate(per_img_inds):
                    org_img = org_imgs[img_id].copy()
                    #try:
                    for kp2d_vis in outputs['pj2d'][inds_list]:
                        if len(kp2d_vis)>0:
                            kp2d_vis = ((kp2d_vis+1)/2 * org_imgs.shape[1])
                            org_img = self.vis_keypoints(org_img, kp2d_vis.detach().cpu().numpy(), skeleton=self.MANO_SKELETON)
                    kp_imgs.append(np.array(org_img))
                plot_dict['pj2d'] = {'figs':kp_imgs, 'type':'image'}

            if vis_name == 'centermap' and outputs['detection_flag']:
                l_centermaps_list = []
                r_centermaps_list = []
                for img_id, (l_centermap, r_centermap) in enumerate(zip(outputs['l_center_map'][used_org_inds], outputs['r_center_map'][used_org_inds])):
                    img_bk = cv2.resize(org_imgs[img_id].copy(),org_imgs.shape[1:3])
                    l_centermaps_list.append(make_heatmaps(img_bk, l_centermap))
                    r_centermaps_list.append(make_heatmaps(img_bk, r_centermap))

        return plot_dict, img_names
    
    def vis_keypoints(self, img, kps, skeleton, score_thr=0.4, line_width=3, circle_rad = 3):
        kps = kps[self.mano2interhand_mapper]
        score = np.ones((21), dtype=np.float32)
        rgb_dict = self.MANO_RGB_DICT
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img.astype('uint8'))

        draw = ImageDraw.Draw(img)
        for i in range(len(skeleton)):
            joint_name = skeleton[i]['name']
            pid = skeleton[i]['parent_id']
            parent_joint_name = skeleton[pid]['name']

            kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
            kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

            if score[i] > score_thr and score[pid] > score_thr and pid != -1:
                draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
            if score[i] > score_thr:
                draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
            if score[pid] > score_thr and pid != -1:
                draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
        return img

def make_heatmaps(image, heatmaps):
    heatmaps = torch.nn.functional.interpolate(heatmaps[None],size=image.shape[:2],mode='bilinear')[0]
    heatmaps = heatmaps.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .detach().cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image
    return image_fused.astype(np.uint8) # image_grid


def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = tagmap.add(-min)\
                       .div(max - min + 1e-5)\
                       .mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .detach().cpu().numpy()

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap*0.9 + image_resized*0.1

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid

def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict

def load_skeleton(path, joint_num):

    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            #print(line)
            splitted = line.strip().split(' ')
            #print(splitted)
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            try:
                skeleton[joint_id]['name'] = joint_name
                skeleton[joint_id]['parent_id'] = joint_parent_id
            except:
                break

    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton

def draw_skeleton(image, pts, bones=None, cm=None, label_kp_order=False,r=3):
    for i,pt in enumerate(pts):
        if len(pt)>1:
            if pt[0]>0 and pt[1]>0:
                image = cv2.circle(image,(int(pt[0]), int(pt[1])),r,(255,0,0),-1)
                if label_kp_order and i in bones:
                    img=cv2.putText(image,str(i),(int(pt[0]), int(pt[1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,215,0),1)
    
    if bones is not None:
        if cm is None:
            set_colors = np.array([[255,0,0] for i in range(len(bones))]).astype(np.int)
        else:
            if len(bones)>len(cm):
                cm = np.concatenate([cm for _ in range(len(bones)//len(cm)+1)],0)
            set_colors = cm[:len(bones)].astype(np.int)
        bones = np.concatenate([bones,set_colors],1).tolist()
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            if (pa>0).all() and (pb>0).all():
                xa,ya,xb,yb = int(pa[0]),int(pa[1]),int(pb[0]),int(pb[1])
                image = cv2.line(image,(xa,ya),(xb,yb),(int(line[2]), int(line[3]), int(line[4])),r)
    return image

def draw_skeleton_multiperson(image, pts_group,**kwargs):
    for pts in pts_group:
        image = draw_skeleton(image, pts, **kwargs)
    return image


class Plotter3dPoses:

    def __init__(self, canvas_size=(512,512), origin=(0.5, 0.5), scale=200):
        self.canvas_size = canvas_size
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta, self.phi = 0, np.pi/2 #np.pi/4, -np.pi/6
        axis_length = 200
        axes = [
            np.array([[-axis_length/2, -axis_length/2, 0], [axis_length/2, -axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, -axis_length/2, axis_length]], dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                                  [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                                  [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = np.ones((self.canvas_size[0],self.canvas_size[1],3), dtype=np.uint8) * 255 if img is None else img
        R = self._get_rotation(self.theta, self.phi)
        #self._draw_axes(img, R)
        for vertices, color in zip(pose_3ds,colors):
            self._plot_edges(img, vertices, bones, R, color)
        return img

    def encircle_plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = np.ones((self.canvas_size[0],self.canvas_size[1],3), dtype=np.uint8) * 255 if img is None else img
        #encircle_theta, encircle_phi = [0, np.pi/4, np.pi/2, 3*np.pi/4], [np.pi/2,np.pi/2,np.pi/2,np.pi/2]
        encircle_theta, encircle_phi = [0,0,0, np.pi/4,np.pi/4,np.pi/4, np.pi/2,np.pi/2,np.pi/2], [np.pi/2, 5*np.pi/7, -2*np.pi/7, np.pi/2, 5*np.pi/7, -2*np.pi/7, np.pi/2, 5*np.pi/7, -2*np.pi/7,]
        encircle_origin = np.array([[0.165, 0.165], [0.165, 0.495], [0.165, 0.825],\
                                    [0.495, 0.165], [0.495, 0.495], [0.495, 0.825],\
                                    [0.825, 0.165], [0.825, 0.495], [0.825, 0.825]], dtype=np.float32) * np.array(self.canvas_size)[None]
        for self.theta, self.phi, self.origin in zip(encircle_theta, encircle_phi, encircle_origin):
            R = self._get_rotation(self.theta, self.phi)
            #self._draw_axes(img, R)
            for vertices, color in zip(pose_3ds,colors):
                self._plot_edges(img, vertices*0.6, bones, R, color)
        return img

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R, color):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin
        org_verts = vertices.reshape((-1, 3))[edges]
        for inds, edge_vertices in enumerate(edges_vertices):
            if 0 in org_verts[inds]:
                continue
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), color, 2, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [ cos(theta),  sin(theta) * sin(phi)],
            [-sin(theta),  cos(theta) * sin(phi)],
            [ 0,                       -cos(phi)]
        ], dtype=np.float32)  # transposed
