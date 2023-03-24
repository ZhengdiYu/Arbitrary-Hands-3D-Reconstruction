#encoding=utf-8
import h5py
import torch
import numpy as np
import json
import torch.nn.functional as F
import cv2
import shutil
import pickle
import yaml
import platform
import os
import glob
import logging
from io import BytesIO
from acr.config import args
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
import imgaug.augmenters as iaa
from imgaug.augmenters import compute_paddings_to_reach_aspect_ratio
from threading import Thread
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


TAG_CHAR = np.array([202021.25], np.float32)
img_exts = ['.bmp', '.dib', '.jpg', '.jpeg', '.jpe', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.tiff', '.tif', '.sr', '.ras', '.exr', '.hdr', '.pic',\
            '.PNG', '.JPG', '.JPEG']

def get_kp2d_on_org_img(kp2d, offset):
    assert kp2d.shape[1]>=2, print('Espected shape of kp2d is Kx2, while get {}'.formt(kp2d.shape))
    if torch.is_tensor(offset):
        offset = offset.detach().cpu().numpy()
    pad_size_h,pad_size_w, lt_h,rb_h,lt_w,rb_w,offset_h,size_h,offset_w,size_w,length = offset
    kp2d_onorg = np.ones_like(kp2d)
    kp2d_onorg[:,0] = (kp2d[:,0]+1)/2 * pad_size_w - offset_w+lt_w
    kp2d_onorg[:,1] = (kp2d[:,1]+1)/2 * pad_size_h - offset_h+lt_w
    return kp2d_onorg
    

class AverageMeter_Dict(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.dict_store = {}
        self.count = 0

    def update(self, val, n=1):
        for key,value in val.items():
            if key not in self.dict_store:
                self.dict_store[key] = []
            if torch.is_tensor(value):
                value = value.item()
            self.dict_store[key].append(value)
        self.count += n
    
    def sum(self):
        dict_sum = {}
        for k, v in self.dict_store.items():
            dict_sum[k] = round(float(sum(v)),2)
        return dict_sum

    def avg(self):
        dict_sum = self.sum()
        dict_avg = {}
        for k,v in dict_sum.items():
            dict_avg[k] = round(v/self.count,2)
        return dict_avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize_kps(kps,image_shape,resize=512,set_minus=True):
    kps[:,0] *= 1.0 * resize / image_shape[0]
    kps[:,1] *= 1.0 * resize / image_shape[1]
    kps[:,:2] = 2.0 * kps[:,:2] / resize - 1.0

    if kps.shape[1]>2 and set_minus:
        kps[kps[:,2]<0.1,:2] = -2.
    kps=kps[:,:2]
    return kps

#______________IO__________________


def collect_image_list(image_folder=None, collect_subdirs=False):
    
    def collect_image_from_subfolders(image_folder, file_list, collect_subdirs, img_exts):
        for path in glob.glob(os.path.join(image_folder,'*')):
            if os.path.isdir(path) and collect_subdirs:
                collect_image_from_subfolders(path, file_list, collect_subdirs, img_exts)
            elif os.path.splitext(path)[1] in img_exts:
                file_list.append(path)
        return file_list

    file_list = collect_image_from_subfolders(image_folder, [], collect_subdirs, img_exts)

    return file_list

def save_results(image_folder, output_dir, results_dict):
    model_name = args().model_path.split('/')[-1]
    path_name = image_folder.split('/')[-1]
    with open(output_dir + f'/{path_name}_hand{model_name}_{args().centermap_conf_thresh}.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    return

def save_result_dict_tonpz(results, test_save_dir, side):
    for img_path, result_dict in results.items():
        if platform.system() == 'Windows':
            path_list = img_path.split('\\')
        else:
            path_list = img_path.split('/')
        file_name = '_'.join(path_list)
        file_name = '_'.join(os.path.splitext(file_name)).replace('.','') + f'_{side}.npz'
        save_path = os.path.join(test_save_dir, file_name)
        # get the results: np.load('/path/to/person_overlap.npz',allow_pickle=True)['results'][()]
        np.savez(save_path, results=result_dict)


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def save_pkl(info,name='../data/info.pkl'):
    check_file_and_remake(name.replace(os.path.basename(name),''))
    if name[-4:] !='.pkl':
        name += '.pkl'
    with open(name,'wb') as outfile:
        pickle.dump(info, outfile, pickle.HIGHEST_PROTOCOL)

def save_yaml(dict_file,path):
    with open(path, 'w') as file:
        documents = yaml.dump(dict_file, file)

def read_pkl(name = '../data/info.pkl'):
    with open(name,'rb') as f:
        return pickle.load(f)

def read_pkl_coding(name = '../data/info.pkl'):
    with open(name, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

def check_file_and_remake(path,remove=False):
    if remove:
        if os.path.isdir(path):
            shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)

def save_h5(info,name):
    check_file_and_remake(name.replace(os.path.basename(name),''))
    if name[-3:] !='.h5':
        name += '.h5'
    f=h5py.File(name,'w')
    for item, value in info.items():
        f[item] = value
    f.close()

def read_h5(name):
    if name[-3:] !='.h5':
        name += '.h5'
    f=h5py.File(name,'r')
    info = {}
    for item, value in f.items():
        info[item] = np.array(value)
    f.close()
    return info

def save_obj(verts, faces, color=None, obj_mesh_name='mesh.obj'):
    #print('Saving:',obj_mesh_name)
    with open(obj_mesh_name, 'w') as fp:
        for v in verts:
            fp.write( 'v %f %f %f %f %f %f\n' % ( v[0], v[1], v[2], color[0], color[1], color[2]))

        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

def save_json(dicts, name):
    json_str = json.dumps(dicts)
    with open(name, 'w') as json_file:
        json_file.write(json_str)


#______________model tools__________________
def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)

#______________interesting tools__________________

def shrink(leftTop, rightBottom, width, height):
    xl = -leftTop[0]
    xr = rightBottom[0] - width

    yt = -leftTop[1]
    yb = rightBottom[1] - height

    cx = (leftTop[0] + rightBottom[0]) / 2
    cy = (leftTop[1] + rightBottom[1]) / 2

    r = (rightBottom[0] - leftTop[0]) / 2

    sx = max(xl, 0) + max(xr, 0)
    sy = max(yt, 0) + max(yb, 0)

    if (xl <= 0 and xr <= 0) or (yt <= 0 and yb <=0):
        return leftTop, rightBottom
    elif leftTop[0] >= 0 and leftTop[1] >= 0 : # left top corner is in box
        l = min(yb, xr)
        r = r - l / 2
        cx = cx - l / 2
        cy = cy - l / 2
    elif rightBottom[0] <= width and rightBottom[1] <= height : # right bottom corner is in box
        l = min(yt, xl)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy + l / 2
    elif leftTop[0] >= 0 and rightBottom[1] <= height : #left bottom corner is in box
        l = min(xr, yt)
        r = r - l  / 2
        cx = cx - l / 2
        cy = cy + l / 2
    elif rightBottom[0] <= width and leftTop[1] >= 0 : #right top corner is in box
        l = min(xl, yb)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy - l / 2
    elif xl < 0 or xr < 0 or yb < 0 or yt < 0:
        return leftTop, rightBottom
    elif sx >= sy:
        sx = max(xl, 0) + max(0, xr)
        sy = max(yt, 0) + max(0, yb)
        # cy = height / 2
        if yt >= 0 and yb >= 0:
            cy = height / 2
        elif yt >= 0:
            cy = cy + sy / 2
        else:
            cy = cy - sy / 2
        r = r - sy / 2

        if xl >= sy / 2 and xr >= sy / 2:
            pass
        elif xl < sy / 2:
            cx = cx - (sy / 2 - xl)
        else:
            cx = cx + (sy / 2 - xr)
    elif sx < sy:
        cx = width / 2
        r = r - sx / 2
        if yt >= sx / 2 and yb >= sx / 2:
            pass
        elif yt < sx / 2:
            cy = cy - (sx / 2 - yt)
        else:
            cy = cy + (sx / 2 - yb)


    return [cx - r, cy - r], [cx + r, cy + r]

def calc_aabb_batch(ptSets_batch):
    batch_size = ptSets_batch.shape[0]
    ptLeftTop     = np.array([np.min(ptSets_batch[:,:,0],axis=1),np.min(ptSets_batch[:,:,1],axis=1)]).T
    ptRightBottom = np.array([np.max(ptSets_batch[:,:,0],axis=1),np.max(ptSets_batch[:,:,1],axis=1)]).T
    bbox = np.concatenate((ptLeftTop.reshape(batch_size,1,2),ptRightBottom.reshape(batch_size,1,2)),axis=1)
    return bbox
'''
    calculate a obb for a set of points
    inputs:
        ptSets: a set of points
    return the center and 4 corners of a obb
'''
def calc_obb(ptSets):
    ca = np.cov(ptSets,y = None,rowvar = 0,bias = 1)
    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    ar = np.dot(ptSets,np.linalg.inv(tvect))
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff    = (maxa - mina)*0.5
    center  = mina + diff
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]]])
    corners = np.dot(corners, tvect)
    return corners[0], corners[1], corners[2], corners[3]




#__________________transform tools_______________________
def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        hom_mat = torch.tensor([0, 0, 1]).float()
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        batch_size, device = rot_mat.shape[0], rot_mat.device
        hom_mat = hom_mat.view(1, 3, 1)
        hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
        hom_mat = hom_mat.to(device)
        rotation_matrix = torch.cat([rot_mat, hom_mat], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa

def rot6d_to_rotmat(x):
    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats

def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
    return pose

def batch_orth_proj(X, camera, mode='2d',keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed

def convert_kp2d_from_input_to_orgimg(kp2ds, offsets):
    offsets = offsets.float().to(kp2ds.device)
    img_pad_size, crop_trbl, pad_trbl = offsets[:,:2], offsets[:,2:6], offsets[:,6:10]
    leftTop = torch.stack([crop_trbl[:,3]-pad_trbl[:,3], crop_trbl[:,0]-pad_trbl[:,0]],1)
    kp2ds_on_orgimg = (kp2ds + 1) * img_pad_size.unsqueeze(1) / 2 + leftTop.unsqueeze(1)
    return kp2ds_on_orgimg

def vertices_kp3d_projection(outputs, params_dict, meta_data=None, presp=args().model_version>3):
    params_dict, vertices, j3ds = params_dict, outputs['verts'], outputs['j3d']
    verts_camed = batch_orth_proj(vertices, params_dict['cam'], mode='3d',keep_dim=True)
    pj3d = batch_orth_proj(j3ds, params_dict['cam'], mode='2d')
    predicts_j3ds = j3ds[:,:24].contiguous().detach().cpu().numpy()
    predicts_pj2ds = (pj3d[:,:,:2][:,:24].detach().cpu().numpy()+1)*256

    cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, \
                                focal_length=args().focal_length, img_size=np.array([512,512])).to(vertices.device)
    projected_outputs = {'verts_camed': verts_camed, 'pj2d': pj3d[:,:,:2], 'cam_trans':cam_trans}

    if meta_data is not None:
        projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d'], meta_data['offsets'])
    return projected_outputs

def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None, cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0,0], camK[1,1] = focal_length, focal_length
        camK[:2,2] = img_size//2
    else:
        camK = proj_mat
    ret, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                              flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)

    if inliers is None:
        return INVALID_TRANS
    else:
        tra_pred = tvec[:,0]            
        return tra_pred

def estimate_translation_np(joints_3d, joints_2d, joints_conf, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = joints_3d.shape[0]
    if proj_mat is None:
        # focal length
        f = np.array([focal_length,focal_length])
        # optical center
        center = img_size/2.
    else:
        f = np.array([proj_mat[0,0],proj_mat[1,1]])
        center = proj_mat[:2,2]

    # transformations
    Z = np.reshape(np.tile(joints_3d[:,2],(2,1)).T,-1)
    XY = np.reshape(joints_3d[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans

def estimate_translation(joints_3d, joints_2d, pts_mnum=4,focal_length=600, proj_mats=None, cam_dists=None,img_size=np.array([512.,512.])):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    if torch.is_tensor(joints_3d):
        joints_3d = joints_3d.detach().cpu().numpy()
    if torch.is_tensor(joints_2d):
        joints_2d = joints_2d.detach().cpu().numpy()
    
    if joints_2d.shape[-1]==2:
        joints_conf = joints_2d[:, :, -1]>-2.
    elif joints_2d.shape[-1]==3:
        joints_conf = joints_2d[:, :, -1]>0
    joints3d_conf = joints_3d[:, :, -1]!=-2.
    
    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float)
    if proj_mats is None:
        proj_mats = [None for _ in range(len(joints_2d))]
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        S_i = joints_3d[i]
        joints_i = joints_2d[i,:,:2]
        valid_mask = joints_conf[i]*joints3d_conf[i]
        if valid_mask.sum()<pts_mnum:
            trans[i] = INVALID_TRANS
            continue
        if len(img_size.shape)==1:
            imgsize = img_size
        elif len(img_size.shape)==2:
            imgsize = img_size[i]
        else:
            raise NotImplementedError
        try:
            trans[i] = estimate_translation_cv2(S_i[valid_mask], joints_i[valid_mask], 
                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i], cam_dist=cam_dists[i])
        except:
            trans[i] = estimate_translation_np(S_i[valid_mask], joints_i[valid_mask], valid_mask[valid_mask].astype(np.float32), 
                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i])

    return torch.from_numpy(trans).float()

def transform_rot_representation(rot, input_type='mat',out_type='vec'):
    '''
    make transformation between different representation of 3D rotation
    input_type / out_type (np.array):
        'mat': rotation matrix (3*3)
        'quat': quaternion (4)
        'vec': rotation vector (3)
        'euler': Euler degrees in x,y,z (3)
    '''
    if input_type=='mat':
        r = R.from_matrix(rot)
    elif input_type=='quat':
        r = R.from_quat(rot)
    elif input_type =='vec':
        r = R.from_rotvec(rot)
    elif input_type =='euler':
        if rot.max()<4:
            rot = rot*180/np.pi
        r = R.from_euler('xyz',rot, degrees=True)
    
    if out_type=='mat':
        out = r.as_matrix()
    elif out_type=='quat':
        out = r.as_quat()
    elif out_type =='vec':
        out = r.as_rotvec()
    elif out_type =='euler':
        out = r.as_euler('xyz', degrees=False)
    return out

# def compute_similarity_transform(S1, S2):
#     '''
#     Computes a similarity transform (sR, t) that takes
#     a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
#     where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
#     i.e. solves the orthogonal Procrutes problem.
#     '''
#     transposed = False
#     if S1.shape[0] != 3 and S1.shape[0] != 2:
#         S1 = S1.T
#         S2 = S2.T
#         transposed = True
#     assert(S2.shape[1] == S1.shape[1])

#     # 1. Remove mean.
#     mu1 = S1.mean(axis=1, keepdims=True)
#     mu2 = S2.mean(axis=1, keepdims=True)
#     X1 = S1 - mu1
#     X2 = S2 - mu2

#     # 2. Compute variance of X1 used for scale.
#     var1 = np.sum(X1**2)

#     # 3. The outer product of X1 and X2.
#     K = X1.dot(X2.T)

#     # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
#     # singular vectors of K.
#     U, s, Vh = np.linalg.svd(K)
#     V = Vh.T
#     # Construct Z that fixes the orientation of R to get det(R)=1.
#     Z = np.eye(U.shape[0])
#     Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
#     # Construct R.
#     R = V.dot(Z.dot(U.T))

#     # 5. Recover scale.
#     scale = np.trace(R.dot(K)) / var1

#     # 6. Recover translation.
#     t = mu2 - scale*(R.dot(mu1))

#     # 7. Error:
#     S1_hat = scale*R.dot(S1) + t

#     if transposed:
#         S1_hat = S1_hat.T

#     return S1_hat


def batch_rodrigues(param):
    #param N x 3
    batch_size = param.shape[0]
    #沿第二维（3个数）进行求二次范数：||x||，下面就是进行标准化，每三个数除以他们的范数。
    l1norm = torch.norm(param + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(param, angle)
    angle = angle * 0.5
    #上面算出的是一个向量的长度：sqrt(x**2+y**2+z**2)/2,所以这个长度的的cos
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    #用四元组表示三维旋转，有时间看一下×××××××××
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)

    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    把四元组的系数转化成旋转矩阵。四元组表示三维旋转
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False,root_rot_mat =None):
    '''
    进行成堆的全局刚性变换。
    '''
    N = Rs.shape[0]
    #确定根节点的旋转变换。
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = torch.from_numpy(np_rot_x).float().cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    elif root_rot_mat is not None:
        np_rot_x = np.reshape(np.tile(root_rot_mat, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float().cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1).cuda()], dim = 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    #print('result',results)
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1).cuda()], dim = 2)
    #print('js w ',Js_w0)
    init_bone = torch.matmul(results, Js_w0)
    #print('init_bone before padded',init_bone)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    #print('init_bone padded',init_bone)
    A = results - init_bone
    #print('new_J:',new_J)

    return new_J, A

def batch_global_rigid_transformation_cpu(Rs, Js, parent, rotate_base = False,root_rot_mat =None):
    '''
    进行成堆的全局刚性变换。
    '''
    N = Rs.shape[0]
    #确定根节点的旋转变换。
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    elif root_rot_mat is not None:
        np_rot_x = np.reshape(np.tile(root_rot_mat, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1)], dim = 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1)], dim = 2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A

def batch_lrotmin(param):
    param = param[:,3:].contiguous()
    Rs = batch_rodrigues(param.view(-1, 3))
    e = torch.eye(3).float()
    Rs = Rs.sub(1.0, e)

    return Rs.view(-1, 23 * 9)


# def rotation_matrix_to_angle_axis(rotation_matrix):
#     """
#     This function is borrowed from https://github.com/kornia/kornia

#     Convert 3x4 rotation matrix to Rodrigues vector

#     Args:
#         rotation_matrix (Tensor): rotation matrix.

#     Returns:
#         Tensor: Rodrigues vector transformation.

#     Shape:
#         - Input: :math:`(N, 3, 4)`
#         - Output: :math:`(N, 3)`

#     Example:
#         >>> input = torch.rand(2, 3, 4)  # Nx4x4
#         >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
#     """
#     if rotation_matrix.shape[1:] == (3,3):
#         rot_mat = rotation_matrix.reshape(-1, 3, 3)
#         hom = torch.tensor([0, 0, 1], dtype=torch.float32,
#                            device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
#         rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

#     quaternion = rotation_matrix_to_quaternion(rotation_matrix)
#     aa = quaternion_to_angle_axis(quaternion)
#     aa[torch.isnan(aa)] = 0.0
#     return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

#__________________intersection tools_______________________

'''
    return whether two segment intersect
'''
def line_intersect(sa, sb):
    al, ar, bl, br = sa[0], sa[1], sb[0], sb[1]
    assert al <= ar and bl <= br
    if al >= br or bl >= ar:
        return False
    return True

'''
    return whether two rectangle intersect
    ra, rb left_top point, right_bottom point
'''
def rectangle_intersect(ra, rb):
    ax = [ra[0][0], ra[1][0]]
    ay = [ra[0][1], ra[1][1]]

    bx = [rb[0][0], rb[1][0]]
    by = [rb[0][1], rb[1][1]]

    return line_intersect(ax, bx) and line_intersect(ay, by)

def get_intersected_rectangle(lt0, rb0, lt1, rb1):
    if not rectangle_intersect([lt0, rb0], [lt1, rb1]):
        return None, None

    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = max(lt[0], lt1[0])
    lt[1] = max(lt[1], lt1[1])

    rb[0] = min(rb[0], rb1[0])
    rb[1] = min(rb[1], rb1[1])
    return lt, rb

def get_union_rectangle(lt0, rb0, lt1, rb1):
    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = min(lt[0], lt1[0])
    lt[1] = min(lt[1], lt1[1])

    rb[0] = max(rb[0], rb1[0])
    rb[1] = max(rb[1], rb1[1])
    return lt, rb

def get_rectangle_area(lt, rb):
    return (rb[0] - lt[0]) * (rb[1] - lt[1])

def get_rectangle_intersect_ratio(lt0, rb0, lt1, rb1):
    (lt0, rb0), (lt1, rb1) = get_intersected_rectangle(lt0, rb0, lt1, rb1), get_union_rectangle(lt0, rb0, lt1, rb1)

    if lt0 is None:
        return 0.0
    else:
        return 1.0 * get_rectangle_area(lt0, rb0) / get_rectangle_area(lt1, rb1)


###############
# DataParallel
###############
def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class _DataParallel(Module):
    r"""Implements data parallelism at the module level.
    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.
    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).
    See also: :ref:`cuda-nn-dataparallel-instead`
    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.
    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])
    Example::
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super(_DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.chunk_sizes = chunk_sizes
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.chunk_sizes)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids, chunk_sizes):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

def DataParallel(module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
    if chunk_sizes is None:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
    standard_size = True
    for i in range(1, len(chunk_sizes)):
        if chunk_sizes[i] != chunk_sizes[0]:
            standard_size = False
    if standard_size:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
    return _DataParallel(module, device_ids, output_device, dim, chunk_sizes)


########
# train_utils
########
def reorganize_items(items, reorganize_idx):
    items_new = [[] for _ in range(len(items))]
    for idx, item in enumerate(items):
        for ridx in reorganize_idx:
            items_new[idx].append(item[ridx])
    return items_new

def justify_detection_state(detection_flag, reorganize_idx):
    if detection_flag.sum() == 0:
        detection_flag = False
    else:
        reorganize_idx = reorganize_idx[detection_flag.bool()].long()
        detection_flag = True
    return detection_flag, reorganize_idx

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = 'module.', drop_prefix='', fix_loaded=False):
    success_layers, failed_layers = [], []
    def _get_params(key):
        key = key.replace(drop_prefix,'')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        if 'mano' in k:
            print(k)

        v = _get_params(k)
        try:
            if v is None:
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix!='':
                k=k.split(prefix)[1]
            success_layers.append(k)
        except:
            logging.info('copy param {} failed, mismatched'.format(k))
            continue
    logging.info('missing parameters of layers:{}, {}'.format(len(failed_layers), failed_layers))
    logging.info('success layers:{}/{}, pre_state_dict have {}'.format(len(success_layers), len(cur_state_dict.keys()), len(pre_state_dict.keys())))
    logging.info('**************************************************************************')
    logging.info('************************** End of loading ********************************')
    logging.info('**************************************************************************')

    x = []
    for i in pre_state_dict.keys():
        if i not in success_layers:
            x.append(i)

    if fix_loaded and len(failed_layers)>0:
        print('fixing the layers that were loaded successfully, while train the layers that failed,')
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad=False
            except:
                print('fixing the layer {} failed'.format(k))

    return success_layers

def load_model(path, model, prefix = 'module.', drop_prefix='',optimizer=None, **kwargs):
    logging.info('using fine_tune model: {}'.format(path))
    if os.path.exists(path):
        pretrained_model = torch.load(path)
        current_model = model.state_dict()
        if isinstance(pretrained_model, dict):
            if 'model_state_dict' in pretrained_model:
                pretrained_model = pretrained_model['model_state_dict']
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

        copy_state_dict(current_model, pretrained_model, prefix = prefix, drop_prefix=drop_prefix, **kwargs)
    else:
        logging.warning('model {} not exist!'.format(path))
        raise ValueError
    return model

def train_entire_model(net):
    N = 0
    exclude_layer = []
    for index,(name,param) in enumerate(net.named_parameters()):
        N += 1
        if '_result_parser' in name:
            print(name)

        if 'mano' not in name:
            param.requires_grad = True
        else:
            if param.requires_grad:
                exclude_layer.append(name)
            param.requires_grad =False

    if len(exclude_layer)==0:
        logging.info('Training all layers. {}'.format(N))
    else:
        logging.info('Train all layers, except: {}/{}'.format(exclude_layer, N))

    return net

def get_remove_keys(dt, keys=[]):
    targets = []
    for key in keys:
        targets.append(dt[key])
    for key in keys:
        del dt[key]
    return targets

def process_idx(reorganize_idx, vids=None):
    result_size = reorganize_idx.shape[0]
    if isinstance(reorganize_idx, torch.Tensor):
        reorganize_idx = reorganize_idx.cpu().numpy()
    used_idx = reorganize_idx[vids] if vids is not None else reorganize_idx
    used_org_inds = np.unique(used_idx)
    per_img_inds = [np.where(reorganize_idx==org_idx)[0] for org_idx in used_org_inds]

    return used_org_inds, per_img_inds

def determine_rendering_order(rendered_img, thresh=0.):
    main_renders = rendered_img[0]
    main_render_mask = (main_renders[:, :, -1] > thresh).cpu().numpy()
    H, W = main_renders.shape[:2]
    render_scale_map = np.zeros((H, W)) + 1
    render_scale_map[main_render_mask] = main_render_mask.sum().item()
    for jdx in range(1,len(rendered_img)):
        other_renders = rendered_img[jdx]
        other_render_mask = (other_renders[:, :, -1] > thresh).cpu().numpy()
        render_scale_map_other = np.zeros((H, W))
        render_scale_map_other[other_render_mask] = other_render_mask.sum().item()
        other_render_mask = render_scale_map_other>render_scale_map
        render_scale_map[other_render_mask] = other_render_mask.sum().item()
        main_renders[other_render_mask] = other_renders[other_render_mask]
    return main_renders[None]

def reorganize_results(outputs, img_paths, reorganize_idx):
    results = {}
    detected = outputs['detection_flag_cache'].detach().cpu().numpy().astype(np.bool_)
    cam_results = outputs['params_dict']['cam'].detach().cpu().numpy().astype(np.float16)[detected]
    trans_results = outputs['cam_trans'].detach().cpu().numpy().astype(np.float16)[detected]
    mano_pose_results = outputs['params_dict']['poses'].detach().cpu().numpy().astype(np.float16)[detected]
    mano_shape_results = outputs['params_dict']['betas'].detach().cpu().numpy().astype(np.float16)[detected]
    joints = outputs['j3d'].detach().cpu().numpy().astype(np.float16)[detected]
    verts_results = outputs['verts'].detach().cpu().numpy().astype(np.float16)[detected]
    pj2d_results = outputs['pj2d'].detach().cpu().numpy().astype(np.float16)[detected]
    pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)[detected]
    hand_type = outputs['output_hand_type'].detach().cpu().numpy().astype(np.int32)[detected]

    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx==vid)[0]
        img_path = img_paths[verts_vids[0]]                
        results[img_path] = [{} for idx in range(len(verts_vids))]
        for subject_idx, batch_idx in enumerate(verts_vids):
            results[img_path][subject_idx]['cam'] = cam_results[batch_idx]
            results[img_path][subject_idx]['cam_trans'] = trans_results[batch_idx]
            results[img_path][subject_idx]['poses'] = mano_pose_results[batch_idx]
            results[img_path][subject_idx]['betas'] = mano_shape_results[batch_idx]
            results[img_path][subject_idx]['j3d'] = joints[batch_idx]
            results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
            results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
            results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
            results[img_path][subject_idx]['hand_type'] = hand_type[batch_idx]
            results[img_path][subject_idx]['detection_flag_cache'] = detected[batch_idx]

    # sort output format.
    new_results = {}
    for name in results.keys():
        data_list = results[name]
        new_results[name] = {'left': [], 'right': []}

        for i in data_list:
            hand_type = i['hand_type']
            if hand_type == 0:
                new_results[name]['left'].append(i)
            elif hand_type == 1:
                new_results[name]['right'].append(i)
            else:
                raise ValueError

    return results

###########
# data processing tools
###########
def image_crop_pad(image, crop_trbl=(0,0,0,0), bbox=None, pad_ratio=1., pad_trbl=None, draw_kp_on_image=False):
    '''
    Input args:
        image : np.array, size H x W x 3
        crop_trbl : tuple, size 4, represent the cropped size on top, right, bottom, left side, Each entry may be a single int.
        bbox : np.array/list/tuple, size 4, represent the left, top, right, bottom, we can derive the crop_trbl from the bbox
        pad_ratio : float, ratio = width / height
        pad_trbl: np.array/list/tuple, size 4, represent the pad size on top, right, bottom, left side, Each entry may be a single int.
    return:
        image: np.array, size H x W x 3
    '''
    if bbox is not None:
        assert len(bbox) == 4, print('bbox input of image_crop_pad is supposed to be in length 4!, while {} is given'.format(bbox))
        def calc_crop_trbl_from_bbox(bbox, image_shape):
            l,t,r,b = bbox
            h,w = image_shape[:2]
            return (int(max(0,t)), int(max(0,w-r)), int(max(0,h-b)), int(max(0,l)))
        crop_trbl = calc_crop_trbl_from_bbox(bbox, image.shape)
    crop_func = iaa.Sequential([iaa.Crop(px=crop_trbl, keep_size=False)])
    image_aug = np.array(crop_func(image=image))
    if pad_trbl is None:
        pad_trbl = compute_paddings_to_reach_aspect_ratio(image_aug.shape, pad_ratio)
    pad_func = iaa.Sequential([iaa.Pad(px=pad_trbl, keep_size=False)])
    image_aug = pad_func(image=image_aug)

    return image_aug, None, np.array([*image_aug.shape[:2], *crop_trbl, *pad_trbl])

def image_pad_white_bg(image, pad_trbl=None, pad_ratio=1.,pad_cval=255):
    if pad_trbl is None:
        pad_trbl = compute_paddings_to_reach_aspect_ratio(image.shape, pad_ratio)
    pad_func = iaa.Sequential([iaa.Pad(px=pad_trbl, keep_size=False,pad_mode='constant',pad_cval=pad_cval)])
    image_aug = pad_func(image=image)
    return image_aug, np.array([*image_aug.shape[:2], *[0,0,0,0], *pad_trbl])

def process_image_ori(originImage, bbox=None):
    orgImage_white_bg, pad_trbl = image_pad_white_bg(originImage)
    image_aug, kp2ds_aug, offsets = image_crop_pad(originImage, bbox=bbox, pad_ratio=1.)
    return orgImage_white_bg, offsets

def img_preprocess(image, imgpath=None, input_size=512, single_img_input=False, bbox=None):
    # args:
    # image: bgr frame
    image = image[:,:,::-1]
    image_org, offsets = process_image_ori(image)

    image = torch.from_numpy(cv2.resize(image_org, (input_size,input_size), interpolation=cv2.INTER_CUBIC))
    offsets = torch.from_numpy(offsets).float()

    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()

    input_data = {
        'image': image,
        'offsets': offsets,
        'data_set': 'internet'} #

    if imgpath is not None:
        name = os.path.basename(imgpath)
        imgpath, name = imgpath, name
        input_data.update({'imgpath': imgpath, 'name': name})
    return input_data

class OpenCVCapture:
    def __init__(self, video_file=None, show=False):
        if video_file is None:
            self.cap = cv2.VideoCapture(int(args().cam_id))
        else:
            self.cap = cv2.VideoCapture(video_file)
            self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.whether_to_show=show

    def read(self, return_rgb=True):
        flag, frame = self.cap.read()
        if not flag:
          return None
        if self.whether_to_show:
            cv2.imshow('webcam',cv2.resize(frame, (240,320)))
            cv2.waitKey(1)
        if return_rgb:
            frame = np.flip(frame, -1).copy() # BGR to RGB
        return frame

class WebcamVideoStream(object):
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        try:
            self.stream = cv2.VideoCapture(src)
        except:
            self.stream = cv2.VideoCapture("/dev/video{}".format(src), cv2.CAP_V4L2)
        
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def split_frame(videopath):
    print(f'processing {videopath} .....')
    # filepath = os.path.dirname(videopath)
    # videoname = os.path.basename(videopath)[:-4]
    if not os.path.exists(videopath):
        raise('{} not exists!'.format(videopath))

    vc = cv2.VideoCapture(videopath)

    path = videopath.replace('.MP4', '').replace('.mp4', '').replace('.MOV', '')
    if not os.path.exists(path):   
        os.mkdir(path)

    if vc.isOpened(): 
        rval , frame = vc.read()
        if 'MOV' not in videopath:
            cv2.imencode('.jpg',frame)[1].tofile(path +'/' + str(0).zfill(6) + '.jpg')
        else:
            cv2.imencode('.jpg',frame[::-1, ::-1])[1].tofile(path +'/' + str(0).zfill(6) + '.jpg')
    else:
        rval = False

    Frame = 1
    timeF = 1
    while rval:
        rval,frame = vc.read()     
        if rval==False:
            break

        if 'MOV' not in videopath:
            cv2.imencode('.jpg',frame)[1].tofile(path +'/' + str(Frame).zfill(6) + '.jpg')
        else:
            cv2.imencode('.jpg',frame[::-1, ::-1])[1].tofile(path +'/' + str(Frame).zfill(6) + '.jpg')

        Frame += 1

    print('save to ', path)
    return path

def save_video(path, out_name):
    print('saving to :', out_name + '.mp4')
    img_array = []
    height, width = 0, 0
    for filename in tqdm(sorted(os.listdir(path), key=lambda x:int(x.split('.')[0]))):
        img = cv2.imread(path + '/' + filename)
        if height != 0:
            img = cv2.resize(img, (width, height))
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(out_name + '.mp4', 0x7634706d, 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('done')

#############################
# temporal optimization
#############################
def extract_motion_sequence(results_track_video, video_track_ids):
    motion_sequence = {}
    frame_ids = sorted(list(results_track_video.keys()))
    subject_ids = []
    for fid, track_ids in video_track_ids.items():
        subject_ids.extend(track_ids)
    subject_ids = np.unique(np.array(subject_ids))
    for subject_id in subject_ids:
        frame_ids_appear = {fid:np.where(np.array(track_ids)==subject_id)[0][0] for fid, track_ids in video_track_ids.items() if subject_id in track_ids}
        motion_sequence[subject_id] = [results_track_video[frame_ids[fid]][sid] for fid, sid in frame_ids_appear.items()]

    return motion_sequence

def smooth_global_rot_matrix(pred_rots, OE_filter):
    rot_mat = batch_rodrigues(pred_rots[None]).squeeze(0)
    smoothed_rot_mat = OE_filter.process(rot_mat)
    smoothed_rot = rotation_matrix_to_angle_axis(smoothed_rot_mat.reshape(1,3,3)).reshape(-1)
    return smoothed_rot

def create_OneEuroFilter(smooth_coeff):
    return {'poses': OneEuroFilter(smooth_coeff, 0.7), 'betas': OneEuroFilter(0.6, 0.7), 'global_orient': OneEuroFilter(smooth_coeff, 0.7)}

def smooth_results(filters, body_pose=None, body_shape=None):
    global_rot = smooth_global_rot_matrix(body_pose[:3], filters['global_orient'])
    body_pose = torch.cat([global_rot, filters['poses'].process(body_pose[3:])], 0)
    body_shape = filters['betas'].process(body_shape)
    return body_pose, body_shape

'''
learn from the minimal hand https://github.com/CalciferZh/minimal-hand
'''
class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
        s = value
    else:
        s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s

class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x, print_inter=False):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    
    if isinstance(edx, float):
        cutoff = self.mincutoff + self.beta * np.abs(edx)
    elif isinstance(edx, np.ndarray):
        cutoff = self.mincutoff + self.beta * np.abs(edx)
    elif isinstance(edx, torch.Tensor):
        cutoff = self.mincutoff + self.beta * torch.abs(edx)
    if print_inter:
        print(self.compute_alpha(cutoff))
    return self.x_filter.process(x, self.compute_alpha(cutoff))