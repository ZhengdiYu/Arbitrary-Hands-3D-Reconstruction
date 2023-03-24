import sys, os, cv2
from tqdm import tqdm
import logging
import torch
import torch.nn as nn

##################
# config and utils
##################
import acr.config as config
from acr.config import args, parse_args, ConfigContext
from acr.utils import *
from acr.utils import justify_detection_state, reorganize_results, collect_image_list, img_preprocess, WebcamVideoStream, split_frame, save_results
from acr.visualization import Visualizer
if args().model_precision=='fp16':
    from torch.cuda.amp import autocast

########################
# models and dataloader
########################
from acr.model import ACR as ACR_v1
from acr.mano_wrapper import MANOWrapper

class ACR(nn.Module):
    def __init__(self, args_set=None):
        super(ACR, self).__init__()
        self.demo_cfg = {'mode':'parsing', 'calc_loss': False}
        self.project_dir = config.project_dir
        self._initialize_(vars(args() if args_set is None else args_set))

        logging.info('Loading {} renderer as visualizer, rendering size: {}'.format(self.renderer, self.render_size))
        self.visualizer = Visualizer(resolution=(self.render_size,self.render_size), renderer_type=self.renderer)

        print('building model')
        self._build_model_()
        print('Initialization finished!')

    def _initialize_(self, config_dict):
        # configs
        hparams_dict = {}
        for i, j in config_dict.items():
            setattr(self,i,j)
            hparams_dict[i] = j

        logging.basicConfig(level=logging.INFO)
        logging.info(config_dict)
        logging.info('-'*66)

        # optimizations parameters
        if self.temporal_optimization:
            self.filter_dict = {}
            self.filter_dict[0] = create_OneEuroFilter(args().smooth_coeff)
            self.filter_dict[1] = create_OneEuroFilter(args().smooth_coeff)

        return hparams_dict

    def _build_model_(self):
        model = ACR_v1().eval()
        model = load_model(self.model_path, model, prefix = 'module.', drop_prefix='', fix_loaded=False) 
        # train_entire_model(model)
        self.model = nn.DataParallel(model.cuda())
        self.model.eval()
        self.mano_regression = MANOWrapper().cuda()

    @torch.no_grad()
    def process_results(self, outputs):

        # temporal optimization
        if self.temporal_optimization:
            out_hand = [] # [0],[1],[0,1]
            for idx, i in enumerate(outputs['detection_flag_cache']):
                if i:
                    out_hand.append(idx) # idx is also hand type, 0 for left, 1 for right
                else:
                    out_hand.append(-1)

            assert len(outputs['params_dict']['poses']) == 2
            for sid, tid in enumerate(out_hand):
                if tid == -1:
                    continue
                outputs['params_dict']['poses'][sid], outputs['params_dict']['betas'][sid] = \
                    smooth_results(self.filter_dict[tid], \
                    outputs['params_dict']['poses'][sid], outputs['params_dict']['betas'][sid])

        outputs = self.mano_regression(outputs, outputs['meta_data'])
        reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
        new_results = reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

        return outputs, new_results

    @torch.no_grad()
    def forward(self, bgr_frame, path):
        with torch.no_grad():
            outputs = self.single_image_forward(bgr_frame, path)

        if outputs is not None and outputs['detection_flag']:
            outputs, results = self.process_results(outputs)

            # visualization: render to raw image
            show_items_list = ['mesh'] # ['org_img', 'mesh', 'pj2d', 'centermap']
            results_dict, img_names = self.visualizer.visulize_result_live(outputs, bgr_frame, outputs['meta_data'], \
                show_items=show_items_list, vis_cfg={'settings':['put_org']}, save2html=False)

            img_name, mesh_rendering_orgimg = img_names[0], results_dict['mesh_rendering_orgimgs']['figs'][0]

            if self.save_visualization_on_img and args().demo_mode!='webcam':
                save_name = os.path.join(self.output_dir + os.path.basename(img_name))
                cv2.imwrite(save_name, mesh_rendering_orgimg[:,:,::-1])
            else:
                cv2.imshow('render_output', mesh_rendering_orgimg[:,:,::-1])
                cv2.waitKey(1)
        else:
            print('no hand detected!')
            results = {}
            results[path] = {}
            if self.save_visualization_on_img and args().demo_mode!='webcam':
                save_name = os.path.join(self.output_dir + os.path.basename(path))
                cv2.imwrite(save_name, bgr_frame)
            else:
                cv2.imshow('render_output', bgr_frame)
                cv2.waitKey(1)

        return results

    @torch.no_grad()
    def single_image_forward(self, bgr_frame, path):
        meta_data = img_preprocess(bgr_frame, path, input_size=args().input_size, single_img_input=True)

        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision=='fp16':
            with autocast():
                outputs = self.model(meta_data, **self.demo_cfg)
        else:
            outputs = self.model(meta_data, **self.demo_cfg)

        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'], outputs['reorganize_idx'])
        meta_data.update({'imgpath':imgpath_org, 'data_set':ds_org})
        outputs['meta_data']['imgpath'] = [path]

        return outputs


def main():
    ################## Model Initialization ####################
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        acr = ACR(args_set=args_set)

    ################## RUN on image forlder ####################
    results_dict = {}
    if args().demo_mode == 'image':

        acr.output_dir = './demos_outputs/single_images_output/'
        print('output dir:', acr.output_dir)
        os.makedirs(acr.output_dir, exist_ok=True)

        image = cv2.imread(imgpath)
        outputs = acr(image, imgpath)
        results_dict.update(outputs)

        if args().save_dict_results:
            save_results(imgpath, acr.output_dir, results_dict)

    ###################### RUN on video ########################
    elif args().demo_mode == 'video' or args().demo_mode == 'folder':
        if not os.path.isdir(args().inputs):
            image_folder = split_frame(args().inputs)
        else:
            image_folder = args().inputs[:-1] if args().inputs.endswith('/') else args().inputs

        print('running on: ', image_folder)
        acr.output_dir = './demos_outputs/' + os.path.basename(image_folder) + f'_results_{args().centermap_conf_thresh}' + '/' + args().model_path.split('/')[-1] + '/'
        print('output dir:', acr.output_dir)
        os.makedirs(acr.output_dir, exist_ok=True)

        file_list = collect_image_list(image_folder=image_folder)
        try:
            file_list = sorted(file_list, key=lambda x:int(os.path.basename(x).split('.')[0])) # please ensure the image name is something like '000000.jpg'
        except:
            print('warning: image filename is not in order.')

        bar = tqdm(file_list)
        for imgpath in bar:
            image = cv2.imread(imgpath)
            outputs = acr(image, imgpath)
            results_dict.update(outputs)

        if args().save_visualization_on_img:
            save_video(acr.output_dir, os.path.basename(image_folder) + '_output_' + os.path.basename(args().model_path.replace('.pkl','')))

        if args().save_dict_results:
            save_results(image_folder, acr.output_dir, results_dict)

    ###################### RUN on webcame ########################
    elif args().demo_mode == 'webcam':
        cap = WebcamVideoStream(args().cam_id)
        cap.start()
        while True:
            frame = cap.read()
            outputs = acr(frame, '0')
        cap.stop()

if __name__ == '__main__':
    main()
