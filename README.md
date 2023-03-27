[![arXiv](https://img.shields.io/badge/arXiv-2303.05938-b31b1b.svg)](https://arxiv.org/abs/2303.05938)
![code visitors](https://visitor-badge.glitch.me/badge?page_id=ZhengdiYu/Arbitrary-Hands-3D-Reconstruction)
[![GitHub license](https://img.shields.io/badge/license-Apache2.0-blue.svg)](https://github.com/ZhengdiYu/Arbitrary-Hands-3D-Reconstruction/blob/main/LICENSE)

# ACR: Attention Collaboration-based Regressor for Arbitrary Two-Hand Reconstruction [CVPR 2023]


This is the official repository of the ACR.

🔥(**CVPR 2023**) **ACR: Attention Collaboration-based Regressor for Arbitrary Two-Hand Reconstruction**

[Zhengdi Yu](https://github.com/ZhengdiYu), [Shaoli Huang](https://scholar.google.com/citations?user=o31BPFsAAAAJ&hl=en&oi=ao), [Chen Fang](http://fangchen.org/), [Toby P. Breckon](https://breckon.org/toby/research/), [Jue Wang](https://juewang725.github.io/)
> *Conference on Computer Vision and Pattern Recognition (CVPR), 2023*

**[[Paper](https://arxiv.org/abs/2303.05938)][[Project Page](https://semanticdh.github.io/ACR/)][[Video](https://youtu.be/WKRkm3Tfn3s)]**

<p float="left">
  <img src="docs/p1.GIF" width="49%" />
  <img src="docs/P2.GIF" width="49%" />
</p>

# News :triangular_flag_on_post:

- [2023/03/24] **Code release!** ⭐
- [2023/03/10] **ACR is on [arXiv](https://arxiv.org/abs/2303.05938) now.**
- [2023/02/27] **ACR got accepted by CVPR 2023!** 🎉
## Requirements

### Conda environments
```
conda create -n ACR python==3.8.8  
conda activate ACR 
conda install -n ACR pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

For rendering and visualization on headless server, please consider install `pytorch3d` follow [the official instruction](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and set `renderer` to `pytorch3d` in `configs/demo.yml`. Note that `pyrender` can only be used on desktop.

### Pre-trained model and data
- Register and download [MANO](https://mano.is.tue.mpg.de/) model. Put `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in `mano/`
- Download pre-trained weights from [here](https://drive.google.com/file/d/1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9/view?usp=share_link) and put it in `checkpoints/`

## Demo
Note: use `-t` to smooth your results. We provide examples in `demos/`

```
# Run a real-time demo:
python -m acr.main --demo_mode webcam -t

# Run on a single image:
python -m acr.main --demo_mode image --inputs <PATH_TO_IMAGE>

# Run on a folder of images:
python -m acr.main --demo_mode folder -t --inputs <PATH_TO_FOLDER> 

# Run on a video:
python -m acr.main --demo_mode video -t --inputs <PATH_TO_VIDEO> 
```

Finally, the visualization will be saved in `demos_outputs/`. In `video` or `folder` mode, the results will also be saved as `<FILENAME>_output.mp4`.

## More qualitative results
![image](https://user-images.githubusercontent.com/63605407/222917470-0daf33b4-868f-442d-8615-2fba6bf6e719.png)
![wild](https://user-images.githubusercontent.com/63605407/224312107-bb102043-80bc-48e3-829d-18248098a623.png)



## Applications
<p float="left">
  <img src="docs/1.gif" width="33%" />
  <img src="docs/2.gif" width="31%" />
  <img src="docs/3.gif" width="33%" />
</p>


<p float="left">
  <img src="docs/4.gif" width="49.5%" />
  <img src="docs/5.gif" width="49.5%" />
</p>

## Citation
```
@inproceedings{yu2023acr,
  title = {ACR: Attention Collaboration-based Regressor for Arbitrary Two-Hand Reconstruction},
  author = {Yu, Zhengdi and Huang, Shaoli and Chen, Fang and Breckon, Toby P. and Wang, Jue},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023}
  }
```

## Acknowledgement
The pytorch implementation of MANO is based on [manopth](https://github.com/hassony2/manopth). We use some parts of the great code from [ROMP](https://github.com/Arthur151/ROMP/). We thank all the authors for their impressive works!

## Contact
For technical questions, please contact zhengdi.yu@durham.ac.uk or ZhengdiYu@hotmail.com

For commercial licensing, please contact shaolihuang@tencent.com
