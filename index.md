# Preliminaries
This should be a description of why the topic is covered in the course. Note you can use full markdown syntax here, e.g. including having a "teaser" image for the section that will be managed automatically.


### [OccNets](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks) @ CVPR 2019 – [arXiv](https://arxiv.org/abs/1812.03828) 
![This will become the caption of the image](https://dellaert.github.io/images/NeRF/ON-teaser.png)
TL;DR of the main auto-encoder idea, and encoding occupancy

TEST AUTOMATION

### [DeepSDF](https://github.com/facebookresearch/DeepSDF) @ CVPR 2019 – [arXiv](https://arxiv.org/abs/1901.05103) 
Given a mesh representation of an object, this method converts the mesh to a Signed Distance Function (SDF) and stores this function in a neural network. The main idea is to use an auto-decoder (decoder + latent codes), that takes as input point coordinates and a shape latent code and outputs SDF value. The decoder weights and shape codes are optimized using MAP during training and in test time  only the shape code is optimized. Directly optimizing latent codes in an auto-decoder helps achieve finer details and generalize to test objects better, while in an auto-encoder setting the encoder always expects to receive inputs similar to what its seen at train time. Therefore in an auto-encoder some info and details are lost through encoding to lower dimensions while optimizing latent codes directly can overfit better to the given input.

### [NASA](https://virtualhumans.mpi-inf.mpg.de/nasa/) @ ECCV 2020 – [arXiv](https://arxiv.org/abs/1912.03207) 
This paper introduces an occupancy net for articulated bodies. Progressively presents 3 models of body, where in first one the body is a rigid object, then a piece-wise rigid object and finally deformable. This paper shows the importance of converting coordinates to canonical frame before passing it into an MLP (in this case query/joint coordinates are converted to canonical frame by inverse joint poses). This is due to the fact that MLPs cannot model matrix multiplication well enough and the explicit conversion on input helps MLP to model the occupancy.

# Hierarchical
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [DeepLS](https://arxiv.org/abs/2003.10983) @ ECCV 2020 – [arXiv](https://arxiv.org/abs/2003.10983) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [NGLOD](https://nv-tlabs.github.io/nglod/) @ CVPR 2021 – [arXiv](https://arxiv.org/abs/2101.10994) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [ConvOccNets](https://pengsongyou.github.io/conv_onet) @ ECCV 2020 – [arXiv](https://arxiv.org/abs/2003.04618) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Inverse Rendering Fundamentals
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [SRN](https://vsitzmann.github.io/srns/) @ NeurIPS 2019 – [arXiv](https://arxiv.org/abs/1906.01618) 
This paper introduces joint modeling of shape and appearance. The differentiable renderer introduces differentiable ray marching modeled by a LSTM that finds surface depth. An MLP then maps the 3D coordinate of the point to appearance features and finally RGB color. The colors are not view dependent. This model generalizes to category-level modeling through use of a hyper-net that maps MLP weights to a lower-dimensional sub-space. 

### [Neural Volumes](https://research.fb.com/publications/neural-volumes-learning-dynamic-renderable-volumes-from-images/) @ SIGGRAPH 2019 – [arXiv](https://arxiv.org/abs/1906.07751) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Deffered Neural Rendering](https://niessnerlab.org/projects/thies2019neural.html) @ SIGGRAPH 2019 – [arXiv](https://arxiv.org/abs/1904.12356) 
This paper focuses on having learned textures to model view-dependent appearance. The geometry is not modeled. Unlike traditional texture maps, learned neural textures are used to store high dimensional features rather than RGB. These can be converted to pixel color using standard texel-to-pixel pipelines but with a differentiable renderer. The texture sampling is hierarchical similar to mimaps to overcome resolution differences. The sampling is made differentiable through bi-linear interpolation.

# Neural Radiance Fields
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Volume Rendering Tutorial](https://drops.dagstuhl.de/opus/volltexte/2010/2709/pdf/18.pdf) @ Schloss Dagstuhl 2010 – 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [IDR](https://lioryariv.github.io/idr/) @ NeurIPS 2020 – [arXiv](https://arxiv.org/abs/2003.09852) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [NeRF](https://www.matthewtancik.com/nerf) @ ECCV 2020 – [arXiv](https://arxiv.org/abs/2003.08934) 
The one, the only, NeRF !!!!

# Neural Light Fields
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [LFN: Light Field Networks](https://www.vincentsitzmann.com/lfns) @ NeurIPS 2021 – [arXiv](https://arxiv.org/abs/2106.02634) 
In this paper a new parameterization for 4D light fields is introduced. This parameterization is based on the 6-dimensional Plücker coordinates in which light fields lie on a 4-dimensional manifold. This parametrization allows or 360 views of scenes unlike the traditional parametrization of light fields. The 6D coordinate is taken as input to a MLP and ray color is given as output. Again, because there is no volume rendering or integration, a pixel color can  At last a meta-learning approach based on hyper-network training is proposed to generalize to different scenes.

### [Light Field Neural Rendering](https://light-field-neural-rendering.github.io/) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2112.09687) 
This work models 4D light field with the use of transformers. Given a sparsely observed scene, this method uses epipolar geometry constraints (much like IBR) to model geometry with help of a transformer and then view-dependent appearance of a light field through a second transformer. Two closest reference views to target view are selected and the sampled points along target rays are projected onto them. First transformer decides which one of the sampled points on the ray is closest to the query point through predicting attentions. Second transformer decides which of the reference view directions to pay more attention to.

### [Learning Neural Light Fields](https://neural-light-fields.github.io/) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2112.01523) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Image Based Rendering
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [pixelNeRF](https://github.com/sxyu/pixel-nerf) @ CVPR 2021 – [arXiv](https://arxiv.org/abs/2012.02190) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Stereo Radiance Fields](https://virtualhumans.mpi-inf.mpg.de/srf/) @ CVPR 2021 – [arXiv](https://arxiv.org/abs/2104.06935) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [SRT](https://srt-paper.github.io/) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2111.13152) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Multi Resolution
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [NSVF](https://github.com/facebookresearch/NSVF) @ NeurIPS 2020 – [arXiv](https://arxiv.org/abs/2007.11571) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Mip-NeRF](https://jonbarron.info/mipnerf/) @ ICCV 2021 – [arXiv](https://arxiv.org/abs/2103.13415) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Mip-NeRF-360](https://jonbarron.info/mipnerf360/) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2111.12077) 
The contributions are three-fold. First, a new parameterization for the space is introduced to model unbounded scenes. The foreground is parametrized linearly as before, but the background is contracted (based on inverse depth) into a bounded sphere of fixed radius. Therefore distant points are distributed proportional to inverse depth i.e. the further the less detailed the scene becomes. A compatible sampling method is introduced to sample uniformly in inverse depth. Second, a hierarchical scheme is used for importance sampling. Two MLPs are used, one proposal MLP and the other NeRF MLP. The NeRF MLP works as before, but the proposal MLP tries to estimate the distribution of important points that should be sampled along the ray. 

# Fast Rendering
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [KiloNeRF](https://creiser.github.io/kilonerf/) @ ICCV 2021 – [arXiv](https://arxiv.org/abs/2103.13744) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [SNeRG](https://arxiv.org/abs/2103.14645) @ ICCV 2021 – [arXiv](https://arxiv.org/abs/2103.14645) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [FastNeRF](https://arxiv.org/abs/2103.10380) @ ICCV 2021 – [arXiv](https://arxiv.org/abs/2103.10380) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Fast Training
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [InstantNGP](https://nvlabs.github.io/instant-ngp/) @ SIGGRAPH 2022 – [arXiv](https://arxiv.org/abs/2201.05989) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Plenoxels](https://alexyu.net/plenoxels) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2112.05131) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [ReLUFields](https://geometry.cs.ucl.ac.uk/group_website/projects/2022/relu_fields/) @ SIGGRAPH 2022 – [arXiv](https://arxiv.org/abs/2205.10824) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Camera Extrinsics
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [BARF](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/) @ ICCV 2021 – [arXiv](https://arxiv.org/abs/2104.06405) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Nerf--](https://nerfmm.active.vision/) @ Arxiv 2021 – [arXiv](https://arxiv.org/abs/2102.07064) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [GARF](https://sfchng.github.io/garf/) @ Arxiv 2022 – [arXiv](https://arxiv.org/abs/2204.05735) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Camera Intrinsics
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [NerfDark](https://bmild.github.io/rawnerf/index.html) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2111.13679) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [DeblurNeRF](https://limacv.github.io/deblurnerf/) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2111.14292) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Learnable Appearance
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [NeRF in-the-wild](https://nerf-w.github.io/) @ CVPR 2021 – [arXiv](https://arxiv.org/abs/2008.02268) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Nerfies](https://nerfies.github.io/) @ ICCV 2021 – [arXiv](https://arxiv.org/abs/2011.12948) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [HyperNeRF](https://hypernerf.github.io/) @ SIGGRAPH Asia 2021 – [arXiv](https://arxiv.org/abs/2106.13228) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Semantic Understanding
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [Semantic NERF](https://shuaifengzhi.com/Semantic-NeRF/) @ ICCV 2021 – [arXiv](https://arxiv.org/abs/2103.15875) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [NeSF](https://nesf3d.github.io/) @ TMLR 2022 – [arXiv](https://arxiv.org/abs/2111.13260) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [PNF](https://abhijitkundu.info/projects/pnf/) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2205.04334) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

# Generative Modeling
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [LOLNeRF](https://ubc-vision.github.io/lolnerf/) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2111.09996) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [GANcraft](https://nvlabs.github.io/GANcraft/) @ ICCV 2021 – [arXiv](https://arxiv.org/abs/2104.07659) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

### [EG3D](https://nvlabs.github.io/eg3d/) @ CVPR 2022 – [arXiv](https://arxiv.org/abs/2112.07945) 
At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.

