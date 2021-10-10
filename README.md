# Deep Virtual Markers

This repository contains the accompanying code for [Deep Virtual Markers for Articulated 3D Shapes, ICCV'21](https://arxiv.org/pdf/2108.09000.pdf)

| [Paper](https://arxiv.org/pdf/2108.09000.pdf) | [Video](https://youtu.be/Raq5axLdG6E) |

<p align="center">
    <img src = "./sample_results/teaser.jpg" width ="35.2%" />
    <img src = "./sample_results/MotionTracking.gif" width ="47%" /> 
</p>

## Athor Information

- [Hyomin Kim]()
- [Jungeon Kim]()
- [Jaewon Kam]()
- [Jaesik Park](http://jaesik.info/) [[Google Scholar]](https://scholar.google.com/citations?user=_3q6KBIAAAAJ&hl=en&oi=ao)
- [Seungyong Lee](http://cg.postech.ac.kr/leesy/) [[Google Scholar]](https://scholar.google.com/citations?user=yGPH-nAAAAAJ&hl=en&oi=ao)

## News

- 2021-08-16: The first virsion of Deep Virtual Markers is published

## Overview

We propose deep virtual markers, a framework for estimating dense and accurate positional information for various types of 3D data. We design a concept and construct a framework that maps 3D points of 3D articulated models, like humans, into virtual marker labels. To realize the framework, we adopt a sparse convolutional neural network and classify 3D points of an articulated model into virtual marker labels. We propose to use soft labels for the classifier to learn rich and dense interclass relationships based on geodesic distance. To measure the localization accuracy of the virtual markers, we test FAUST challenge, and our result outperforms the state-of-the-art. We also observe outstanding performance on the generalizability test, unseen data evaluation, and different 3D data types (meshes and depth maps). We show additional applications using the estimated virtual markers, such as non-rigid registration, texture transfer, and realtime dense marker prediction from depth maps.

## Getting Started

### Prerequisites

- Ubuntu 18.06 or higher
- CUDA 10.2 or higher
- pytorch 1.6 or higher
- python 3.8 or higher
- GCC 6 or higher

### Environment Setup

- We recommend using docker
```
docker pull min00001/cuglmink
```

## Demo Code (Docker)
Get sample data and pre-trained weight from [here](https://1drv.ms/u/s!AtCM45bsnwBNmGzeBIEI7Y5dMvis?e=ZemVdP)
```
docker pull min00001/cuglmink
./run_dvm_test.sh
```
<img src = "./sample_results/1.png" width ="16%" /> <img src = "./sample_results/2.png" width ="16%" /> <img src = "./sample_results/2_.png" width ="16%" /> <img src = "./sample_results/3.png" width ="16%" /> <img src = "./sample_results/4.png" width ="16%" /> <img src = "./sample_results/5.png" width ="16%" />

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## Useful Links
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

## Citing Deep Virtual Markers

```
@inproceedings{kim2021deep,
  title={Deep Virtual Markers for Articulated 3D Shapes},
  author={Hyomin Kim, Jungeon Kim, Jaewon Kam, Jaesik Park and Seungyong Lee},
  booktitle={ICCV},
  year={2021}
}
```

## Related projects

**NOTE** : Our implementation is based on the ["4D-SpatioTemporal ConvNets"](https://github.com/chrischoy/SpatioTemporalSegmentation) repository
- [4D-SpatioTemporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://github.com/chrischoy/SpatioTemporalSegmentation)
- [Minkowski Engine, a neural network library for sparse tensors](https://github.com/StanfordVL/MinkowskiEngine)
- [Fully Convolutional Geometric Features, ICCV'19, fast and accurate 3D features](https://github.com/chrischoy/FCGF)
