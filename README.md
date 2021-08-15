# Deep Virtual Markers

This repository contains the accompanying code for [Deep Virtual Markers for Articulated 3D Shapes, ICCV'21]()

<p align="center"><img src = "./sample_results/teaser.jpg" height ="500" /> 

## Getting Started

Get sample data and pre-trained weight from [here]() (will be updated)

### Simple Test (Docker)
```
docker pull min00001/cuglmink
./run_dvm_test.sh
```
<img src = "./sample_results/1.png" width ="16%" /> <img src = "./sample_results/2.png" width ="16%" /> <img src = "./sample_results/2_.png" width ="16%" /> <img src = "./sample_results/3.png" width ="16%" /> <img src = "./sample_results/4.png" width ="16%" /> <img src = "./sample_results/5.png" width ="16%" />

<!-- ### Prerequisites

- Ubuntu 18.06 or higher
- CUDA 10.2 or higher
- pytorch 1.6 or higher
- python 3.8 or higher
- GCC 6 or higher

### Installing -->

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## About Coupe Project
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using it. In addition,ersonalization technology through userreference analysis is under study.  
    
Please checkout out other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)


## Related projects

**NOTE** : Our implementation is based on the ["4D-SpatioTemporal ConvNets"](https://github.com/chrischoy/SpatioTemporalSegmentation) repository
- [4D-SpatioTemporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://github.com/chrischoy/SpatioTemporalSegmentation)
- [Minkowski Engine, a neural network library for sparse tensors](https://github.com/StanfordVL/MinkowskiEngine)
- [Fully Convolutional Geometric Features, ICCV'19, fast and accurate 3D features](https://github.com/chrischoy/FCGF)
