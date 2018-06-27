# Object recoloring

This project is to do many-to-many mapping on color dataset.

<p align="center"><img width="100%" height='60%'src="images/comparation.png" /></p>

# Overview 
- [Dependences](#dependences)
- [Installation](#installtion)
- [Data](#Data)
- [Instructions](#instructions)
- [Citation](#citation)
- [Contact](#contact)
# Dependences 
- Python2.7, NumPy, SciPy, NVIDIA GPU
- **Tensorflow:** the version should be more 1.0(https://www.tensorflow.org/) and [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
- **Dataset:**  [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html) 

# Installation 
- Install Tensorflow and TensorBoard

# Data
- Download RGBD dataset: 17 folders(using seriver to download and unzip)
- Changing path of  `data_to_tfrecords.py` to get tfrecord 
```bash
$ python data_to_tfrecords.py
```

# Instructions
- Cloning the repository
```bash
$ git clone  https://github.com/yaxingwang/Mix-and-match-networks.git
$ cd Mix-and-match-networks/RGBD/
```

- Training 
```bash
$ python train.py 
```
During training, the loss and generated image can be monitored by tensorboard as:
```bash
$ cd logs 
$ tensorboard --logdir=./ 
```


The pretrained model is provided [here](https://drive.google.com/drive/folders/10K-Ip_EMWBTRtBkvxFSiDOylkVZymWnz). If you have trained model, just put it as following:

# Citation

If this work or color dataset are useful for your research, please cite  papers:
```
@article{wang2018mix,
  title={Mix and match networks: encoder-decoder alignment for zero-pair image translation},
    author={Wang, Yaxing and van de Weijer, Joost and Herranz, Luis},
      journal={arXiv preprint arXiv:1804.02199},
        year={2018}
        }
 @article{yu2018weakly,
   title={Weakly Supervised Domain-Specific Color Naming Based on Attention},
     author={Yu, Lu and Cheng, Yongmei and van de Weijer, Joost},
       journal={arXiv preprint arXiv:1805.04385},
         year={2018}
         }

```




# Contact

If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: yaxing@cvc.uab.es
