# SERR-U-Net-Retinal-Vessel-Segmentation
SERR-U-Net: A improved U-Net Framework with a Saliency Mechanism for Retinal Vessel Segmentation

# 1 Overview
# 2 Datasets
#  2.1 DRIVE
#  2.2 STARE
# 3 About Model
#  3.1 U-Net
#  3.2 SE Net
#  3.3 Recurrent Block and ResNet
#  3.4 Enhanced-Super-Resolution Generative Adversarial Networks (ESRGAN)
# 4 Environments
#  4.1 Requirements
#  4.2 About Keras
#  4.3 Result Evaluation
# 5 Future Work
 # 1 Overview
Introduction to human eye structure and fundus images:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312094752660.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)
Fundus disease is a modern medical term, and fundus is the part of the eye that can be seen through the pupil. The vascular analysis of fundus image is one of the important basis for the current diagnosis of ophthalmic diseases and systemic cardiovascular and cerebrovascular diseases such as diabetes mellitus, glaucoma and hypertension. For example, in patients with diabetes mellitus, there will be arteriole hemangioma and other focal areas, and in patients with hypertension, there will be arteriosclerosis in the retinal vessels, which also reflects the changes in the blood vessels of the whole body.
In the clinical application of medicine, fundus image blood vessel is one of the more complex medical images. At the same time, the rapid growth of fundus image data leads to the strong subjectivity and low efficiency of doctors and experts if they only rely on experience judgment and manual observation, which virtually increases the workload of doctors and experts. In a 2019 survey, there were an estimated 93 million cases of retinopathy, which accounts for 7 to 8 percent of all blindness worldwide. All of these retinal-related cases could triple due to an aging population, changing lifestyles and other risks. These numbers have driven a large number of researchers to develop intelligent diagnostic tools to meet human needs, so the use of computer aided automatic detection and segmentation of the vascular network in fundus images has important clinical significance.
We propose an improved neural network based on **U-NET**, which uses **SE**, **residual structure** and **recurrent block.** **We propose a data enhancement method, which not only randomly cropped the fundus image, but also rotated the fundus image. In addition, **ESRGAN**, **Gamma transform** and **CLAHE** are also used.**
We train and evaluate on **Ubuntu 18.04**, it will also work for Windows and OS.
Workstation configuration is as follows:
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021031209273596.png)
**Notice：This Project structure updated on March 12！**
You can find old version in branch old.

# 2 Datasets
Most retinal vascular segmentation methods are evaluated in the DRIVE and STARE datasets due to their high quality and have been created for at least 15 years. These two databases will be used to train and test the proposed supervised learning approach. STARE consists of 20 fundus images, 10 of which have lesions and 10 of which have no lesions. The image resolution is 605×700. Each image corresponds to the result of manual segmentation by 2 experts, but it has no mask and needs to be set manually by itself. There are 40 color fundus images in DRIVE, of which 7 have symptoms of early diabetic retinopathy and 33 have no symptoms of diabetic retinopathy. The pixel of each image is 565×584, and each image corresponds to the results manually marked by 2 experts. The data itself has a special mask, which is convenient to call.
## 2.1 DRIVE


[https://drive.grand-challenge.org/Download/](https://drive.grand-challenge.org/Download/)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312094952825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)
## 2.2 STARE
[http://cecas.clemson.edu/~ahoover/stare/](http://cecas.clemson.edu/~ahoover/stare/)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312094936586.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)


# 3 About Model
## 3.1 U-Net
U-NET is strengthened and modified on the basis of FCN (U-NET adopts superposition instead of summation operation in FCN in shallow feature fusion). The network consists of two parts: one is the Contracting path to obtain context information and the other is the Expanding path to be symmetrically used for accurate positioning. Every two 3×3 convolution layers on the contraction path will be superimposed with a 2×2 maximum pooling layer with a step size of 2. After each convolution layer, ReLU activation function will be used to perform down-sampling on the original image, and each down-sampling will increase the number of channels by a factor of one. In the upward sampling of the extended path, at each step there will be a 2×2 activation function working for the ReLU layer and the 3×3 convolution layer. At the same time, the upsampling of each step will be attached to the feature map of the relative shrinkage path. The last layer of the network is a 1×1 convolution layer, and through this series of operations, the 64-channel feature vector can be converted into the number of required classification results. In general, the whole network of U-NET has 23 convolution layers. The three-dimensional network structure is shown in the figure below, which is composed of input layer, hidden layer and output layer.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312095744618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)
## 3.2 SE Net
Squeeze - and - Excitation Networks (SE.net) is Momenta team (WMW) put forward a new network structure, using SE.net hu jie team made the last ImageNet 2017 race Image Classification task, winner of the Top - 5 Error on ImageNet dataset reduced to 2.251%, the original best result was 2.991%. The author inserts SE Net Block into the existing classification networks and achieves good results, which can explicitly build the dependency between feature channels. In addition, instead of introducing new spatial dimensions for the fusion of feature channels, a feature recalibration strategy is adopted. Specifically, it is to automatically obtain the importance degree of each feature channel through learning, and then capture the useful features according to this importance degree and discard the features that are not important to the current task to a certain extent. The core idea of the network is to learn feature weights according to Loss, so that useful feature maps have significant weights and useless or less useful feature maps have small weights. The model trained in this way can achieve better results. Although the addition of SE block structure increases some parameters and computation in some original classification networks, it achieves better results.
The squeezes and controls are the two most critical steps. The image below is the SE module. Given an input X, the feature number is C1. Different from CNN, three operations are used to recalibrate the previously acquired features.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312095911346.png)

The first advertisement operation will follow the spatial dimension for feature compression, each characteristic of the two-dimensional channel into a real, the feelings of the real part represents the global field, this method is very useful in many tasks, the size of input characteristics of the integrated C * H * W * 1 * 1 C character description, for a drawing of the characteristic, the calculation is as follows.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312100013105.png)
The second is the operations. It is a mechanism similar to the doors in RNN. A parameter w is used to generate a weight for each feature channel, which is used to explicitly build the correlation between feature channels. The operation includes two full-connection layers and the Sigmoid activation function. The full-connection layer can well fuse all the input characteristic information, while the Sigmoid function can also map the input to the interval of 0~1. The calculation is as follows.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312100046547.png)

Where Z is the global description obtained by the Squeeze operation, δ represents the ReRelu function to ensure that the output is positive, W1 and W2 are two fully connected layers, where R is the scaling parameter, mainly used to reduce the computational complexity and the number of parameters of the network.
The last is the Reweight operation. The weight results for the output are treated as the importance metrics for each feature channel after feature selection. The re-calibration of the original features on the channel dimensions is completed using the multiplicative weight over the previous features. The fusion operation is calculated as follows.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312100220212.png)



## 3.3 Recurrent Block and ResNet
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312100849942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)



## 3.4 Enhanced-Super-Resolution Generative Adversarial Networks (ESRGAN)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312100641873.png)

# 4 Environments
## 4.1 Requirements

```python
Python 3.7
Keras 2.2.5
scikit-learn 0.18.1
tensorboard 1.2.0
matplotlib 2.0.2
```

## 4.2  About Keras
Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility). supports both convolutional networks and recurrent networks, as well as combinations of the two. supports arbitrary connectivity schemes (including multi-input and multi-output training). runs seamlessly on CPU and GPU. Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.


## 4.3 Result Evaluation
An effective performance evaluation index is needed to judge the quality of the segmentation method of fundus ocular blood vessels. The results of the segmentation of blood vessels are usually compared with the gold standard manually marked by experts. A segmentation result evaluation for pixel points is shown in the table below.
| Object segmentation | Positive  |Negative |
|--|--|--|
| Vessel point |  TP|FP|
| Background points |TN  |FN|

In the table, true positive (TP) refers to correctly segmented blood vessel pixels, false positive (FP) refers to wrongly segmented blood vessel pixels, true negative (TN) refers to correctly segmented background pixels, and false negative (FN) refers to wrongly segmented background pixels. According to the segmentation results in the above table, several evaluation indexes such as accuracy (ACC), specificity (SP), sensitivity (SE) and Precision (Precision) can be obtained. The calculation method is shown in the following table.

| Evaluation index of vascular segmentation performance | Calculation method  |
|--|--|
| ACC |  TP / (TP + FN)|
| SP|TN/ (TN + FP) |
| SE|TP / (TP + FN) |

**The ROC curve is as follows:**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312102033729.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)
**The segmentation effect is as follows:**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210312102410413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgzODc4NQ==,size_16,color_FFFFFF,t_70)

# 5 Future Work

Attention-based Unet and DeepLab-v3+ are also worth to try.
