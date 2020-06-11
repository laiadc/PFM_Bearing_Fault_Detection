# Deep learning for predictive maintenance of rolling bearings

A deep learning approach for detecting early development of rolling bearing failures and their root cause.

This repository contains the code used to monitor the health of rolling bearings. The results are illustrated in a set of [Jupyter](https://jupyter.org/) notebooks. The whole code is the result of a Master's Thesis of the [Master's Degree in the
Fundamental Principles of Data Science](http://www.ub.edu/datascience/master/) supervised by Dr. Jordi Vitrià. This repository also includes the Master's Thesis report, in pdf format. Any contribution or idea to continue the lines of the proposed work will be very welcome.

In order to perform predictive maintenance of rolling bearings, the project is split into three parts:

+ **Early detection of the failure**: It is important to detect and classify bearing failures at the early stages of their development so that one cantake appropriate actions before the machine breaks.  For this reason, an early detection model is developed. This model is based on a one-dimensional convolutional autoencoder, which is an unsupervised method which detects if there has been a deviation from the normal (healthy) behaviour of the machine.

<p align="center"><img src="https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Images/ae.png"  align=middle width=400pt />
</p>
<p align="center">
<em>Representation of a one-dimensional convolutional autoencoder.  The autoencoder is trained only with healthy data so that it learns to reconstruct only the healthy waveforms. The reconstruction error is taken as an indicator of anomalous (unhealthy) behaviour. </em>
</p>

+ **Classification of the failure**:  a classification model is trained to determine the position (inner ring, outer ring, rolling elementor cage) of the bearing failure. This model is trained so that it is robust to different operation conditions (such as the rotation velocity) and so that it can be trained with few data samples. In order to accomplish these requirements, a convolutional neural network is  trained using triplet learning strategies or using a Siamese network with contrastive loss.

<p align="center"><img src="https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Images/siamese.png"  align=middle width=350pt />
</p>
<p align="center">
<em>Representation of a Siamese network.  The goal of the network is to create useful embeddings from the input data, so that the classifier can predict the position of the failure regardless of the operating condition of the machine.</em>
</p>

<p align="center"><img src="https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Images/triplet.PNG"  align=middle width=500pt />
</p>
<p align="center">
<em>Representation of a triplet learning network. This network is similar to the Siamese network, but it is designed to work with small data sets. Also, the training phase is more efficient, even though it is more difficult to implement in Tensorflow.</em>
</p>


+ **Continuous learning of the models**: Usually,  all the historical data is not provided at the training stage,  but it isacquired progressively with time. Therefore, it is interesting to be able to con-tinuously train the model as new data arrives.  This new data can contain un-seen behaviour for the model. It is desirable to make the model learn this newbehaviour without forgetting about the previously learnt experiences. To dealwith this problem, we will use a technique called Elastic Weight Consolidation.


<p align="center"><img src="https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Images/EWC1.PNG"  align=middle width=345pt />
</p>
<p align="center">
<em>Representation of Elastic Weight Consolidation. The goal is to find a combination of network parameters which is suitable to perform both tasks A and B.</em>
</p>


## Notebooks
(Currently tested on TensorFlow 2.0.0/2.1.0)

### [Bearing failure detection using one-dimensional Convolutional Autoencoder](https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Bearing_Fault_Detection.ipynb) 
In this network we show how to perform early detection of bearing failures using an unsupervised method. The results of this notebook will be used to put the labels for the classification model.

<sub>*NOTE*: This notebook contains interactive plots created with *plotly* library. These plots are not displayed in Github viewer. To see them, either download the notebook and open it locally (and ensure the notebook is in trusted mode), or use [nbviewer](https://nbviewer.jupyter.org/) to visualize it online. </sub>

### [Siamese network for the classification of bearing failures](https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Siamese.ipynb)
In this notebook we use a Siamese network trained using contrastive loss to classify the root cause of the bearing failures. 

### [Triplet learning strataefies for the classification of bearing failures](https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Triplet_learning.ipynb)
In this notebook we use different triplet learning strategies to classify the root cause of bearing failures.

### [Elastic weight consolidation - concept drift](https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/EWC-concept-drift.ipynb)
In this notebook we use Elastic Weight Consolidation to train the triplet learning classification model on two very different data sets: IMS and Paderborn. Since the data sets are quite different, the change of data simulates concept drift.

### [Elastic weight consolidation - new behaviour](https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/EWC-new_behaviour.ipynb)
In this notebook we use Elastic Weight Consolidation to train the triplet learning classification model on two similar data sets: CWRU and CWRU with noise. Since the data sets are similar, we simulate seeing a new behaviour of the same data set. 


## Contributions

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/laiadc/PFM_Bearing_Fault_Detection/issues).

## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [laiadc](https://github.com/laiadc)
* Email: [ldomingoc@gmail.com](ldomingoc@gmail.com)

### BibTex reference format for citation for the Code
```
@misc{BearingsDomingo,
title={Deep learning for predictive maintenance of rolling bearings},
url={https://github.com/laiadc/PFM_Bearing_Fault_Detection/},
note={GitHub repository containing deep learning approach for detecting early development of rolling bearing failures and their root cause.},
author={Laia Domingo Colomer},
  year={2020}
}
```
### BibTex reference format for citation for the report of the Master's Thesis

```
@misc{BearingsDomingoMasterThesis,
title={Deep learning for predictive maintenance of rolling bearings},
url={https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Master_Thesis_Report.pdf},
note={Report of the Master's Thesis: Deep learning for predictive maintenance of rolling bearings.},
author={Laia Domingo Colomer},
  year={2020}
}
```


