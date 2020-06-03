# Deep learning for predictive maintenance of rolling bearings

A deep learning approach for detecting early development of rolling bearing failures and their root cause.

This repository contains the code used to monitor the health of rolling bearings. The results are illustrated in a set of [Jupyter](https://jupyter.org/) notebooks. The whole code is the result of a Master's Thesis of the [Master's Degree in the
Fundamental Principles of Data Science](http://www.ub.edu/datascience/master/) supervised by Dr. Jordi Vitrià. This repository also includes the Master's Thesis report, in pdf format. Any contribution or idea to continue the lines of the proposed work will be very welcome.

In order to perform predictive maintenance of rolling bearings, the project is split into three parts:

+ **Early detection of the failure**: It is important to detect and classify bearing failures at the early stages of their development so that one cantake appropriate actions before the machine breaks.  For this reason, an early detection model is developed. This model is based on a one-dimensional convolutional autoencoder, which is an unsupervised method which detects if there has been a deviation from the normal (healthy) behaviour of the machine.

<p align="center"><img src="https://github.com/laiadc/PFM_Bearing_Fault_Detection/blob/master/Images/ae.png"  align=middle width=445pt />
</p>
<p align="center">
<em>Representation of a one-dimensional convolutional autoencoder.  The autoencoder is trained only with healthy data so that it learns to reconstruct only the healthy waveforms. The reconstruction error is taken as an indicator of anomalous (unhealthy) behaviour. </em>
</p>

+ **Classification of the failure**:  a classification model is trained to determine the position (inner ring, outer ring, rolling elementor cage) of the bearing failure. This model is trained so that it is robust to different operation conditions (such as the rotation velocity) and so that it can be trained with few data samples. In order to accomplish these requirements, a convolutional neural network is  trained using triplet learning strategies or using a Siamese network with contrastive loss.

+ **Continuous learning of the models**: Usually,  all the historical data is not provided at the training stage,  but it isacquired progressively with time. Therefore, it is interesting to be able to con-tinuously train the model as new data arrives.  This new data can contain un-seen behaviour for the model. It is desirable to make the model learn this newbehaviour without forgetting about the previously learnt experiences. To dealwith this problem, we will use a technique called Elastic Weight Consolidation.
