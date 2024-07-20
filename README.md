
# Tips
After the paper is fully accepted, we will publicly release all the code, including the training code.

# Introdunction
This is the official implementation of the paper "Learnable Cross-scale Sparse Attention Guided Feature Fusion for UAV Object Detection".

## Abstract
Object detection in Unmanned Aerial Vehicle (UAVs) faces a significant challenge in computer vision. Traditional methods are difficult to model object appearance feature with large scale variations and viewpoint differences, when drones fly at different altitudes and capture images from diverse shooting angles. To address this issue, we propose a Learnable Cross-scale Sparse Attention (LCSA) guided  feature fusion method to improve the performance of UAV object detection. Specifically, the LCSA feature fusion module enables each point in a feature map to aggregate discriminative information from a set of points with learnable offsets in neighbor feature maps. It enhances local discriminative features of the object by facilitating semantic information interaction across multiple feature maps. The LCSA can function as a novel neck method that complements the existing neck methods and is also transplantable to different object detection frameworks. Moreover, we also employ a scale-aware loss function to integrate the normalized Wasserstein distance with CIoU in order to improve the incompatibility of IoU for objects with large scale variance. Experimental results on the SeaDroneSeev2 and VisDrone2019-DET datasets show that the proposed method achieves superior performance. 
At a resolution of 640*640, our method achieves 81.9% AP50 and 47.4% AP on SeaDroneSeev2, surpassing baseline 4.9% and 4.8%, achieves state-of-the-art performance. Furthermore, our method outperforms baseline by 5% AP on VisDrone2019-DET.


## Results on Seadronesee object detectionV2:



# Quick start


### Dataset

