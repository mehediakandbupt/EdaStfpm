# Efficient Dual Attention Based Knowledge  Distillation Network for Unsupervised  Wafer Map Anomaly Detection

## Abstract:
Detecting wafer map anomalies is crucial for preventing yield loss in semiconductor fabrication, although intricate patterns and resource-intensive labeled data prerequisites hinder precise deep-learning segmentation. This paper presents an innovative, unsupervised method for detecting pixel-level anomalies in wafer maps. It utilizes an efficient dual attention module with a knowledge distillation network to learn defect distributions without anomalies. Knowledge transfer is achieved by distilling information from a pre-trained teacher into a student network with similar architecture, except an efficient dual attention module is incorporated atop the teacher network’s feature pyramid hierarchies, which enhances feature representation and segmentation across pyramid hierarchies that selectively emphasize relevant and discard irrelevant features by capturing contextual associations in positional and channel dimensions. Furthermore, it enables student networks to acquire an improved knowledge of hierarchical features to identify anomalies across different scales accurately. The dissimilarity in feature pyramids acts as a discriminatory function, predicting the likelihood of an abnormality, resulting in highly accurate pixel-level anomaly detection. Consequently, our proposed method excelled on the WM-811K and MixedWM38 datasets, achieving AUROC, AUPR, AUPRO, and F1-Scores of (99.65%, 99.35%), (97.31%, 92.13%), (90.76%, 84.66%) respectively, alongside an inference speed of 3.204 FPS, showcasing its high precision and efficiency.

### please follow the new Releases form Anomalib repo [https://github.com/openvinotoolkit/anomalib] to update the code if necessary. the version is used in this code is an v0.7.0. Also for feature extraction there has some changes in the repo for add functionality.


Article Link: https://ieeexplore.ieee.org/abstract/document/10560027


Please cite this if you find this useful, Thanks.
#### bib:
@ARTICLE{10560027,
  author={Hasan, Mohammad Mehedi and Yu, Naigong and Mirani, Imran Khan},
  journal={IEEE Transactions on Semiconductor Manufacturing}, 
  title={Efficient Dual Attention Based Knowledge Distillation Network for Unsupervised Wafer Map Anomaly Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Image segmentation;Anomaly detection;Semiconductor device modeling;Knowledge engineering;Training;Fabrication;Wafer map anomaly detection;Knowledge distillation;Attention network},
  doi={10.1109/TSM.2024.3416055}}
#### or:
M. M. Hasan, N. Yu and I. Khan Mirani, "Efficient Dual-Attention-Based Knowledge Distillation Network for Unsupervised Wafer Map Anomaly Detection," in IEEE Transactions on Semiconductor Manufacturing, vol. 37, no. 3, pp. 293-303, Aug. 2024, doi: 10.1109/TSM.2024.3416055. keywords: {Feature extraction;Image segmentation;Anomaly detection;Semiconductor device modeling;Training;Fabrication;Knowledge transfer;Wafer map anomaly detection;knowledge distillation;attention network},
