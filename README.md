# ChangeViT
Codes and models for ***[ChangeViT: Unleashing Plain Vision Transformers for Change Detection. (arXiv preprint)](https://arxiv.org/pdf/2406.12847)***

# Update
- [2024/6/19] The training code will be public available at about 2024/7/5.

# Abstract
Change detection in remote sensing images is essential for tracking environmental changes on the Earth’s surface. Despite the success of vision transformers (ViTs) as backbones in numerous computer vision applications, they remain underutilized in change detection, where convolutional neural networks (CNNs) continue to dominate due to their powerful feature extraction capabilities. In this paper, our study uncovers ViTs’ unique advantage in discerning large-scale changes, a capability where CNNs fall short. Capitalizing on this insight, we introduce ChangeViT, a framework that adopts a plain ViT backbone to enhance the performance of large-scale changes. This framework is supplemented by a detail-capture module that generates detailed spatial features and a feature injector that efficiently integrates fine-grained spatial information into high-level semantic learning. The feature integration ensures that ChangeViT excels in both detecting large-scale changes and capturing fine-grained details, providing comprehensive change detection across diverse scales. Without bells and whistles, ChangeViT achieves state-of-the-art performance on three popular high-resolution datasets (i.e., LEVIR-CD, WHU-CD, and CLCD) and one low-resolution dataset (i.e., OSCD), which underscores the unleashed potential of plain ViTs for change detection. Furthermore, thorough quantitative and qualitative analyses validate the efficacy of the introduced modules, solidifying the effectiveness of our approach.



# Citation

