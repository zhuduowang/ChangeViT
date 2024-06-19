## ChangeViT
Codes and models for ***[ChangeViT: Unleashing Plain Vision Transformers for Change Detection ](https://arxiv.org/pdf/2406.12847).***

[Duowang Zhu](https://scholar.google.com/citations?user=9qk9xhoAAAAJ&hl=en&oi=ao), [Xiaohu Huang](https://scholar.google.com/citations?user=sBjFwuQAAAAJ&hl=en&oi=ao), Haiyan Huang, Zhenfeng Shao, Qimin Cheng

[[paper]](https://arxiv.org/pdf/2406.12847)

## Update
- [2024/6/19] The training code will be publicly available at about 2024/7/5.

## Abstract
In this paper, our study uncovers ViTs' unique advantage in discerning large-scale changes, a capability where CNNs fall short. Capitalizing on this insight, we introduce ChangeViT, a framework that adopts a plain ViT backbone to enhance the performance of large-scale changes. This framework is supplemented by a detail-capture module that generates detailed spatial features and a feature injector that efficiently integrates fine-grained spatial information into high-level semantic learning. The feature integration ensures that ChangeViT excels in both detecting large-scale changes and capturing fine-grained details, providing comprehensive change detection across diverse scales. Without bells and whistles, ChangeViT achieves state-of-the-art performance on three popular high-resolution datasets (i.e., LEVIR-CD, WHU-CD, and CLCD) and one low-resolution dataset (i.e., OSCD), which underscores the unleashed potential of plain ViTs for change detection. Furthermore, thorough quantitative and qualitative analyses validate the efficacy of the introduced modules, solidifying the effectiveness of our approach.

## Framework
<p align="center">
    <img width=800 src="figures/framework.png"/> <br />
</p>

Figure 1. Overview of the proposed $\textbf{ChangeViT}$. bi-temporal images $I_{1}$ and $I_{2}$ are firstly fed into shared ViT to extract high-level semantic features and detail-capture module to extract low-level detailed information. Subsequently, a feature injector is introduced to inject the low-level details into high-level features. Finally, a decoder is utilized to predict changed probability maps.


## Performance
<table>
    <caption>
Performance comparison of different change detection methods on LEVIR-CD, WHU-CD, and CLCD datasets, respectively. The best results are highlighted in <span class="bold">bold</span> and the second best results are <span class="underline">underlined</span>. All results of the three evaluation metrics are described as percentages (%).
    </caption>
    <thead>
        <tr>
            <th rowspan="2">Method</th>
            <th rowspan="2">#Params(M)</th>
            <th rowspan="2">FLOPs(G)</th>
            <th colspan="3">LEVIR-CD</th>
            <th colspan="3">WHU-CD</th>
            <th colspan="3">CLCD</th>
        </tr>
        <tr>
            <th>F1</th>
            <th>IoU</th>
            <th>OA</th>
            <th>F1</th>
            <th>IoU</th>
            <th>OA</th>
            <th>F1</th>
            <th>IoU</th>
            <th>OA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>DTCDSCN</th>
            <th>$41.07$</th>
            <th>$20.44$</th>
            <th>$87.43$</th>
            <th>$77.67$</th>
            <th>$98.75$</th>
            <th>$79.92$</th>
            <th>$66.56$</th>
            <th>$98.05$</th>
            <th>$57.47$</th>
            <th>$40.81$</th>
            <th>$94.59$</th>
        </tr>
        <tr>
            <th>SNUNet</th>
            <th>$12.04$</th>
            <th>$54.82$</th>
            <th>$88.16$</th>
            <th>$78.83$</th>
            <th>$98.82$</th>
            <th>$83.22$</th>
            <th>$71.26$</th>
            <th>$98.44$</th>
            <th>$60.82$</th>
            <th>$43.63$</th>
            <th>$94.90$</th>
        </tr>
        <tr>
            <th>ChangeFormer</th>
            <th>$41.03$</th>
            <th>$202.79$</th>
            <th>$90.40$</th>
            <th>$82.48$</th>
            <th>$99.04$</th>
            <th>$87.39$</th>
            <th>$77.61$</th>
            <th>$99.11$</th>
            <th>$61.31$</th>
            <th>$44.29$</th>
            <th>$94.98$</th>
        </tr>
        <tr>
            <th>BIT</th>
            <th>$\textbf{3.55}$</th>
            <th>$\textbf{10.63}$</th>
            <th>$89.31$</th>
            <th>$80.68$</th>
            <th>$98.92$</th>
            <th>$83.98$</th>
            <th>$72.39$</th>
            <th>$98.52$</th>
            <th>$59.93$</th>
            <th>$42.12$</th>
            <th>$94.77$</th>
        </tr>
        <tr>
            <th>ICIFNet</th>
            <th>$23.82$</th>
            <th>$25.36$</th>
            <th>$89.96$</th>
            <th>$81.75$</th>
            <th>$98.99$</th>
            <th>$88.32$</th>
            <th>$79.24$</th>
            <th>$98.96$</th>
            <th>$68.66$</th>
            <th>$52.27$</th>
            <th>$95.77$</th>
        </tr>
        <tr>
            <th>DMINet</th>
            <th>$\underline{6.24}$</th>
            <th>$\underline{14.42}$</th>
            <th>$90.71$</th>
            <th>$82.99$</th>
            <th>$99.07$</th>
            <th>$88.69$</th>
            <th>$79.68$</th>
            <th>$98.97$</th>
            <th>$67.24$</th>
            <th>$50.65$</th>
            <th>$95.21$</th>
        </tr>
        <tr>
            <th>GASNet</th>
            <th>$23.59$</th>
            <th>$23.52$</th>
            <th>$90.52$</th>
            <th>$83.48$</th>
            <th>$99.07$</th>
            <th>$91.75$</th>
            <th>$84.76$</th>
            <th>$99.34$</th>
            <th>$63.84$</th>
            <th>$46.89$</th>
            <th>$94.01$</th>
        </tr>
        <tr>
            <th>AMTNet</th>
            <th>$24.67$</th>
            <th>$21.56$</th>
            <th>$90.76$</th>
            <th>$83.08$</th>
            <th>$98.96$</th>
            <th>$92.27$</th>
            <th>$85.64$</th>
            <th>$99.32$</th>
            <th>$75.10$</th>
            <th>$60.13$</th>
            <th>$96.45$</th>
        </tr>
        <tr>
            <th>EATDer</th>
            <th>$6.61$</th>
            <th>$23.43$</th>
            <th>$91.20$</th>
            <th>$83.80$</th>
            <th>$98.75$</th>
            <th>$90.01$</th>
            <th>$81.97$</th>
            <th>$98.58$</th>
            <th>$72.01$</th>
            <th>$56.19$</th>
            <th>$96.11$</th>
        </tr>
        <tr>
            <th>ChangeViT-T (Ours)</th>
            <th>$11.68$</th>
            <th>$27.15$</th>
            <th>$\underline{91.81}$</th>
            <th>$\underline{84.86}$</th>
            <th>$\underline{99.17}$</th>
            <th>$\underline{94.53}$</th>
            <th>$\underline{89.63}$</th>
            <th>$\underline{99.57}$</th>
            <th>$\underline{77.31}$</th>
            <th>$\underline{63.01}$</th>
            <th>$\underline{96.67}$</th>
        </tr>
        <tr>
            <th>ChangeViT-S (Ours)</th>
            <th>$32.13$</th>
            <th>$38.80$</th>
            <th>$\textbf{91.98}$</th>
            <th>$\textbf{85.16}$</th>
            <th>$\textbf{99.19}$</th>
            <th>$\textbf{94.84}$</th>
            <th>$\textbf{90.18}$</th>
            <th>$\textbf{99.59}$</th>
            <th>$\textbf{77.57}$</th>
            <th>$\textbf{63.36}$</th>
            <th>$\textbf{96.79}$</th>
        </tr>
    </tbody>
</table>


## Usage

### Data Preparation
- Download the [LEVIR-CD](https://chenhao.in/LEVIR/), [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html), [CLCD](https://github.com/liumency/CropLand-CD), and [OSCD](https://rcdaudt.github.io/oscd/) datasets.

- Crop each image in the dataset into 256x256 patches.

- Prepare the dataset into the following structure and set its path in the configuration file.
    ```
    ├─Train
        ├─A          jpg/png
        ├─B          jpg/png
        └─label      jpg/png
    ├─Val
        ├─A 
        ├─B
        └─label
    ├─Test
        ├─A
        ├─B
        └─label
    ```

### Download checkpoint


## Training


## Inference


## Citation
```bibtex
@article{zhu2024changevit,
  title={ChangeViT: Unleashing Plain Vision Transformers for Change Detection},
  author={Duowang Zhu, Xiaohu Huang, Haiyan Huang, Zhenfeng Shao, and Qimin Cheng},
  journal={arXiv preprint arXiv:2406.12847},
  year={2024}
}
```
