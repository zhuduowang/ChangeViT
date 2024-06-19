## ChangeViT
Codes and models for ***[ChangeViT: Unleashing Plain Vision Transformers for Change Detection ](https://arxiv.org/pdf/2406.12847).***

[Duowang Zhu](https://scholar.google.com/citations?user=9qk9xhoAAAAJ&hl=en&oi=ao), [Xiaohu Huang](https://scholar.google.com/citations?user=sBjFwuQAAAAJ&hl=en&oi=ao), Haiyan Huang, Zhenfeng Shao, Qimin Cheng

[[paper]](https://arxiv.org/pdf/2406.12847)

## Update
- [2024/6/19] The training code will be publicly available at about 2024/7/5.

## Abstract
In this paper, our study uncovers ViTs' unique advantage in discerning large-scale changes, a capability where CNNs fall short. Capitalizing on this insight, we introduce ChangeViT, a framework that adopts a plain ViT backbone to enhance the performance of large-scale changes. This framework is supplemented by a detail-capture module that generates detailed spatial features and a feature injector that efficiently integrates fine-grained spatial information into high-level semantic learning. The feature integration ensures that ChangeViT excels in both detecting large-scale changes and capturing fine-grained details, providing comprehensive change detection across diverse scales. Without bells and whistles, ChangeViT achieves state-of-the-art performance on three popular high-resolution datasets (i.e., LEVIR-CD, WHU-CD, and CLCD) and one low-resolution dataset (i.e., OSCD), which underscores the unleashed potential of plain ViTs for change detection. Furthermore, thorough quantitative and qualitative analyses validate the efficacy of the introduced modules, solidifying the effectiveness of our approach.

## Framework

## Performance

<table style="border-collapse: collapse; border: none; border-spacing: 0px;">
	<caption>
		Table 1. Performance comparison of different change detection methods on LEVIR-CD, WHU-CD, and CLCD datasets, respectively. The best results are highlighted in <b>bold</b> and the second best results are <u>underlined</u>. All results of the three evaluation metrics are described as percentages (%).
	</caption>
	<tr>
		<td rowspan="2" style="border-right: 1px solid black; border-top: 2px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Method
		</td>
		<td rowspan="2" style="border-right: 1px solid black; border-top: 2px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			#Params(M)
		</td>
		<td rowspan="2" style="border-right: 1px solid black; border-top: 2px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			FLOPs(G)
		</td>
		<td colspan="3" style="border-right: 1px solid black; border-top: 2px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			LEVIR-CD
		</td>
		<td colspan="3" style="border-right: 1px solid black; border-top: 2px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			WHU-CD
		</td>
		<td colspan="3" style="border-top: 2px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			CLCD
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			IoU
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			OA
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			IoU
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			OA
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			IoU
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			OA
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			DTCDSCN
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			41.07
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			20.44
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			87.43
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			77.67
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.75
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			79.92
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			66.56
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.05
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			57.47
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			40.81
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.59
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			SNUNet
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			12.04
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			54.82
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			88.16
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			78.83
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.82
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			83.22
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			71.26
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.44
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			60.82
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			43.63
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.90
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			ChangeFormer
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			41.03
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			202.79
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			90.40
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			82.48
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			99.04
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			87.39
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			77.61
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			99.11
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			61.31
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			44.29
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.98
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			BIT
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>3.55</b>
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>10.63</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			89.31
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			80.68
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.92
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			83.98
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			72.39
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.52
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			59.93
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			42.12
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.77
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			ICIFNet
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			23.82
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			25.36
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			89.96
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			81.75
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.99
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			88.32
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			79.24
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.96
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			68.66
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			52.27
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			95.77
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			DMINet
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>6.24</u>
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>14.42</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			90.71
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			82.99
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			99.07
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			88.69
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			79.68
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.97
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			67.24
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			50.65
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			95.21
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			GASNet
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			23.59
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			23.52
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			90.52
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			83.48
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			99.07
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			91.75
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			84.76
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			99.34
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			63.84
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			46.89
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.01
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			AMTNet
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			24.67
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			21.56
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			90.76
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			83.08
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.96
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			92.27
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			85.64
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			99.32
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			75.10
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			60.13
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			96.45
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			EATDer
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			6.61
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			23.43
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			91.20
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			83.80
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.75
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			90.01
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			81.97
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.58
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			72.01
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			56.19
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			96.11
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ChangeViT-T</b>
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			11.68
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			27.15
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>91.81</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>84.86</u>
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>99.17</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>94.53</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>89.63</u>
		</td>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>99.57</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>77.31</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>63.01</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>96.67</u>
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ChangeViT-S</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			32.13
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			38.80
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>91.98</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>85.16</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>99.19</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>94.84</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>90.18</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>99.59</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>77.57</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>63.36</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>96.79</b>
		</td>
	</tr>
</table>

<table style="border-collapse: collapse; border: none; border-spacing: 0px;">
	<caption>
		Table 2. Performance comparison of different change detection methods on the OSCD dataset. The best results are highlighted in <b>bold</b> and the second best results are <u>underlined</u>. All results of the three evaluation metrics are described as percentages (%).
	</caption>
	<tr>
		<td rowspan="2" style="border-right: 1px solid black; border-top: 2px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Method
		</td>
		<td colspan="3" style="border-top: 2px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			OSCD
		</td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			F1
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			IoU
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			OA
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			DTCDSCN
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			36.13
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			22.05
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.50
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			SNUNet
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			27.02
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			15.62
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			93.81
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			ChangeFormer
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			38.22
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			23.62
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.53
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			BIT
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			29.58
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			17.36
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			90.15
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			ICIFNet
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			23.03
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			13.02
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.61
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			DMINet
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			42.23
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			26.76
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			95.00
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			GASNet
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			10.71
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			5.66
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			91.52
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			AMTNet
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			10.25
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			5.40
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.29
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			EATDer
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			54.23
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			36.98
		</td>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			93.85
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ChangeViT-T</b>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>55.13</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>38.06</u>
		</td>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<u>95.01</u>
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>ChangeViT-S</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>55.51</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>38.42</b>
		</td>
		<td style="border-bottom: 2px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>95.05</b>
		</td>
	</tr>
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
