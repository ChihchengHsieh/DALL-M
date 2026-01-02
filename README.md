# Data augmentation with LLMs (DALL-M)
MMTF is in another [anonymised repo](https://anonymous.4open.science/r/MMTF-0071/README.md).


# Motivation
The motivation to augment clinical dataset is from the attempt of multimodal contrastive learning from [[Best of Both Worlds: Multimodal Contrastive Learning with Tabular and Imaging Data]](https://arxiv.org/abs/2303.14080), where we employed the same strategy but found no significant improvement on both classification and detection tasks, as shown in the following tables. We believe this behaviour is attributed to lack of clinical features, when we only have 9 clinical features available on REFLACX dataset, and the work used 120 clinical features from UK Biobank.

## Classification (CheXpert)
|Weights|Deployement Strategy|F1|Precision|Accuracy|Recall|AUC|
|:---|:---|---:|---:|---:|---:|---:|
|Mutlimodal Contrastive Learning|Linear Evaluation|0.3844|0.6583|0.8808|0.2714|0.6245|
|Mutlimodal Contrastive Learning|Linear Evaluation <br> (fix first 2 layers)|**0.5139**|0.6818|**0.8930**|*0.4124*|**0.6909**|
|Mutlimodal Contrastive Learning|Linear Evaluation <br> (for first 20 epochs)|*0.5098*|0.6588|0.8904|**0.4158**|*0.6908*|
|Mutlimodal Contrastive Learning|Fine-tuned|0.5021|0.6783|*0.8916*|0.3986|0.6843|
|ImageNet|Linear Evaluation|0.3600|0.6100|0.8755|0.2554|0.6147|
|ImageNet|Linear Evaluation <br> (fix first 2 layers)|0.4742|0.6832|0.8896|0.3631|0.6682|
|ImageNet|Linear Evaluation <br> (for first 20 epochs)|0.4866|*0.6943*|*0.8916*|0.3746|0.6741|
|ImageNet|Fine-tuned|0.4872|0.6741|0.8900|0.3814|0.6760|
|Random Initialisation|N/A|0.3524|**0.7276**|0.8829|0.2325|0.6094|

## Detection (REFLACX)
|Weights|Deployement Strategy|mAP|mAR|
|:---|:---|---:|---:|
|Mutlimodal Contrastive Learning|Linear Evaluation|0.0787|0.4175|
|Mutlimodal Contrastive Learning|Linear Evaluation <br> (fix first 2 layers)|0.1065|0.4989|
|Mutlimodal Contrastive Learning|Fine-tuned|Serious<br>Overfitting||
|ImageNet|Linear Evaluation|0.0970|0.4153|
|ImageNet|Linear Evaluation <br> (fix first 2 layers)|**0.1142**|**0.5371**|
|ImageNet|Fine-tuned|Serious<br>Overfitting||
|Random Initialisation|N/A|*0.1125*|*0.4268*|

{Context to add after publication}