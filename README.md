<p align="center" width="100%">
<img src="./pandagpt.png" alt="PandaGPT-4" style="width: 40%; min-width: 300px; display: block; margin: auto;">
</p>

# PandaGPT: Empowering Large Language Models with Visual and Auditory Intelligence

![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Model Weight License](https://img.shields.io/badge/Model_Weight%20License-CC%20By%20NC%204.0-red.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)


**Team:** [Yixuan Su](https://yxuansu.github.io/), [Tian Lan](https://github.com/gmftbyGMFTBY), [Huayang Li](https://sites.google.com/view/huayangli), [Deng Cai](https://jcyk.github.io/) 


This repo contains related resources of PandaGPT.

This repo contains
- The <a href='#weights'>weights</a> for the fine-tuned model.
- The <a href='#data'>data</a> used for fine-tuning the model.
- The <a href='#example_usage'>example usage</a> of OpenAlpaca.
- The <a href='#code'>code</a> for fine-tuning the model.

**Usage and License Notices:**


****

<span id='weights'/>

# Model Weights:

|**Model Size**|**Learning Tasks**|**Delta Weights Address**|
|:-------------:|:-------------:|:-------------:|
|7B|Image Captioning|[openllmplayground/pandagpt_7b_v0_image_captioning_only](https://huggingface.co/openllmplayground/pandagpt_7b_v0_image_captioning_only)|
|7B|Visual Instruction||
|7B|Image Captioning + Visual Instruction||
|13B|Image Captioning||
|13B|Visual Instruction||
|13B|Image Captioning + Visual Instruction||

|**Model Name**|**Model Card**|**Model Description**|
|:-------------:|:-------------:|:-------------:|
|`openllmplayground/openalpaca_7b_preview_2bt`|[[Link]](https://huggingface.co/openllmplayground/openalpaca_7b_preview_2bt/)|```The OpenAlpaca model fine-tuned from the previewed version of OpenLLaMA that is trained with 200 billion tokens.```|
|`openllmplayground/openalpaca_7b_preview_3bt`|[[Link]](https://huggingface.co/openllmplayground/openalpaca_7b_preview_3bt/)|```The OpenAlpaca model fine-tuned from the previewed version of OpenLLaMA that is trained with 300 billion tokens.```|
