# GlobEnc
[NAACL 2022] GlobEnc: Quantifying Global Token Attribution by Incorporating the Whole Encoder Layer in Transformers

Check this Colab notebook to see what BERT attends to <br>
<a href="https://colab.research.google.com/github/mohsenfayyaz/GlobEnc/blob/main/GlobEnc_Demo.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>


## Abstract
> There has been a growing interest in interpreting the underlying dynamics of Transformers. While self-attention patterns were initially deemed as the primary choice, recent studies have shown that integrating other components can yield more accurate explanations. This paper introduces a novel token attribution analysis method that incorporates all the components in the encoder block and aggregates this throughout layers. We quantitatively and qualitatively demonstrate that our method can yield faithful and meaningful global token attributions. Our extensive experiments reveal that incorporating almost every encoder component results in increasingly more accurate analysis in both local (single layer) and global (the whole model) settings. Our global attribution analysis surpasses previous methods by achieving significantly higher results in various datasets.


## Setup
```
conda create -n attention-env
pip install pip-tools
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
