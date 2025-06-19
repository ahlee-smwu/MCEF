# MCEF: Multimodal Correlated Equal Fusion Network for Emotion Recognition in Utterance Level

We propose a MCEF that equally integrates each modality and implement a 2 loss system that considers both intra- and inter-modal characteristics.

In this research, we propose the \textbf{MCEF (Multimodal Correlated Equal Fusion)} network for utterance-level emotion recognition, designed to address these limitations. Our model equally integrates text, audio, and visual modalities using an equal cross-attention fusion mechanism that ensures no single modality dominates the fusion process. Additionally, we introduce a two-loss system comprising an intra-modal classification loss and an inter-modal correlation loss based on Soft-HGR, which enhances both modality-specific representation learning and cross-modal feature alignment.
Experimental results on the IEMOCAP dataset demonstrate the effectiveness of our approach. The proposed model achieves an accuracy of 70.98%, which is comparable to the performance of state-of-the-art methods, despite using significantly fewer parameters—only 20.7M, which is approximately 25.97% to 60% of the size of competitive models. These results highlight the potential of MCEF as a lightweight yet effective framework for balanced multimodal emotion recognition.

This repository is based on SDT model and source.

## Model Architecture
<!-- ![Image of SDT](fig/SDT.jpg) -->
<div align="center">
    <img src="fig/SDT.jpg" width="85%" title="SDT."</img>
</div>

## Setup
- Check the packages needed or simply run the command:
```console

pip install -r requirements.txt
```
- Download the preprocessed datasets from [here](https://drive.google.com/drive/folders/1J1mvbqQmVodNBzbiOIxRiWOtkP6qqP-K?usp=sharing), and put them into `data/`.

## Run model
- Run the model on IEMOCAP dataset:
```console

bash exec_iemocap.sh
```

## Acknowledgements
- Special thanks to the [COSMIC](https://github.com/declare-lab/conv-emotion) and [MMGCN](https://github.com/hujingwen6666/MMGCN) for sharing their codes and datasets.

## Citation
