# Emotion Recognition Master's Thesis
Recognizing Emotions from EEG data with **vision transformers** and **continuous wavelet transform (CWT)**

<img src="figures/cwt.gif" alt="cwt" width="500"/>

###### *Note: this is currently still a work in progress -- code and results are not final!*

## Background
Emotion recognition is regarded as an important topic in the field of affective computing. The ability to accurately classify emotions from brain data would have potential for applications in BCIs, human-machine interaction, psychotherapy, and medicine.

As the capability of machine learning models have improved in recent years, the feasibility of developing an effective emotion recognition model has grown alongside it, which has brought more attention to the topic as of late. Previous attempts at emotion recognition utilized both conventional machine learning approaches (KNN, SVMs, RF), and deep learning approaches (RNNs, CNNs, GCNNs). Most recently, *transformers* have shown the most promising improvements over previous methods.

Approaches that utilize transformers vary in terms of model architecture and features extracted from EEG data. For example:

Model | Features | Architecture
 --- | --- | ---
SAG-CET | spatiotemporal | TF + GCNN
STS-TF | spatiotemporal | TF
ERTNet | spatiotemporal | TF + CNN
SECT | DE, PSD | TF
MACTN | temporal | TF + CNN

Of these approaches, the **vision transformer** and **CWT** are the least explored architecture and feature combination. Interestingly, this combination also shows promise, since CWT provides a visual input that lends itself to a vision transformer.

<img src="figures/fcwt.png" alt="fcwt" width="500"/>

This thesis demonstrates an approach to emotion recognition using vision transformers trained on CWT features extracted from EEG data.

## Methods
In this project, a 3D vision transformer is trained to recognize emotions from EEG data by finetuning in *two phases*: first on the public DEAP dataset, and again on a private dataset.
### Data and preprocessing
The first dataset used is a preprocessed version of the [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) dataset. This dataset provides 32-channel EEG recordings of participants' responses to emotional stimuli. These stimuli were also rated on continuous scales of valence and arousal (1 - 9). With the valence value, the data can be split into 3 distinct classes by using thresholds of 3 and 6:
* **Unpleasant**: valence < 3
* **Neutral**: 3 <= valence < 6
* **Pleasant**: 6 <= valence <= 9
<img src="figures/circumplex.png" alt="circumplex" width="500">

The second dataset used is a [private dataset](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14765) that contains 128-channel EEG recordings of reponses to both image and video stimuli collected at the University of Georgia. All stimuli were hand-selected to illicit unpleasant, neutral, or pleasant responses, and these contenet categories were used as emotional classes during model training. 

### Feature Extraction
CWT features were extracted using the [fCWT](https://github.com/fastlib/fCWT) library. These features were then reoriented to create "frames" that display CWT values for each channel at one point in time.

<img src="figures/example_frame.png" alt="frame" width="500"/>

These frames are then stacked together to create "video" inputs for a 3D vision transformer model (an example of which can be seen at the top of the page). These videos were also augmented using Gaussian noise to increase the number of samples used to train the model.

#### Determining Channel Order with Hilbert Curves
Transformers are highly dependent on the sequence of the input data. Vision transformers and video vision transformers learn from spatial relationships between patches in the same way that transformers in NLP learn from the order of input tokens. In the current project, it is important to ensure that EEG channels are ordered in a way that maintains spatial continuity between electrodes, so that spatial relationships between pixels correspond to spatial relationships in the real world. Furthermore, since the model is finetuned using multiple datasets with varying numbers of channels, it is also important to ensure that relative positions in the frame always correspond to similar brain regions, regardless of the resolution. Otherwise, patterns learned from one dataset would become meaningless as the model was trained and evaluated on a different dataset with a different arrangement. This creates the need for a general method for obtaining similar orders from any number of EEG channels. Doing so would ensure that relative positions in those orders would always correspond to similar brain regions.

To the best of my knowledge, because there are no cases in the literature in which a model is finetuned with multiple datasets that have different numbers of channels, this is the first time this problem has been encountered.

Solving this issue requires a method that can convert 2D locations of electrodes placed on a scalp to a 1D list of channels. In mathematics, this problem is solved with Hilbert curves, which can be used to fill a square space. In other words, a Hilbert curve can map all points in 2D space onto a 1D line while maintaining continuity. In this case, a Hilbert curve can be used to traverse a 2D matrix containing either 32 or 128 EEG channels. As the curve traverses the matrix, it visits each brain region in a similar order, and adds the names of electrodes to a list as it encounters them along its path. During feature extraction, these lists are used to reorder the channels in both datasets before CWT values are calculated. An example is provided below for the 128-channel private dataset:


### Model and training
The model is an implementation of a 3D vision transformer obtained from the [vit-pytorch](https://github.com/lucidrains/vit-pytorch#3d-vit) library:
```python
from vit_pytorch.vit_3d import ViT

vit = ViT( # vision transformer parameters follow suggestions from Awan et al. 2024
    image_size=32,
    frames=768,
    image_patch_size=16,
    frame_patch_size=96,
    num_classes=4,
    dim=768,
    depth=16,
    heads=16,
    mlp_dim=1024,
    channels=3,
    dropout=0.5,
    emb_dropout=0.1,
    pool='mean'
)
```
A unique model was created for each of the 32 subjects present in the DEAP dataset. Each model was trained for 50 epochs on an 80% split of the data containing both original and augmented samples. This split was also stratified by class of emotion. Accuracy metrics were recorded for valence and arousal individually, along with overall accuracy. 

## Results (so far) 
Metric | Valence | Arousal | Overall
--- | --- | --- | ---
*subject-dependent models average accuracies (95% CI)* | (98.57, 99.28)% | (98.61, 99.38)% | (97.87, 98.93)%
*cross-subject model accuracies* | 99.84% | 99.88% | 99.73%

## Plans
This is still a work in progress. There are a number of things that I am still currently working on. A (non-comprehensive) to-do list is below:
- [x] Train a cross-subject model (model trained on data from all participants)
- [x] Confusion matrix for cross subject model
- [ ] Other classification metrics for all models (precision, recall, f1, etc.)
- [ ] Finetune cross-subject model on private dataset

## Other info
This thesis will be submitted to the graduate faculty of The University of Georgia in partial fulfillment of the requirements for the degree Master of Science in Artificial Intelligence.

### Acknowledgments
I would like to thank UGA's Institute for Artificial Intelligence for the support provided during the writing of this thesis. It wouldn't have been possible without the resources and guidance they've given me.

### Packages and data:
[fCWT](https://github.com/fastlib/fCWT)

[vit-pytorch](https://github.com/lucidrains/vit-pytorch#3d-vit)

[DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html)

<img src="figures/uga_iai.png" alt="iai" width="300">
