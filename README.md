# Datasets Expansion
This is the official repository of [Expanding Small-Scale Datasets with Guided Imagination](https://arxiv.org/pdf/2211.13976.pdf).

## Abstract
The power of Deep Neural Networks (DNNs) depends heavily on the training data quantity, quality and diversity. However, in many real scenarios, it is  costly and time-consuming to collect and annotate large-scale data. This has severely hindered the application of DNNs. To address this  challenge, we explore  a new task of  dataset expansion, which seeks to automatically  create new labeled samples to expand  a small dataset.  To this end,  we present  a   Guided Imagination Framework (GIF)  that leverages the recently developed big generative models (e.g., DALL-E2) and reconstruction models (e.g., MAE) to "imagine'' and   create informative new data from  seed data to expand small datasets. Specifically, GIF conducts imagination by optimizing    the latent features of  seed  data in a semantically meaningful space, which are fed into the  generative models to generate photo-realistic images with new contents. For guiding  the imagination towards  creating   samples useful for model training, we exploit the zero-shot recognition ability of   CLIP  and introduce   three   criteria to encourage informative sample generation,  i.e., prediction consistency, entropy maximization and diversity promotion.  With these essential criteria as guidance,   GIF works well for expanding  datasets in different domains,   leading to  29.9% accuracy gain on average over  six natural image datasets, and  12.3% accuracy gain on average over three medical image datasets.


<p align="center">
<img src="./figure/Introduction.png" weight=800>
</p>
