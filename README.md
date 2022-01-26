# 2D Wound Segmentation and Measurement

This project aims to:

1. Segment wounds using UNET architecture. 
    
Output Example: ![UNET Output](https://raw.githubusercontent.com/Pele324/ChronicWoundSeg/master/figures/Intro.png)

2. Calculate the area of the wound by summing the number of white pixels.

3. Segment the reference object (coin) using openCV Library. Reference paper: [Paper](https://www.researchgate.net/figure/Wound-detection-from-grabCut-algorithm_fig4_333430280)

4. Calculate the area of the coin by summing the number of white pixels.

5. Divide the area of the wound by the area of the coin.

## The Unet Architecture:

![unet](https://github.com/MohamedGamalElSherbiny/Wound_Segmentation/blob/main/images/unet.png)

Reference paper of the architecture: U-Net: Convolutional Networks for Biomedical Image Segmentation by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. [Paper Link](https://arxiv.org/pdf/1505.04597.pdf)

## Data:

The dataset used [Nature Paper](https://www.nature.com/articles/s41598-020-78799-w)
