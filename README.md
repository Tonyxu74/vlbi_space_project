# Trying to Reconstruct Very-Long-Baseline-Interferometry (VLBI) Images

### About
Inspired by the recent imaging of the black hole M87, I am trying to reconstruct images from incomplete UV-plane visibility data. For validation data, I downloaded images from http://vlbiimaging.csail.mit.edu, and for training data I downloaded from http://www.vision.caltech.edu/Image_Datasets/Caltech256. 

#### Brief background
In any imaging problem, the larger the "aperture" you have to look at, the better resolution you can obtain at a certain distance. For objects extremely far away, the required aperture size becomes extremely large, comparable to the size of the Earth. For these problems, rather than crafting an Earth-sized mirror or lens, VLBI is used. Instead of focusing light at a certain position like conventional telescopes, VLBI creates an artifical mirror by precisely linking together telescopes stationed a distance apart. These telescopes (more specifically, a group of two telescopes) each take in a piece of the puzzle, which results in a not-quite-complete image of the object you are looking for. This image is in the UV-plane, essentially a fourier transform of the conventional brightness image that you are looking for, and only contains a few data points. This data is also called a complex visibility image. A major achievement of the Event Horizon Telescope's imaging of M87 involves reconstructing the black hole image from this incomplete data, and is also what I am attempting to do with machine learning methods. 

#### What I've done so far
A quick archive of my current progress (still a work-in-progress!):
* A UNet regression model
  * MSE Loss to train a regression problem, with the input being the incomplete visibility image, and the output being the complete visibility image.
  * Since complex visibility has both imaginary and real parts, or a magnitude and phase, I attempt training separate UNets for the magnitude and phase.
  * To mitigate the lack of data available, I created the UV-plane generator function, which mimics the shape of UV-plane images obtained from rotating telescopes. With this, I could use any input image to train my model. I used the Caltech256 image dataset.
  * With the PyTorch support for FFT and iFFT, I created a combined training algorithm that trains the magnitude and phase models simultaneously. 

* AUTOMAP from [Zhu et al.'s "Image reconstruction by domain-transform manifold learning"](https://arxiv.org/ftp/arxiv/papers/1704/1704.08841.pdf)
  * Specifically found to work well on reconstructing incomplete k-space (another name for UV-plane) data, currently attempting to implement this.

### Output image highlights
These images are a small sample of my current outputs. Though they are noisy, the general shape and positioning seem to be correct, but at a smaller size. 

__Output Prediction &emsp; &emsp; &emsp; Label__

<img src="https://github.com/Tonyxu74/vlbi_space_project/blob/master/image_highlights/162_prediction.png" width="200" height="200"><img src="https://github.com/Tonyxu74/vlbi_space_project/blob/master/image_highlights/162_gt.png" width="200" height="200">

<img src="https://github.com/Tonyxu74/vlbi_space_project/blob/master/image_highlights/167_prediction.png" width="200" height="200"><img src="https://github.com/Tonyxu74/vlbi_space_project/blob/master/image_highlights/167_gt.png" width="200" height="200">

<img src="https://github.com/Tonyxu74/vlbi_space_project/blob/master/image_highlights/272_prediction.png" width="200" height="200"><img src="https://github.com/Tonyxu74/vlbi_space_project/blob/master/image_highlights/272_gt.png" width="200" height="200">

<img src="https://github.com/Tonyxu74/vlbi_space_project/blob/master/image_highlights/295_prediction.png" width="200" height="200"><img src="https://github.com/Tonyxu74/vlbi_space_project/blob/master/image_highlights/295_gt.png" width="200" height="200">
