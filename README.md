We train VGG16 model at 87.87% test accuracy on cifar-10,when we add gaussian noise(variance of 0.01) and blur(gaussian kernel of 1.0) to 32×32 test images of cifar-10,the prediction accuracy falls down to 32.55% and 43.32% respectively.We downsample the inputs and upsample the corresponding feature maps with different size,find it can produce higher prediction accuracy.
The pristine test images are like:

<img width="500"  src="https://github.com/zcy5417/Net-for-distort-images-recognition/raw/master/test_images/pristine.png"/>
