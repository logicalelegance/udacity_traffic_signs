# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[sample_image]: ./writeup_images/sample_train_image.png "Sample Training Image"
[train_histo]: ./writeup_images/train_histo.png "Training Data Histogram"
[valid_histo]: ./writeup_images/valid_histo.png "Validation Data Histogram"
[rebalance_histo]: ./writeup_images/rebalanced_histo.png "Rebalanced and augmented Data Histogram"
[sign1]: ./sign_images/sign-18.jpg "Traffic Sign 1 (type 18, General Caution)"
[sign2]: ./sign_images/sign-22.jpg "Traffic Sign 2 (type 22, Bumpy Road)"
[sign3]: ./sign_images/sign-30.jpg "Traffic Sign 3 (type 30, Ice/Snow)"
[sign4]: ./sign_images/sign-32.jpg "Traffic Sign 4 (type 32, No limits)"
[sign5]: ./sign_images/sign-36.jpg "Traffic Sign 5 (type 36, Straigt or right)"
[signs_input]: ./sign_images/signs_resized.png "Signs at input size"
[conv1]: ./sign_images/conv1.png "Convolution layer 1"
[conv2]: ./sign_images/conv2.png "Convolution layer 2"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/logicalelegance/udacity_traffic_signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I obtained the length of each of the data subsets and output them into a jupyter cell.

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is 32x32 pixels with 3 color channels
* The number of unique classes/labels in the data set is 43 sign types.

#### 2. Include an exploratory visualization of the dataset.

To get a sense of the quality of the data, I performed two actions.
First, I displayed a representative image with matplotlib, just to get a sense of what the data looked like. While there's only one shown in the current jupyter notebook, I did jump around and check out some other images. My sense was that the quality of the images was highly variable. Some were quite clear, while others had issues with sharpness and lighting.

![alt test][sample_image]

The next and more important step was to get a sense of the distribution of labels within the training and validation sets.
To do this I used matplotlib's `plt.hist()` function to show a histogram of each set, with label types corresponding to the frequency bins.

![alt text][train_histo]

Above the training histogram immediately illustrates a problem. The training data is highly unbalanced. The lower third of label types is overwhelmingly overrepresented in the training set. Some labels are included in the set 2000 times, while others less than 250. This will have an effect on our ability to train the network and will need to be addressed.

Just to verify this distribution carries over to the validation set, I also plotted the histogram for that subset.

![alt text][valid_histo]

The disribution is exactly the same. Which is good! This means the validation set was properly randomly sampled from the trainig set.

I did not visualize anything from the test set so as not to influence any of my decisions during network architecture setup and training.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided *against* converting the images to greyscale. After experimenting with network architectures and thinking about the data set, it was seemed clear that did not provide an advantage to training. I suspect this is because there is color context in the images (red sign boundaries, blue arrow, etc.) that helps the network distinguish between images.

My primary focus in preprocessing was balancing the training data set and providing *more* data to the network to improve accuracy.

To that end I performed two major preprocessing steps: data augmentation, and replication.

For augmentation, I decided the most straightforward approach, after reviewing example data, was to create additional synthetic data by applying small rotations (+/- 5º, +/-10º, +/-15º) and including those new images in the set. My reasoning was that small rotation was the most common difference between images within the set and that adding more would be helpful.

In order to correct for the data imbalance discovered during the visualization step, I implemeneted a method to replicate images that were underrepresented in the data set. I settled on this after researching methods for addressing data set imbalance and noting that it was very straightforward. After applying augmentation and this rebalancing, the training set histogram looked like this:

![alt text][rebalance_histo]

This is much improved, now the worst case count discrepancy is 2x rather than 10x, and for the most part images are equally represented within the data set. This means the network will "see" each training set image nearly equally often. 

The last step in preprocessing was to normalize each image's pixel value to (-1, 1) to have the mean close to zero and equal variance.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6, valid padding 				|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x16, valid padding      									|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16, valid padding 				|
| Fully connected		| Input 400, output 120.       									|
| ELU					|												|
| Fully connected		| Input 120, output 84.       									|
| ELU					|												|
| Dropout				| Dropout layer, 0.5 keep probability during training |
| Softmax				| Final operation on logits. |

A few comments on this architecture. First, the initial design is the LeNet style network straight from the MNIST lab. I found that was underperforming with initial trials so I made a few modifications.

First, I increased the depth of the first layer from 6 to 18, reasoning that using RGB data required some extension in the network to allow for additional color data.

Second, after research I switched the RELU activations to ELU. My reading indicated it was a superior activation function that allows for more flexibility and reduces overtraining.

Finally, and probably most importantly, I added dropout to the final fully connected layer. This had the greatest effect on my ability to prevent overtraining from happening too quickly and improve training accuracy over several epochs.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained using the Adam optimizer (supposedly somewhat superior to standard Stochastic Gradient Descent). I experimented with alternate batch sizes, number of epochs, and training rates, but the defaults that were used in the MNIST lab turned out to be better than any others I tried. Those values were:

| Hyperparameter | Value  |
|:--------------:|:------:|
| Batch size | 128 |       
| Epochs     | 10 |
| Training Rate | 0.001 |
| Dropout Keep Probability | 50% |

I tried batch sizes of 64 and 256, both were worse for accuracy. I varied the training rate, trying 0.010 and 0.0005 and both seemed no better or slightly worse. The dropout rate was also determined empirically, 25% and 75% were not significantly better than 50% (or were worse).

I used 10 epochs because after using 20+, it was clear that the network was not improving accuracy after about the 10th iteration and it wasn't worth the extra time.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

At each epoch I saved a checkpoint of my model. The best model of the 10 epochs, epoch 9 had:

* training set accuracy of 99.9%
* validation set accuracy of 97.5%
* test set accuracy of 95.3%

I was fairly pleased with these results. The actual training takes place in cell `In [12]` of the notebook. Training set and validation set accuracy were also calculated within that loop. Test set accuracy was evaluated on the saved model in cell `In [16]`.

I've already discussed my approach to dialing in the model in previous sections, but to summarize, I iterated with various architecture changes (convolution layer depth, strides, hyperparameters). Those approaches that improved training accuracy were kept, those that did not were discarded. Initially, my network had accuracy < 93% on the validation set, so all of the changes I made were successful in improving to 97.5%.

The dropout layer was the single most important change in preventing overtraining. Before adding that layer, my network would often reach close to 100% training set accuracy, but get stuff below 93% of validation set accuracy.

I think the results show that my network architecture and parameters are well suited to the problem set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][sign1] ![alt text][sign2] ![alt text][sign3] 
![alt text][sign4] ![alt text][sign5]

Here are all the signs, as resized for input to the network:

![alt text][signs_input]

Most of these images are fairly clear but in the above image the last two images may present difficulties as they are poorly cropped (and thus lose detail when scaled down to 32x32) and are presented at angles.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No limits      		| No Limits									| 
| Straight and right    | Straight and right						|
| Bumpy road ahead   	| Bumpy road ahead							|
| General Caution  		| Wild animals crossing			 			|
| Beware of ice/snow	| Traffic Signals					|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is far worse than the test set accuracy of this model of 95.3%. I can only speculate that those last two images were poorly framed and not well represented in the training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image, the model is moderately sure that this is a no limits sign(probability of 52%), and the image does contain a no limits sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .52         			| No limits   									| 
| .30     				| End of speed limit 										|
| .27					| Keep right										|
| .25	      			| End of no passing				 				|
| .16				    | Slippery Road      							|

Note that several of the likely (but wrong) possibilities are other "end of limits" variants.

For the second image the model is again moderately certain that it's "Go straight or right" with 95% confidence (and it's right).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .51         			| Go straight or right   									| 
| .23     				| End of all speed and passing limits									|
| .16					| Keep right										|
| .09	      			| Ahead only			 				|
| .09				    | Slippery road	

Again it's interesting that several wrong guess are variants of the correct guess (keep right or ahead only).

For the third image the model is far more confident, with the max probability being 95%. This is the correct answer (bumpy road).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95        			| Bumpy road									| 
| .63    				| Bicycles crossing								|
| .40					| Road work									|
| .40	      			| Traffic signals			 				|
| .25				    | Wild animals crossing

I think this was a difficult image due to the "Bumpy road" portion being downsampled so far, but it still did very well at distinguishing it.

For the fourth image, matters are worse. The max probability is 13% and it's the wrong answer (correct answer was General Caution, the model predicted Road Work).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .13        			| Road Work							| 
| .13    				| Beware of ice/snow						|
| .08					| No passing..									|
| .07	      			| Right of way at next...		 				|
| .05				    | Slippery road

None of the top five here are correct. The image angle/crop must have removed too much detail for classification.

Finally the fifth image is also incorrectly classified. With 36% confidence, the model predicted "No Entry", when the correct label is "Beware of ice/snow".

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .36       			| No entry						| 
| .28    				| Dangerous Curve Right						|
| .16					| Bumpy road								|
| .16	      			| Slippery Road	 				|
| .16				    | Traffic signals

Again, none of these are correct. The angle must have confused it or snow/ice must be not well covered in the training set.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I visualized the network at two points, using the "End of limits" image as the stimulus.

At the first convolution layer, the feature maps look like this:

![alt text][conv1]

It seems like this is looked for edges for the most part with certain layers lighting up more for the larger gradients and others picking out smaller edge features. It's apparent that using Valid padding may also be cropping out portions of the image that are useful. Had I known I might have switched to Same Padding.

The second convolution layer looks like this:

![alt text][conv2]

Here it's pretty clear the layer is looking primarily for diagonal features and lines. Which this sign lights up pretty well!

