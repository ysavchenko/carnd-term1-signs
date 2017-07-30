# **Traffic Sign Recognition** 

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

[train1]: ./images/train_1.png "Train Image 1"
[train2]: ./images/train_2.png "Train Image 2"
[train3]: ./images/train_3.png "Train Image 3"
[train4]: ./images/train_4.png "Train Image 4"
[train5]: ./images/train_5.png "Train Image 5"
[hist_train]: ./images/hist_train.png "Training set histogram"
[hist_valid]: ./images/hist_valid.png "Validation set histogram"
[hist_test]: ./images/hist_test.png "Testing set histogram"
[gray_source]: ./images/gray_original.png "Before grayscale"
[gray_mean]: ./images/gray_mean.png "Mean grayscale"
[gray_weighted]: ./images/gray_weighted.png "Weighted grayscale"
[zoom_before]: ./images/zoom_before.png "Zoom before"
[zoom_after]: ./images/zoom_after.png "Zoom after"
[new_01]: ./images/sign_01.jpg
[new_02]: ./images/sign_02.jpg
[new_03]: ./images/sign_03.jpg
[new_04]: ./images/sign_04.jpg
[new_05]: ./images/sign_05.jpg
[new_06]: ./images/sign_06.jpg
[new_07]: ./images/sign_07.jpg
[new_08]: ./images/sign_08.jpg
[new_09]: ./images/sign_09.jpg
[new_10]: ./images/sign_10.jpg
[new_11]: ./images/sign_11.jpg
[new_12]: ./images/sign_12.jpg
[worst_precision]: ./images/worst_precision.png
[worst_recall]: ./images/worst_recall.png
[70km_results]: ./images/70km_results.png
[conv1_visual]: ./images/conv1_visual.png

## Rubric Points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I've started data visualization with showing images themselves. Here are 5 random images from the test set with their class labels.

![Train image 1][train1] ![Train image 2][train2] ![Train image 3][train3] ![Train image 4][train4] ![Train image 5][train5]

The next question was about the distribution of different classes in training, testing and validation data sets. Here are histograms for each data set:

![Training set histogram][hist_train]

![Validation set histogram][hist_valid]

![Testing set histogram][hist_test]

These histograms show that class distribution is similar among datasets. Let's convert this into numbers. After comparing class shares we see the following results:

	Difference between Training and Validation set
	MAX: 75.4% for label 21
	MEAN: 17.5%
	Difference between Training and Testing set
	MAX: 21.3% for label 27
	MEAN: 6.8%

This shows that class distribution is pretty similar between sets. Let's explore some more and see most common labels in each sets:

	For Training set:
	Most common label is 2 with share of 5.776028%
	Least common label is 0 with share of 0.517256%
	For Validation set:
	Most common label is 1 with share of 5.442177%
	Least common label is 0 with share of 0.680272%
	For Testing set:
	Most common label is 2 with share of 5.938242%
	Least common label is 0 with share of 0.475059%

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not choose grayscale as the first step, because I believed that color contains the crucial information about street signs and discarding color would make them less recognizable by the model. 

So the first and only thing I did initially was data normalization (which is the process of augumenting pixel color values to have 0 mean value and making absolute values not larger than 1). This makes it easier for opmimizer to find optimal variable values for the model to minimize the loss.

Here is the code I used for this:

```python
image_array = image_array.astype(np.float32)
image_array -= 128
image_array /= 128.
```
	
First step (converting image to float32) was essential because initially color components were uint8 which do not support floating point values.

After experimenting with different model architectures I've decided to give removing color from training images a try. My first approach was just to average a sum of all color components like this:

```python
X_train = X_train.mean(axis=3)
X_valid = X_valid.mean(axis=3)
X_test = X_test.mean(axis=3)

X_train = X_train.reshape(X_train.shape + (1,))
X_valid = X_valid.reshape(X_valid.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))
```

The last step (reshaping arrays) was done because calculating mean over the axis reduced dimensionality and we need it to be preserved so our model architecture continues to work.

This method of grayscaling did not prove to be very effective and validation accuracy lowered when trained on such images. So after a short research I ended up on weighted approach, when before color components are averaged they are multiplied by weight, which depends on human perception of each color component (for example, human eye is most sensitive to the green color so it has the highest multiplier). This is the code I used:

```python
def to_grayscale(im, weights = np.c_[0.2989, 0.5870, 0.1140]):
    tile = np.tile(weights, reps=(im.shape[0],im.shape[1],1))
    return np.sum(tile * im, axis=2)
```

Training with images transformed to gray this way produced jigher validation results, so I kept this. And here is a small vizualization of difference between those images:

![Gray image source][gray_source] ![Gray image mean][gray_mean] ![Gray image weighted][gray_weighted]

As you probably can see (depending on your monitor), blue background of the sign is darker on the last image. This is because blue has the lowest weight (11%) and with simple averaging blue component takes 1/3 of the resulting color making it lighter. Since the last image is closer to the human perception of color components it showed better results in validation.

And finally I tried to make training data set larger by adding image augmentation. I decided against using rotation and flip because sign meaning can change if it is upside down or flipped. So instead I've chosen zoom. Here is the code I used for this:

```python
from scipy.misc import imresize

def zoom_image(image, ratio=1.2):
    zoomed = imresize(image, ratio)
    crop_index = int((zoomed.shape[0] - image.shape[0]) / 2)
    return zoomed[crop_index:crop_index + image.shape[0], crop_index:crop_index + image.shape[1],:]
```

The image size is increased using image resize fry SciPy library and then we crop it back to the original size. At the end we get something like this:

![Zoom before][zoom_before] ![Zoom after][zoom_after]

Modified images are added to the training dataset using the same labels as original ones. This makes training dataset two times larger.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer 				| Description |
|:-----------------:|:-----------:|
| Input 				| 32x32x1 Grayscale image |
| Convolution 5x5 	| 1x1 stride, no padding, output 28x28x18 |
| RELU					| 
| Max pooling			| 2x2 stride, output 14x14x18
| Convolution 5x5 	| 1x1 stride, no padding, output 10x10x48 |
| RELU					| 
| Max pooling			| 2x2 stride, output 5x5x48
| Flatten				| Output 1200
| Dropout				| 0.5 pass through probability
| Fully connected	| From 1200 to 360
| RELU					| 
| Dropout				| 0.5 pass through probability
| Fully connected	| From 360 to 252
| RELU					| 
| Dropout				| 0.5 pass through probability
| Fully connected	| From 252 to 43

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The final model is trained on `30` epochs with batch size `128` and learning rate `0.0005`. Compared to the original LeNet architecture batch size is the same and learning rate is lower and number of epochs is higher (I describe below how I ended up with these values). Also another hyper-parameter is dropout pass-through rate, which is `0.5` during training.

I've kept the original Adam optimizer, it seems to do a good job with this model. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of `1.000`
* validation set accuracy of `0.976`
* test set accuracy of `0.968`

Here is how I came up with the final network architecture:

* Initial training was run on the original LeNet architecture. The only change was increasing input dimensionality to `32x32x3` (first training was done on color images) and output dimensionality to `43` (we have `43` classes for signs). Validation accuracy for this model was `0.903`.
* Next I trippled the size of the model. I did it because color images has three times more data. So I've increased output of the first convolutional layer to have depth of `18`, the second to `48` and so on. Validation accuracy on this model was `0.943`.
* Then I've tried to recude learning rate (to `0.0005`), model began to train slower, so I've increased the number of epochs to `30`. Validation accuracy increased a little to `0.951`.
* One thing I noticed at this point was that training accuracy quickly reached `1.000` (on first 10 epochs or so), so to combat overfitting to the data I've added 3 dropout layers (before each of the fully connected layers). After these layers were added training still reached `1.000` accuracy, but this time it was much later (near 30th epoch). Validation accuracy reached `0.971`.
* Only then I've tried converting images to grayscale. The first approach (as mentioned earlier) was to average all color components. Also to work with grayscale images I've reduced input dimensionality to `32x32x1`. This did not go as well and validation score dropped to `0.962`.
* Another attempt on grayscale images was to use weighted color components (to mimic human color perception). This made things better with validation accuracy between `0.975` and `0.985` on different training runs.
* Finally, while model was still the same I've trained it again with added zoomed images. Validation score was essentially the same.

#### Evaluating model performance on a test set

As I mentioned previously, this model got `96.8%` accuracy on a test set. To understand this performance better I've decided to calculate precision and recall for each street sign type in our set.

Precision is a number of true positives (number of signs correctly recognized) devided by the number of all predictions of this particular sign type (true positives plus false positives). And recall is again true positives, but divided by a total number of signs of this type in the set (true positives plus false negatives). Precision is a measure of how accurate model predictions are and recall - how thorough are the predictions (the higher recall is the more images of a particular kind the model finds).

Here is the code I used to calculate both:

```python
true_positive = np.zeros(n_classes)
false_positive = np.zeros(n_classes)

# Run trained model on a test set and get probability for each class
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_result = sess.run(softmax, feed_dict={x: X_test, dropout: 1})

# Apply argmax to the result to receive classes with highest probability
test_result = test_result.argmax(axis=1)

# Calculate true positives and false negatives for each class
for true_class, predicted_class in zip(y_test, test_result):
    if true_class == predicted_class:
        true_positive[predicted_class] += 1
    else:
        false_positive[predicted_class] += 1
        
precision = true_positive / (true_positive + false_positive)
recall = true_positive / counts
```

This produced some interesting results. Precision of most classes were close to the mean accuracy across all classes (the one we've got from evaluating the model accuracy as a whole) with only few classes having significantly lower precision. While recall had larger difference among classes, with one class in particular reaching as low as `50%` recall.

Here is a histogram of bottom 5 precision and recall results for the test set.

![Bottom 5 precision][worst_precision]

![Bottom 5 recall][worst_recall]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are twelve German traffic signs that I found on the web:

![New 1][new_01] ![New 2][new_02] ![New 3][new_03] ![New 4][new_04] ![New 5][new_05] ![New 6][new_06]
![New 7][new_07] ![New 8][new_08] ![New 9][new_09] ![New 10][new_10] ![New 11][new_11] ![New 12][new_12]

Signs here are easily recognizable by the human eye, but our model might have a problem with signs which have more information on the background compared to others. For example, the second `General caution` sign ![New 4][new_04] or `Slippery road` ![New 6][new_06] have many visual details on the backgroung which could confuse our model.

Also there could be problems in classifying `70km/h speed limit` sign. It has the same general characteristics as the other speed limit signs and the model might confuse it with a different speed limit.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image | Prediction | Correct |
|:------:|:-------:|:-------:| 
| ![New 1][new_01] | Right-of-way on the next intersection | YES |
| ![New 2][new_02] | Priority road | YES |
| ![New 3][new_03] | General caution | YES |
| ![New 4][new_04] | General caution | YES |
| ![New 5][new_05] | No entry | YES |
| ![New 6][new_06] | Slippery road | YES |
| ![New 7][new_07] | Yield | YES |
| ![New 8][new_08] | Roundabout mandatory | YES |
| ![New 9][new_09] | Speed limit (30km/h) | **NO** |
| ![New 10][new_10] | No passing | YES |
| ![New 11][new_11] | Road work | YES |
| ![New 12][new_12] | Bumpy road | YES |

The model was able to correctly guess 11 of the 12 traffic signs, which gives an accuracy of `91.67%`. This is similar to the results we've got from our test set (`96.8%`).

Looking back at precision and recall for the image quessed incorrectly (the one we got from the test set) was:

```
Precision=98.46%, recall=96.97%
```

Those are not the lowest precision and recall on the test set, so I can only say that probably the new image we tested on was different from the ones in train/test set. May be the framing of the sign or a font used for `70` was different. This means that this model still needs work to be better.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here is the code I used to calculate softmax probability for the new images:

```python
import tensorflow as tf
def classify_images(data):
    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        result = sess.run(softmax, feed_dict={x: data, dropout: 1})
        
    return result
```

Then I applied top 5 on the result with the following code:

```python
result = classify_images(X_new)
with tf.Session() as sess:
    top_5 = sess.run(tf.nn.top_k(tf.constant(result), k=5))
```

After printing the probabilities I saw that most of them were `99.9%` to `100%` sure on a particular class except for the image #8 (the one recognized incorrectly). Here is a histogram for probabilities of different classes:

![70km/h results][70km_results]

As you can see, the second best prediction of the model (with probability of `24.2%`) was a correct class (70km/h). And as I suspected, the model sees all speed limit classes similarly because top 5 consists only of the speed limit classes.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Using the provided visualisation code I've visualized the output of the first convolution layer on one of the images (the first one among the new images I tested on). Here how it looks like:

![Conv1 visualization][conv1_visual]

From this visualization you can see what the network is most interested in. It highlights the shape of the sign (triangle) and the shape of the figure within the sign. Everything else (like background behind the sign) is mostly ignored, which is good, the model should focus on the sign itself and not its surroundings.

