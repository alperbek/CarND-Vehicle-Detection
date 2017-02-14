
## Vehicle Detection


```python
import numpy as np
import pandas as pd
import cv2
import glob

from scipy.ndimage.measurements import label

from skimage.feature import hog

from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
```

First of all, before everything, the training images are read:


```python
nonvehicles = pd.DataFrame(glob.glob('./non-vehicles/non-vehicles/*/*.png'), columns=['image'])
nonvehicles['label'] = 0

vehicles = pd.DataFrame(glob.glob('./vehicles/vehicles/*/*.png'), columns=['image'])
vehicles['label'] = 1
```


```python
print('Number of vehicle samples is:')
print(vehicles['image'].shape[0])
print('Number of nonvehicle samples is: ')
print(nonvehicles['image'].shape[0])
print('Shape of an image is:')
print(mpimg.imread(vehicles['image'].iloc[0]).shape)
```

    Number of vehicle samples is:
    8792
    Number of nonvehicle samples is: 
    8968
    Shape of an image is:
    (64, 64, 3)


And two examples (one for vehicle samples, one for non-vehicle samples) are read:


```python
vehicle_sample = mpimg.imread(vehicles['image'].iloc[400])
nonvehicle_sample = mpimg.imread(nonvehicles['image'].iloc[400])
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(vehicle_sample)
ax1.set_title('Vehicle Example')
ax2.imshow(nonvehicle_sample)
ax2.set_title('Nonvehicle Example')
```




    <matplotlib.text.Text at 0x7ff825bb5978>




![png](output_6_1.png)


And a container class for training parameters is defined:


```python
class Parameters:
    def __init__(self):
        self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32) # Spatial binning dimensions
        self.hist_bins = 32   # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.vis = True
```


```python
parameters = Parameters()
```

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

A helper function named "get_hog_features" is used for extracting HOG features:


```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
```

And here is an example of the hog image output for a vehicle and a nonvehicle image, for each color channel:


```python
vehicle_sample = cv2.cvtColor(vehicle_sample, cv2.COLOR_RGB2YCrCb)
nonvehicle_sample = cv2.cvtColor(nonvehicle_sample, cv2.COLOR_RGB2YCrCb)

for channel in range(0, 3):
    _, vehicle_hog = get_hog_features(vehicle_sample[:,:,channel], parameters.orient, parameters.pix_per_cell, parameters.cell_per_block, vis=parameters.vis)
    _, nonvehicle_hog = get_hog_features(nonvehicle_sample[:,:,channel], parameters.orient, parameters.pix_per_cell, parameters.cell_per_block, vis=parameters.vis)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(vehicle_hog)
    ax1.set_title('Vehicle Example')
    ax2.imshow(nonvehicle_hog)
    ax2.set_title('Nonvehicle Example')
```


![png](output_13_0.png)



![png](output_13_1.png)



![png](output_13_2.png)


#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented with many different color spaces and channels, and I am ended up with the YCrCb color space.

HSV was a good candidate too, but with YCrCb, I had a slightly better performance on the video.

Also, I experimented on the isolated channels of YCrCb for increasing the speed (i.e. decreasing the feature size), but it was not satisfying in terms of detection vehicles. For example, Y channel was good at a bluish white car, but Cr channel was good at a red car (as expected).

Also, I experimented with the isolated S channel of HSV images, but the result was not as satisfying as YCrCb.

In orientations and pixels per cell, I tried various options between certain ranges (6 to 10 for orientations, for example) and decided that 9 orientations and 8x8 pixel per cell was of the optimal result, as it is intuitively plausible considering the size of the images.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used spatial bins and color histograms together with the HOG features.

The helper functions for extracting these features are below:


```python
# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
```

Here, the features (HOG, spatial, histogram) are extracted from the data:


```python
car_features = extract_features(vehicles['image'], color_space=parameters.color_space, 
                        spatial_size=parameters.spatial_size, hist_bins=parameters.hist_bins, 
                        orient=parameters.orient, pix_per_cell=parameters.pix_per_cell, 
                        cell_per_block=parameters.cell_per_block, 
                        hog_channel=parameters.hog_channel, spatial_feat=parameters.spatial_feat, 
                        hist_feat=parameters.hist_feat, hog_feat=parameters.hog_feat)
notcar_features = extract_features(nonvehicles['image'], color_space=parameters.color_space, 
                        spatial_size=parameters.spatial_size, hist_bins=parameters.hist_bins, 
                        orient=parameters.orient, pix_per_cell=parameters.pix_per_cell, 
                        cell_per_block=parameters.cell_per_block, 
                        hog_channel=parameters.hog_channel, spatial_feat=parameters.spatial_feat, 
                        hist_feat=parameters.hist_feat, hog_feat=parameters.hog_feat)
```

The vehicle and nonvehicle features are stacked together and scaled:


```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
```

Also, their labels (1 for vehicles, 0 for nonvehicles) are stacked:


```python
y = np.hstack((vehicles['label'], nonvehicles['label']))
```

Both shuffling and train-test splitting is made by using sklearn's train_test_split function:


```python
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)
```

As the support vector classifier, LinearSVC() is used since using SVC witf 'rbf' didn't improve the performance so much, and considering its large difference in terms of speed, LinearSVC() is decided to be used.


```python
svc = LinearSVC()

svc.fit(X_train, y_train)

accuracy = svc.score(X_test, y_test)
print("Accuracy of the support vector classifier is: ", accuracy)
```

    Accuracy of the support vector classifier is:  0.992961711712



```python
class SearchParameters:
    def __init__(self):
        self.window_size = 64
        self.scale = 1.4
        self.cell_per_step = 1

        self.ystart = 380
        self.ystop = 660
```


```python
search_p = SearchParameters()
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the cell below, two functions slide_and_search and extract_window_features are presented. The slide_and_search function:
    - Initializes an empty heatmap
    - Defines block sizes and block steps (window steps)
    - Creates a windowed image (in the shape of training images)
    - Calls extract_window_features function with windowed_image and necessary parameters.

Inside the extract_window_features:
    - Precomputed hog channels (precomputed for the whole image, because taking hog features of windows separately makes the algorithm incredibly slow) are sliced to get hog feature vector relevant to the window of image.
    - Spatial and histogram features are also extracted, and they are stack together with the hog features.
    - Since the same scaling on the training phase is required on the inference stage also, features are scaled.
    - By using features, a prediction is made (car or noncar)
    - If the prediction is 1 (car), a rectangle is drawn (with a correct scale) and heatmap is updated.
 
In class SearchParameters,these parameters are defined:
    - scale: This is where the size of the sliding windows is defined. Scaling the image with a constant and using (64, 64) sized windows basically changes the "sliding window size". Let us call the scaled window are as "effective area". 64x64 sized windows (when the scale is 1) are small for the test images and the video, and they are struggling to see the "big picture". To overcome this, 1.4 is selected as the scale, and this selection is made by general computer vision intuition and trial-error. Scales bigger than 1.7 was giving too large windows and introduced more errors. Also, scales smaller than 1.2 was poor on performance both in terms of quality and speed. So, 1.4 is ended up as a good scale.
    - cell_per_step : Instead of defining the window overlap directly, step counts of the windows are defined by specifying what size of slide per step (in terms of cells) are to be done. When the cell_per_step is low (for example, 1), occurence of false positives are increased. But also, heatmap thresholding becomes easier since the cars are classified more "strongly" (more window overlaps on cars). So, I decided on a low cell_per_step parameter.
    - ystart and ystop: These are defined according to two goals:
        - Making the sliding window search only in the region of interest, so having a faster operation
        - Reducing the amount of false positives by detecting irrelevant parts such as trees as cars.


```python
def slide_and_search(image_to_search, draw_image, hog_all):
    heatmap = np.zeros_like(draw_image[:,:,0]).astype(np.float32)
    
    
    nxblocks = np.int(image_to_search.shape[1] / parameters.pix_per_cell) - 1
    nyblocks = np.int(image_to_search.shape[0] / parameters.pix_per_cell) - 1
    block_per_window = np.int(search_p.window_size / parameters.pix_per_cell) - 1
    
    nxsteps = np.int((nxblocks - block_per_window) / search_p.cell_per_step)
    nysteps = np.int((nyblocks - block_per_window) / search_p.cell_per_step)
    
    for xstep in range(nxsteps):
        for ystep in range(nysteps):
            ypos = ystep*search_p.cell_per_step
            xpos = xstep*search_p.cell_per_step
            xleft = xpos*parameters.pix_per_cell
            ytop = ypos*parameters.pix_per_cell
            windowed_image = cv2.resize(image_to_search[ytop:ytop+search_p.window_size, xleft:xleft+search_p.window_size], (64, 64))
            
            extract_window_features(draw_image, windowed_image, block_per_window, hog_all, ypos, xpos, xleft, ytop, heatmap)
    
    return draw_image, heatmap
    
    
def extract_window_features(draw_image, windowed_image, block_per_window, hog_all, ypos, xpos, xleft, ytop, heatmap):
    hog_feat0 = hog_all[0][ypos:ypos + block_per_window, xpos:xpos + block_per_window].ravel()
    hog_feat1 = hog_all[1][ypos:ypos + block_per_window, xpos:xpos + block_per_window].ravel()
    hog_feat2 = hog_all[2][ypos:ypos + block_per_window, xpos:xpos + block_per_window].ravel()
    
    hog_features = np.hstack((hog_feat0, hog_feat1, hog_feat2))
    spatial_features = bin_spatial(windowed_image, size=parameters.spatial_size)
    hist_features = color_hist(windowed_image, nbins=parameters.hist_bins)

    test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

    test_features = X_scaler.transform(test_features)
    test_prediction = svc.predict(test_features)
    
    if test_prediction == 1:
        xbox_left = np.int(xleft*search_p.scale)
        ytop_draw = np.int(ytop*search_p.scale)
        win_draw = np.int(search_p.window_size*search_p.scale)
        box = [(xbox_left, ytop_draw + search_p.ystart), (xbox_left + win_draw, ytop_draw + win_draw + search_p.ystart)]
        cv2.rectangle(draw_image, box[0], box[1],
                      (0, 0, 255), 6)
        add_heat(heatmap, box)
        
def add_heat(heatmap, box):
    # for box in bbox_list:
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap
    
```

Here is the single image pipeline:


```python
def pipeline(image):
    draw_image = np.copy(image)
    image = image.astype(np.float32) / 255
    image_to_search = image[search_p.ystart:search_p.ystop,:,:]
    image_to_search = cv2.cvtColor(image_to_search, cv2.COLOR_RGB2YCrCb)
    
    hog_all = []
    if search_p.scale != 1:
        imshape = image_to_search.shape
        image_to_search = cv2.resize(image_to_search, (np.int(imshape[1] / search_p.scale), np.int(imshape[0] / search_p.scale)))
    for channel in range(0, 3):
        hog_all.append(get_hog_features(image_to_search[:,:,channel], parameters.orient, parameters.pix_per_cell, parameters.cell_per_block, feature_vec=False))
    
    draw_image, heatmap = slide_and_search(image_to_search, draw_image, hog_all)
    
    return draw_image, heatmap
```

Helper function for visualizing multiple images in a cell:


```python
def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])
```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

A cell below, there is an example of the sliding window search without false positive rejection. 

In general, to optimize the performance of the classifier,
    - Feature vector is defined to have HOG parameters on YCrCb, spatial bins and color channel histograms.
    - Searching parameters (scale, cell_per_step, ystart, ystop) is specified as explained in the Question 1 - Sliding Window Search
    - Heatmap thresholding is applied to filter out detections with low overlapping windows (less than or equal to 3 windows) to reject false positives.
    
The optimized result is on the 2nd question of the "Video Implementation" together with the thresholding functions.


```python
out_images = []
out_titles = []
heatmaps = []
search_path = './test_images/*'
test_images = glob.glob(search_path)
for img_path in test_images:
    image = mpimg.imread(img_path)
    out_img, heat_map = pipeline(image)
    out_images.append(out_img)
    out_titles.append(img_path)
    heatmaps.append(heat_map)
image_figure = plt.figure(figsize=(12, 24))
visualize(image_figure, 8, 2, out_images, out_titles)
```


![png](output_37_0.png)


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output1.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For filtering false positives, I used heatmaps to mark detected boxes. In false positive areas, much less boxes are detected compared to the vehicles.

After creating the heatmap (by mutating a heatmap by the add_heat function), the heatmap is thresholded by apply_threshold. So, the detections with less overlapping bounding boxes are excluded in the heatmap.

By using label() function from skimage, a label map is formed. And by using it and passing it to the draw_labeled_bboxes, merged true positive rectangles are drawn.


```python
# I used this function above also, but pasted here just for convenience
def add_heat(heatmap, box):
    # for box in bbox_list:
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bboxes =[]
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img, bboxes
```


```python
out_images = []
out_titles = []
heatmaps = []
labelmaps = []
search_path = './test_images/*'
test_images = glob.glob(search_path)
for img_path in test_images:
    image = mpimg.imread(img_path)
    out_img, heat_map = pipeline(image)
    unthresholded_heat_map = heat_map
    heat_map = apply_threshold(heat_map, 3)
    labels = label(heat_map)
    out_img, box = draw_labeled_bboxes(np.copy(image), labels)
    out_images.append(out_img)
    out_titles.append(img_path)
    heatmaps.append(unthresholded_heat_map)
    labelmaps.append(labels[0])
image_figure = plt.figure(figsize=(12, 24))
visualize(image_figure, 8, 2, out_images, out_titles)
```


![png](output_41_0.png)


And now, the corresponding unthresholded heatmaps:


```python
heatmap_figure = plt.figure(figsize=(12, 24))
visualize(heatmap_figure, 8, 2, heatmaps, out_titles)
```


![png](output_43_0.png)


And this is the label map of the heat map:


```python
labelmap_figure = plt.figure(figsize=(12, 24))
visualize(image_figure, 8, 2, labelmaps, out_titles)
```


![png](output_45_0.png)



```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def process_frame(image):
    out_img, heat_map = pipeline(image)
    heat_map = apply_threshold(heat_map, 3)
    labels = label(heat_map)
    out_img, box = draw_labeled_bboxes(np.copy(image), labels)
    return out_img
```


```python
video_output = "output1.mp4"
clip = VideoFileClip("project_video.mp4")
clip = clip.fl_image(process_frame)
%time clip.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video output1.mp4
    [MoviePy] Writing video output1.mp4


    100%|█████████▉| 1260/1261 [29:51<00:01,  1.38s/it]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output1.mp4 
    
    CPU times: user 2h 22min 33s, sys: 39.8 s, total: 2h 23min 13s
    Wall time: 29min 52s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```





<video width="960" height="540" controls>
  <source src="output1.mp4">
</video>




### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It was a nice and fun project to do! I liked the last two projects because they "balanced" my knowledge on computer vision compared to the deep learning.

Selecting parameters for feature extraction (color space, hog parameters, which features should be extracted) was not easy. With an approach between general computer vision intuition, trial-error, and experimenting through the lessons, I was able to overcome it.

My pipeline will likely fail on less common vehicle designs. Also, white cars on very white roads could be problematic for seconds.

And also, "vehicle like objects" could generate false positives.

To make it more robust, a more optimized tracking algorithm can be implemented by defining a vehicle class and making/modifying list of vehicle objects from detected vehicles.

Since there is not much time (October cohort), I decided to postpone it after the end of the term.

Also, I am thinking about re-implementing the algorithm with the deep learning approach. Using SVMs on this project was a nice experience, and makes me wonder what would be the performance of the pipeline with a deep learning approach.


```python

```
