# Using Custom Classes to Train and Predict with TensorFlow Object Detection API
Hide Inada

# Overview
Object detection is used to automatically identify the position of objects in an image.
This is a huge advance comparing with image classification which only provides the class of an object.
For example, [my image classification app](https://github.com/hideyukiinada/ic) can predict 1000 classes, but does not return the location of each object.
Is it difficult to use a machine learning system detect an object?  Not really.  If you use a pre-trained model, it is easy to do so.  However you can only classify the location of objects the model was trained for. For example, the below photo shows images of puppies with my dog Aimee at a puppy social. Though all puppies were detected, they are all marked as a dog, and you don't know which one is Aimee.  

For you to detect an object that was not in the model, then more steps are needed.

Here are the logical steps:
1. Obtain the machine learning software
2. Verify that object detection works with sample images
3. Create your own dataset for object detection
4. Download the pre-trained model to use as a base
5. Train the model with your dataset
6. Export the model after the training
7. Run prediction with the model

No matter what ML software you use, I believe that these steps are the same or similar.
Using these steps, I was able to train the model with my own dataset and used the model for predicting positions of my dogs.

I used TensorFlow Object Detection API, and I would like to go over step-by-step how I did it.  You do _not_ need to know about machine learning to follow this tutorial, but need to know Python to make modifications to scripts.

I'd also like to thank authors of articles I used as a reference.  They are listed at the bottom of this page.

# Steps
## Step 1. Obtain the machine learning software
### 1.1 Set up TensorFlow
You need TensorFlow on your machine.
[This page on TensorFlow website](https://www.tensorflow.org/install/pip) has instructions.

### 1.2 Download Object Detection API source code
```
git clone https//github.com/tensorflow/model
```

### 1.3 Navigate to the main directory
Once you check out the code, cd into the main directory:
```
cd models/research/object_detection
```
You can also have a look at [read me on Github](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Step 2. Verify that object detection works with sample images
### 2.1 Convert the tutorial from Jupyter notebook to a regular Python file (Optional)
This step is optional if you like to work with Jupyter notebook instead of a regular Python script.

In models/research/object_detection, you'll find a file called

object_detection_tutorial.ipynb

You want to convert this to a Python script by typing:

```
jupyter nbconvert --to script object_detection.ipynb
```

At the end of the script, it contains the code to display a sample image using matplotlib, but I had a problem in matplotlib on my box, so I added code to save images on the file system using the PIL package.

At the end of the step, 

??? Add photo here.

This photo is included in the source tree and licensed under [Apache License 2.0](https://github.com/tensorflow/models/blob/master/LICENSE)


I started working on playing with object detection using TensorFlow's object detection API, so I'll use this page to record steps I have taken.  This is still work in progress and this note is for me so that I won't forget the steps I had to take.

I believe the complete steps to re-train the model for your custom dataset seem to be:

1. Download the the TensorFlow objection detection API code from their github repo.
2. Verify that it works by generating the sample images with bounding box with their sample tutorial script.
3. Create your own dataset for object detection.
4. Mark bounding boxes.
5. Download the pre-trained model to retrain (Download the checkpoint).
6. Create a configuration file for setting parameters for training
7. Train the model.
8. Export the graph to be used for prediction.
9. Run prediction.

So far I've done through step 3.

## Step 1. Git clone the the TensorFlow objection detection API code from their github repo.


## Step 3. Create your own dataset for object detection.
I used my iPhone and recorded a video of my dogs Aimee and Pink for about 4 minutes 40 seconds.  I exported jpeg images from the MOV file with 3 frames/second.  I got 839 jpeg images with the below command:

```
ffmpeg -i IMG_6204.MOV -vf fps=3 ../frames/aimee_pink_%05d.jpg
```

## Step 4. Mark bounding boxes.
This is a two step process.
### Download and build labelimg
https://github.com/tzutalin/labelImg has steps to install.
It was referring to py2app which I haven't used, so I looked it up:
https://pypi.org/project/py2app/

I followed the steps the listed on the above page.  As I already had python & pip set up, I started with:
```
pip install py2app
pip install PyQt5
```
After installing PyQt5, I typed:
```
pyrcc5
```  
to verify that pyqt5 was successfully installed.

```
git clone https://github.com/tzutalin/labelImg 
cd labelImg 
make qt5py3
rm -rf build dist
python setup.py py2app -A
mv "dist/labelImg.app" /Applications
```
 
Note: I already had lxml so pip install lxml did not install it.


### Actually tag photos
Using labelImg was straightforward, but it took a long time to go through.  I annotated 707 files with Pascal VOC format.  (I went through 749 images and some of the images did not have any dogs, and I also annotated 3 images with the text format by mistake.)

### Converting images to TF Records
I used which was included in the source code and tweaked:
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pet_tf_record.py

At the end, I was able to create a file:
dog.tfrecords

## Step 7. Train the model.

### Downloading the pre-trained model
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md has the list of pre-trained model.
I downloaded faster_rcnn_resnet50_coco model.

Corresponding config file was already in the source tree.
samples/configs/faster_rcnn_resnet50_coco_config

num_classes: 3
fine_tune_check_point: path where the downloaded model was copied
tf_record_input_reader | "input_path: " : path to the tf records file that I have created.
label_map_path: path to the label_map file that I have created
 
I wasn't able to locate train.py at the object_detection directory where others listed in their articles.
I did a search on the internet and found out that the script was moved to the directory called legacy.

I tweaked the train.py to do the following:
??? (to be updated)

I stopped training at 30956 step with loss = 0.0803
The last step took 0.233 sec/step.

I believe that you need to follow the instructions below page to export the model:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md

Note I haven't done this process myself yet.

# References
&#91;1&#93; Priyanka Kochhar, Building a Toy Detector with Tensorflow Object Detection API, https://www.kdnuggets.com/2018/02/building-toy-detector-tensorflow-object-detection-api.html

&#91;2&#93; Dat Tran, How to train your own Object Detector with TensorFlow’s Object Detector API, https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

&#91;3&#93; Harrison, Training Custom Object Detector - Tensorflow Object Detection API Tutorial, https://pythonprogramming.net/training-custom-objects-tensorflow-object-detection-api-tutorial/?completed=/creating-tfrecord-files-tensorflow-object-detection-api-tutorial/

&#91;4&#93; Harrison, Testing Custom Object Detector - Tensorflow Object Detection API Tutorial, https://pythonprogramming.net/testing-custom-object-detector-tensorflow-object-detection-api-tutorial/?completed=/training-custom-objects-tensorflow-object-detection-api-tutorial/

&#91;5&#93; BalA VenkatesH, TensorFlow object detection with custom objects, https://medium.com/coinmonks/tensorflow-object-detection-with-custom-objects-34a2710c6de5
