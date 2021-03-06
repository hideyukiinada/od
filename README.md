# Using Custom Classes to Train and Predict with TensorFlow Object Detection API
February 5, 2019

Hide Inada

<IMG SRC="assets/images/od_test00038_cropped.jpg">

# Overview
Object detection is used to automatically identify the position of objects in an image.
This is a huge improvement compared to image classification which only provides the class of an object.
For example, [my image classification app](https://github.com/hideyukiinada/ic) can predict 1000 classes, but does not return the location of each object.
Is it difficult to set up and use a machine learning system to detect an object?  Not really.  If you use a pre-trained model, it is easy to do so.  However you can only detect locations of objects for which the model was trained. For example, the below photo shows images of puppies with my dog Aimee at a puppy social that I processed with YOLO without using my own dataset. Though all puppies were detected, they are all marked as a dog, and you don't know which one is Aimée.  

<img src='assets/images/aimee_puppy_social.jpg' width='400px'>
If you are interested in checking out YOLO, please refer to my blog post in December at https://hideyukiinada.github.io/2018/12/23/yolo.html.

For you to detect an object that was not in the model provided, then more steps are needed to retrain the model.

Here are the high-level steps:
1. Obtain the machine learning software
2. Verify that object detection works with sample images
3. Create your own dataset for object detection
4. Download the pre-trained model to use as a base
5. Train the model with your dataset
6. Convert the model to be used for prediction
7. Run prediction with the model

No matter what ML software you use, I believe that these steps are the same or similar.
Following these steps, I was able to train a model with my own dataset and used the model for predicting positions of my dogs. In the cover photo, you see the label "aimee" appears on the top-left of the bounding box.

I used TensorFlow Object Detection API, and I would like to go over step-by-step how I did it.

I would also like to thank authors of articles I used as a reference.  They are listed at the bottom of this page.  Without tips from people who have already done this, going through this exercise myself would have been much more difficult.

# What you need to go through the steps yourself
## Software
* TensorFlow
* TensorFlow Object Detection API source code
* Python
* Python pip
* Software to mark (annotate) the location of object (e.g. labelImg)

## Hardware
* Hardware to run training & prediction

If you run training locally, I strongly recommend a machine with a GPU.  If you haven't, please check out [my article on the speed gain provided by GPU over CPU](https://www.linkedin.com/pulse/whats-speed-difference-between-running-ml-job-gpu-vs-cpu-inada/).  I believe that you can run training in Google's cloud but I haven't tried that myself.

## Skills
* Python programming skill

For going through this tutorial to reproduce my result, neither knowledge for TensorFlow programming nor familiarity with machine learning is needed.

## Expected time to take
* 2 days (with annotating the photos taking the largest chunk).

# Steps
## Step 1. Obtain the machine learning software
### 1.1. Set up TensorFlow
You need TensorFlow on your machine.
[This page on TensorFlow website](https://www.tensorflow.org/install/pip) has instructions.

### 1.2. Download Object Detection API source code
```
git clone https//github.com/tensorflow/model
```

### 1.3. Navigate to the main directory
Once you check out the code, cd into the main directory:
```
cd models/research/object_detection
```
You can also have a look at [read me on Github](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Step 2. Verify that object detection works with sample images
### 2.1. Convert the tutorial from Jupyter notebook to a regular Python file (Optional)
This step is optional if you like to work with Jupyter notebook instead of a regular Python script.

In models/research/object_detection, you'll find a file called

object_detection_tutorial.ipynb

You want to convert this to a Python script by typing:

```
jupyter nbconvert --to script object_detection_tutorial.ipynb
```

This produces a script called object_detection_tutorial.py in the current directory.

At the end of the script, it contains the code to display a sample image using matplotlib, but I had a problem in matplotlib on my box, so I added code to save images on the file system using the PIL package.

Once you run the script, you should see the photos below:

<img src='assets/images/sample_0.jpg' width='400px'>
<img src='assets/images/sample_1.jpg' width='400px'>

These images are included in the source tree and licensed under [Apache License 2.0](https://github.com/tensorflow/models/blob/master/LICENSE)

## Step 3. Create your own dataset for object detection
There are four sub-steps in this step:

1. Obtain images with objects that you want to detect
2. Obtain a software product to mark location of your objects in each image
3. Mark a location in each image
4. Convert the image and location data into a file format that your ML software can process

### Step 3.1. Obtain images with objects that you want to detect
I used my iPhone and recorded a video of my dogs Aimée and Pink for about 4 minutes 40 seconds.  I exported jpeg images from the MOV file with 3 frames/second.  I got 839 jpeg images with the below command:

```
ffmpeg -i IMG_6204.MOV -vf fps=3 ../frames/aimee_pink_%05d.jpg
```
Each JPEG file was 1920x1080 with 3 channels.

If you want to classify objects that are accessible to you, I recommend using a video instead of taking a photo or collecting images over the Internet because:
1) It is much faster to capture a video with lots of images instead of taking separate photos
2) You can capture way more diverse shapes of your objects especially if you are capturing images of animals.  
    For example, in most of the images contained in my video, my dogs are not in picture-card-quality poses but in more dynamic and realistsic positions.  As I mentioned in my article [Limitation of Neural Networks](https://www.linkedin.com/pulse/limitation-neural-networks-hideyuki-inada/), it's very important that you have images that are similar to what you want to predict.  Just having multiple photos of dogs sitting with their face in the center of a frame will result in failure to detect dogs in actual images.
3) Typically a photo contains way more pixels than a frame in a video file which is not needed for object detection adding overhead to process a high-resolution images

Also you may feel like you want to just hire someone else to delegate this laborious task of annotating images, but I recommend at least you go through images for a few hours yourself to come up with a guideline before you ask someone else to do it.

Criteria I had to establish for me are the following:
* How to handle images in which a dog is only partially seen
* How to handle images in which a dog is blocked by another dog or another object
* How to handle their hair in positioning a bounding box

### Step 3.2. Obtain a software product to mark location of your objects in each image
I used a product called _labelImg_.  [The developer's github page](https://github.com/tzutalin/labelImg) has detailed steps to install and use the software.  I set up labelImg on my Mac so these steps are for that.  I think steps for Linux should be slightly different as /Applications directory is specific to Mac.

I followed the steps the listed on the above page.  As I already had Python & pip set up, I started with:
```
pip install py2app
pip install PyQt5
```
After installing PyQt5, I typed the below command to verify that PyQt5 resource compiler was successfully installed as mentioned in [a Stack Overflow article](https://stackoverflow.com/questions/46986431/make-no-rule-to-make-target-qt5py3-stop).
```
pyrcc5
```  

Then I cloned the repo and followed the steps on the page. (Note I already had lxml so pip install lxml did not install it.)
```
git clone https://github.com/tzutalin/labelImg 
cd labelImg 
make qt5py3
rm -rf build dist
python setup.py py2app -A
mv "dist/labelImg.app" /Applications
```

With these steps, you now have labelImg installed in your Applications folder.

### Step 3.3. Mark a location in each image 
Marking a location or annotation for each object means:

* Identify an object in each image
* Assign a label to each object
* Marking a bounding box for each object

At the end of this step, for each image file, you want to have a corresponding XML file that contains the coordinates of objects in the image.

<img src='assets/images/labelimg_screenshot.png' width='800px'>

Below is the actual file (aimee_pink_00537.xml) that was created for the image aimee_pink_00537.jpg.
```
<annotation>
        <folder>frames</folder>
        <filename>aimee_pink_00537.jpg</filename>
        <path>/Volumes/Toshiba/data/programs/ai/dataset/od_aimee_pink/frames/aimee_pink_00537.jpg</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>1920</width>
                <height>1080</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>aimee</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>695</xmin>
                        <ymin>239</ymin>
                        <xmax>1049</xmax>
                        <ymax>722</ymax>
                </bndbox>
        </object>
        <object>
                <name>pink</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>1116</xmin>
                        <ymin>129</ymin>
                        <xmax>1624</xmax>
                        <ymax>567</ymax>
                </bndbox>
        </object>
</annotation>

```

This format is called [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) format. (If you want to view a dataset tagged with this format, I found that you can download from the [GluonCV's website](https://gluon-cv.mxnet.io/build/examples_datasets/pascal_voc.html))

Using labelImg was straightforward, but it took a long time to go through. labelImg has short cut keys and they helped a lot.  Out of 839 images that I had, I went through 749 images. Some of the images did not have any dogs and I also annotated 3 images with the text format by mistake, so I ended up annotating 707 files.

### Step 3.4. Convert the image and location data into a file format that your ML software can process
Now you have a set of JPEG images and corresponding XML files in the annotation directory.
In this step, you need to combine all of them into a single TFRecord file format that the training script needs.

I made a copy of models/research/object_detection/create_pascal_tf_record.py and modified it so that it reads from my image and annotation directories.  I didn't make changes to make the code for a general purpose use, and my changes were rather hacky, so I'd rather not post my changes in this article, but I don't think it will take much time for a Python programmer to get it to work. If you are really stuck in this step, I may be able to help, so please PM me.

At the end, I was able to create a single file called dog.tfrecords.  Also, please make sure that the size of the generated .tfrecords file is not very small compared to the original JPEG files.  I tweaked directory names and resulted in a very small tfrecords file, which wasn't right and I had to redo some steps to produce the correct .tfrecords file.

## Step 4. Download the pre-trained model to use as a base
You can train a model from scratch, but it takes a long time.  Instead, you can download a pre-trained model to shrink the time needed for training.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md has the list of pre-trained model.
I downloaded faster_rcnn_resnet50_coco model as I have experience with using ResNet50 and was happy with its performance in my image classification project.  There are many other models available and you can pick the one that best fits your needs.  

## Step 5. Train the model with your dataset
In addition to the dataset in TFRecords format that you created in step 3 and the pre-trained model that you downloaded in step 4, you need the following items to train the model:

* Configuration file to set parameters for training
* Training script
* Label map file

### Step 5.1. Configuration file

You need a config file that matches the downloaded pre-trained model.  A matching config file should already be in the samples/configs directory of the source tree.
In my case, I used:
models/research/object_detection/samples/configs/faster_rcnn_resnet50_coco.config

I tweaked the following parameters:

| Parameter | Value | Example |
|---|---|---|
| num_classes | Number of classes that your dataset has | 3 | 
| fine_tune_check_point | Path where the downloaded model was copied and the prefix of the checkpoint | /home/hinada/downloaded_model/model.ckpt |
| tf_record_input_reader input_path | Path to the tf records file | /home/hinada/data/dog.tfrecords |
| tf_record_input_reader label_map_path | Path to the label_map file that I have created | /home/hinada/data/dog_label_map.pbtxt |
| eval_input_reader input_path | Path to the tf records file | /home/hinada/data/dog.tfrecords |
| eval_input_reader label_map_path | Path to the label_map file that I have created | /home/hinada/data/dog_label_map.pbtxt |
    
There are two issues to note here.

First, regarding the 'fine_tune_check_point' parameter, if you download the pre-trained model and place it in a directory (e.g. /home/hinada/downloaded_model), if you type ls, you should see:
```
checkpoint  frozen_inference_graph.pb  model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta  pipeline.config  saved_model
```
I specified the full path of the directory plus "model.ckpt".

Second, regarding eval_input_reader settings, I started tweaking the configuration file after I created the TFRecord dataset, so I didn't know that there are parameters for the validation set in a configuration file.  I just specified the training set for validation, but if you want to check the validation accuracy during training, you might want to create a separate validation dataset in TFRecord and specify it here.
       
### Step 5.2. Training script
Initially I wasn't able to locate train.py at the object_detection directory where others listed in their articles.
I did a search on the internet and found out that the script was moved to the directory called _legacy_ under object_detection.

I duplicated train.py and renamed the copy to train_dog.py.
To make it work in my environment, I did the following:
* Added additional Python Paths to include "../.." and "_&lt;the path where I cloned the repo&gt;_/models/research/slim"
* Set the 'train_dir' flag to a directory where I want to store check point files (e.g. '/tmp/od/checkpoint_and_summaries')
* Set 'pipeline_config_path' flag to the full path of the config file (e.g. '../samples/configs/faster_rcnn_resnet50_coco.config')

### Step 5.3. Label map file       
You need the file to create a human-readable label to an integer.
Under the objection_detection/data directory, there are many examples that you can use as a base.
In my case, I created a file called dog_label_map.pbtxt which contains below:

```
item {
  id: 1
  name: 'aimee'
}

item {
  id: 2
  name: 'pink'
}

item {
  id: 3
  name: 'olivia'
}
```

### Step 5.4. Running the training script
Once you are done with all the changes, you can just run the training script.
I trained the model for a few hours and stopped training at 30956 step with loss = 0.0803.

During the training, you might want to make sure that check point files are created in the directory you specified in your
training script.

## Step 6. Convert the model to be used for prediction
Once you are done with training, you need to convert the model to a form that the prediction script can process.

You'll be using a script included in a source tree. I made a copy of object_detection/export_inference_graph.py and saved the copy as export_inference_graph_dog.py.

You can also refer to [exporting_models.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md) in the source tree for more details.

The set up I did is very similar to what I did to run the training script:
* Added additional Python Paths to include ".." and "_&lt;the path where I cloned the repo&gt;_/models/research/slim"
* Set 'pipeline_config_path' flag to the full path of the config file (e.g. '../samples/configs/faster_rcnn_resnet50_coco.config')
* Set the 'output_directory' flag to a directory to output the model (e.g. '/tmp/od/exported_model')
* Set the latest checkpoint file in 'trained_checkpoint_prefix' flag to '/tmp/od/checkpoint_and_summaries/model.ckpt-29463'

This one needs a little explanation:
if you type ls in a directory where you saved your checkpoint files, you'll see something like:
```
checkpoint                               model.ckpt-22086.index                model.ckpt-27001.meta
events.out.tfevents.1549336476.puppylin  model.ckpt-22086.meta                 model.ckpt-29463.data-00000-of-00001
graph.pbtxt                              model.ckpt-24544.data-00000-of-00001  model.ckpt-29463.index
model.ckpt-19627.data-00000-of-00001     model.ckpt-24544.index                model.ckpt-29463.meta
model.ckpt-19627.index                   model.ckpt-24544.meta                 pipeline.config
model.ckpt-19627.meta                    model.ckpt-27001.data-00000-of-00001
model.ckpt-22086.data-00000-of-00001     model.ckpt-27001.index
```
In my case, model.ckpt-29463 is the prefix for the latest checkpoint file, so I specified the path to this directory
as well as this prefix.

## Step 7. Run prediction with the model
I made a copy of the tutorial script which I converted from the Jupyter notebook in step 2.
I saved it as object_detection_dog.py.

Changes are the following:
1. Changed MODEL_NAME from 'ssd_mobilenet_v1_coco_2017_11_17' to a directory that I specified in the previous step (e.g. '/tmp/od/exported_model')

In the original code you'll see the following,
```
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
```

By redefining MODEL_NAME, you are instructing the prediction script to read the model that you have exported.


2. Got rid of the code to download the model.
3. Set PATH_TO_LABELS to the full path of my label map file.
4. Added code to read a test image one at a time, predict and save.

If you want to process a video like I did, you can just reassemble the processed image using ffmpeg.

That's it. Actual work besides annotating the dataset should take less than a day.

# References
&#91;1&#93; Priyanka Kochhar, Building a Toy Detector with Tensorflow Object Detection API, https://www.kdnuggets.com/2018/02/building-toy-detector-tensorflow-object-detection-api.html

&#91;2&#93; Dat Tran, How to train your own Object Detector with TensorFlow’s Object Detector API, https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

&#91;3&#93; Harrison, Training Custom Object Detector - Tensorflow Object Detection API Tutorial, https://pythonprogramming.net/training-custom-objects-tensorflow-object-detection-api-tutorial/?completed=/creating-tfrecord-files-tensorflow-object-detection-api-tutorial/

&#91;4&#93; Harrison, Testing Custom Object Detector - Tensorflow Object Detection API Tutorial, https://pythonprogramming.net/testing-custom-object-detector-tensorflow-object-detection-api-tutorial/?completed=/training-custom-objects-tensorflow-object-detection-api-tutorial/

&#91;5&#93; BalA VenkatesH, TensorFlow object detection with custom objects, https://medium.com/coinmonks/tensorflow-object-detection-with-custom-objects-34a2710c6de5
