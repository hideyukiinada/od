# (Work-in-Progress page.  Not ready for consumption!)
Object Detection

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
```
git clone https//github.com/tensorflow/model
```

Download required packages as specified in the repo.
Converted the tutorial in the Jupyter notebook format to a regular Python file as I find it easier for me to work with.

I had a problem in matplotlib's plot to actually display the generated images, so I added code to save images on the file system.

## Step 3. Create your own dataset for object detection.
I used my iPhone and recorded a video of my dogs Aimee and Pink for about 5 minutes.  For 30 frames/second, this should result
in 9000 images if I export each frame.  For now, I plan on exporting 2 frames for each second which translates to 600 images.

## Step 4. Mark bounding boxes.
This is a two step process.
### Download and build labelimg

### Actually tag photos


## Step 7. Train the model.
I wasn't able to locate train.py at the object_detection directory where others listed in their articles.
I did a search on the internet and found out that the script was moved to the directory called legacy.


