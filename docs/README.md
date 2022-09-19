# A better way for evaluating TFOD models

The script included in the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) (TFOD) for evaluating the trained models is not very convenient to use. The biggest "flaw" is that it is meant to run continuously. It will wait for a new checkpoint to appear and then it will evaluate the new checkpoint. While doing this the model is loaded in the computer memory. The problem is that when the script is run parallel with training, the whole model is loaded twice. Many users do not have the resources to load their models twice. 

This script offers a more convenient solution which is to run the script after training. The script will search for all available checkpoints and will evaluate them. As a bonus this script can also calculate confusion matrices for each model and save them to tensorboard.

## The low effort solution to evaluating all checkpoints

## How to use this script calculating mAP

## On confusion matrices for object detection

## Potential improvements