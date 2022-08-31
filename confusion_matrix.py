import os
import io

from typing import Union
import numpy as np
from absl import app, flags
from matplotlib import pyplot as plt

import tensorflow as tf
from detection_utils import LabelMapUtil, Detector

# to test

# tfrecord
# python Tensorflow/scripts/confusion_matrix.py --model_dir=Tensorflow/workspace/models/complete_centernet_resnet --input_file=Tensorflow/workspace/annotations/test_added.record
# images
# python Tensorflow/scripts/confusion_matrix.py --model_dir=Tensorflow/workspace/models/complete_centernet_resnet --input_file=confusion_test
# single image
# python Tensorflow/scripts/confusion_matrix.py --model_dir=Tensorflow/workspace/models/complete_centernet_resnet --input_file=confusion_test/000021.jpg

flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
                       'where event and checkpoint files will be written.')
flags.DEFINE_integer(
    'checkpoint_nr', -1, 'Model ckeckpoint to load. -1 for latest.')
flags.DEFINE_string(
    'input_file', 'default', 'Path to the file to read the test data from. '
                             'Can be a .tfrecord file, a folder with .jpg and .xml '
                             'or a single .jpg (with the .xml in the same folder).'
                             'Can also use \'default\' to use the file from config.')
flags.DEFINE_bool('test', False, 'This script is usually run on the val set'
                  'If run on test set set this to true so the results do not get overwritten')

FLAGS = flags.FLAGS

def main(unused_argv):
    flags.mark_flag_as_required('model_dir')

    matrix_creator = ConfusionMatrix(FLAGS.model_dir, checkpoint_nr=FLAGS.checkpoint_nr)

    matrix_creator.evaluate(FLAGS.input_file)
    
    matrix_creator.print_matrices_to_terminal()
    
    matrix_creator.save_results_as_images()

class ConfusionMatrix:
    def __init__(self, model_dir: str, checkpoint_nr: int = -1) -> None:
        """Loads the label map and initializes the detector and the matrix

        Args:
            model_dir (str): Path to the model folder
            checkpoint_nr (int, optional): Checkpoint to load. -1 for latest. Defaults to -1.
        """
        
        # self.labels is dict 'class_name': class_id
        self.labels = LabelMapUtil().parse_label_map_from_config(os.path.join(model_dir, 'pipeline.config'), from_path=True)

        self.detector = Detector(model_dir, checkpoint_nr)

        self.reset_matrices()

    def reset_matrices(self) -> None:
        """This function creates the different confusion matices.
        
        1st dimension is true label and 2nd dimension is predicted label
        """
        
        size = len(self.labels)

        self.iou_matrix = np.empty((size, size), dtype=object)
        # not elegant but only solution i could find
        for i in range(size):
            for u in range(size):
                self.iou_matrix[i,u] = []
        
        self.confidence_matrix = np.empty((size, size), dtype=object)
        for i in range(size):
            for u in range(size):
                self.confidence_matrix[i,u] = []
        
        self.detected_matrix = np.empty(size, dtype=object)
        for i in range(size):
            self.detected_matrix[i] = []

    def evaluate(self, input_path: str) -> None:
        """Runs the given data through the model, compares it to ground truth and adds it to the matrices

        Args:
            input_path (str): Path to the dataset. See detector for possibilities
        """
        
        assert isinstance(input_path, str), 'input path needs to be str'
        
        self.detector.load_iterable_dataset(input_path)
                
        for predictions, image in self.detector:
            
            # iterate through all annotations to check if it was detected     
            for box in image.get_annotations():
                
                # convert the class name to it's id
                # id start at 1 but inexing is at 0 so -1
                annotation_id = self.labels[box.get_class()] - 1
                
                detected = False
                
                # compare the current annotation to all predictions
                for pred in predictions:
                    
                    # convert class name to id
                    # id start at 1 but inexing is at 0 so -1
                    prediction_id = self.labels[pred.get_class()] - 1
                    
                    # explication maybe use small threshold
                    iou = self.detector.calculate_iou(box, pred)
                    if iou > 0:
                        """
                        for the moment for each annotation all predictions that intersect with it are saved.
                        Saving means that the confidence and iou are added to a list that we will take the mean after.
                        
                        This is not optimal because a detection with a low confidence modifies the iou as much as a detection with a high confidence.
                        This also works the other was around: A detection with a low iou (maybe a detection that belongs to a different annotaion) modifies the confidence as much as a detection with a high iou.
                        A simple improvement would be to multiply iou and confidence to judge the combined effect.
                        
                        Over the large number of samples these effects are less important.  
                        
                        Plus for each annotation it is saved if it was correctly detected
                        a box counts as detected if one detection has the correct class and an iou of more than 50%
                        """  
                        
                        confidence = pred.get_confidence()
                        self.confidence_matrix[annotation_id, prediction_id].append(confidence)
                        self.iou_matrix[annotation_id, prediction_id].append(iou)
                        
                        if iou > .5 and annotation_id == prediction_id:
                            detected = True
            
                self.detected_matrix[annotation_id].append(detected)
                
    def print_matrices_to_terminal(self) -> None:
        """This function simply prints the result matrices to the terminal
        """
        
        iou_matrix, confidence_matrix, detected_matrix = self.get_mean_matrices()
        print(' confusion matrix results '.center(37, '-'), '\n')
        print('iou confusion matrix')
        print(iou_matrix, '\n')
        print('confidence confusion matrix')
        print(confidence_matrix, '\n')
        print('detection matrix')
        print(detected_matrix, '\n')

    def get_figures(self, matrix: np.ndarray, ticks: Union[list[str], str], title: str) -> plt.figure:
        """Creates a figure from the given matrix. The ticks for the firste dimension are the class names and the ticks for the other dimension need to be given.

        Args:
            matrix (np.ndarray): The matrix with the values
            ticks (list[str]): list of the ticks to put on the second axis. Needs to match matrix dimensions. Set to 'class_names' to use the class names.
            title (str): Title of the image

        Returns:
            plt.figure: matplotlib figure
        """
        
        assert isinstance(matrix, np.ndarray), 'matrix needs to be ndarray'
        assert isinstance(title, str), 'title needs to be a string'
        assert isinstance(ticks, list) and all(isinstance(x, str) for x in ticks) or ticks == 'class_names', 'ticks needs to be a list of strings or "class_names"'
        
        height = matrix.shape[0]
        if len(matrix.shape) == 1:
            width = 1
            matrix = np.reshape(matrix, (height, 1))
        elif len(matrix.shape) == 2:
            width = matrix.shape[1]
        else: assert False, 'matrix needs to be 1D or 2D'
            
        
        assert height == len(self.labels), 'matrix height needs to match labels'
        assert width == len(ticks) or ticks == 'class_names' and width == len(self.labels), 'matrix width needs to match label size'
        
        # get class names from labels in order
        class_names = []
        for i in range(len(self.labels)):
            for key in self.labels:
                if self.labels[key] == i+1:
                    class_names.append(key)

        figure = plt.figure(figsize=matrix.shape)
        plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)

        if ticks == 'class_names':
            indices = np.arange(width)
            plt.xticks(indices, class_names, rotation=45)
        else:
            indices = np.arange(width)
            plt.xticks(indices, ticks, rotation=45)
        
        indices = np.arange(height)
        plt.yticks(indices, class_names)

        for line in range(height):
            for column in range(width):
                plt.text(
                    column, line, round(matrix[line, column], 3), horizontalalignment="center", color='gray',
                )

        plt.tight_layout()
        plt.xlabel("Pred Label")
        plt.ylabel("True label")

        return figure
    
    def mean(self, lis: list[float]) -> float:
        """Returns the mean of values in a list. Returns 0 if list empty.

        Args:
            lis (list[float]): List of values to calculate the mean of

        Returns:
            float: mean
        """
        
        assert isinstance(lis, list) and all(isinstance(x, float) for x in lis), 'lis needs to be list of floats'
        
        if not lis:
            return 0.0
        
        return sum(lis) / len(lis)
        
    def get_mean_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """This function returns the matrices but replaces the lists with it's means

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: iou matrix, confidence matrix and detected matrix with means in that order
        """
        
        # vecotize mean function
        mean = np.vectorize(self.mean)
        
        return mean(self.iou_matrix), mean(self.confidence_matrix), mean(self.detected_matrix)
        
    def save_results_as_images(self, to_drive=False, to_tensorboard=True) -> None:
        """Saves the results as images either as png or in tensorboard

        Args:
            to_drive (bool, optional): If the files should be saved as pngs. Defaults to False.
            to_tensorboard (bool, optional): If the files should be saved to tensorboard. Defaults to True.
        """
        
        assert isinstance(to_drive, bool), 'to_drive needs to be bool'
        assert isinstance(to_tensorboard, bool), 'to_tensorboard needs to be bool'
        
        iou_matrix, confidence_matrix, detected_matrix = self.get_mean_matrices()
        
        confidence_figure = self.get_figures(confidence_matrix, 'class_names', 'Confidence confusion matrix')
        iou_figure = self.get_figures(iou_matrix, 'class_names', 'IoU confusion matrix')
        detected_figure = self.get_figures(detected_matrix, ['detected'], 'Detected percent')
        
        if to_tensorboard:
            model_path = self.detector.model_dir
            
            if not FLAGS.test:
                writer = tf.summary.create_file_writer(os.path.join(model_path, 'eval'))
            else:
                writer = tf.summary.create_file_writer(os.path.join(model_path, 'test'))
                
            with writer.as_default():
                tf.summary.image(
                    'Confidence confusion matrix',
                    self.plot_to_image(confidence_figure),
                    step=self.detector.step,
                )
                tf.summary.image(
                    'IoU confusion matrix',
                    self.plot_to_image(iou_figure),
                    step=self.detector.step,
                )
                tf.summary.image(
                    'Detected %',
                    self.plot_to_image(detected_figure),
                    step=self.detector.step
                )
            print('Saved images to tensorboard')
        if to_drive:
            confidence_figure.savefig('confidence_cm.png')
            iou_figure.savefig('iou_cm.png')
            detected_figure.savefig('detected.png')
            print('Saved images to drive')
    
    
    def plot_to_image(self, figure: plt.figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        
        borrowed from tensorflow official guide:
        https://www.tensorflow.org/tensorboard/image_summaries
        
        Returns:
            type: TF image to send to tensorboard
        """

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        figure.savefig(buf, format="png")

        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

if __name__ == '__main__':
    app.run(main)