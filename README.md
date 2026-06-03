# inkling

inkling is a convolutional neural network used for classifying drawings. It was created using numpy/tensorflow/keras and trained on a subset of Google's QuickDraw dataset. It includes visualizations of training information using matplotlib/scikit-learn, as well as a demo with gradio.

I created this project as a first step into machine learning. I used the QuickDraw dataset because it was easy to understand, familiar and fun!

## Quick Start

### Training
1. Download the numpy bitmap files corresponding to the categories listed in the Knowledge base from the [Google QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) and place them in the data directory.
2. Create a virtual environment with the Python version dictated in .python-version and activate it
3. Install dependencies with `pip -r requirements.txt`.
3. Run train.py with `python train.py`.

### Demonstration
This project comes with a sample model, trained with the parameters used in the repo. To use the demonstration, run the demo.py script.

## Knowledge Base

### Drawings
inkling currently supports these drawing categories:
* apple
* carrot
* cat
* house
* umbrella
* airplane
* clock
* cloud
* star
* tree

More categories can be added by downloading the corresponding numpy bitmap files, adding the new categories in shared.py and retraining the model.

### Concepts
Convolutional Neural Networks are systems designed to identify spatial relationships between pixels, inspired by the human visual system. They are composed of the following layers:
* **Input layers** receive raw image data.
* **Convolution layers** scans the input data using filters and extracts features to output a feature map.
* **Activation layers** introduce non-linearity into the network to allow models to learn more complex patterns. Examples of activation functions include ReLU and Leaky ReLU.
* **Pooling layers** reduce the dimensions of feature maps to prevent overfitting (memorization of training data) and reduce resource usage.
* **Flattening layers** convert the multi-dimensional feature maps into a one-dimensional vector.
* **Dense layers** performs reasoning and produces final classification scores.
* **Output layers** convert final scores into probabilities using activation functions such as softmax.

### Useful Resources
* [Introduction to Convolution Neural Network by GeeksForGeeks](https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/)
* [Convolution Neural Network (CNN) by TensorFlow](https://www.tensorflow.org/tutorials/images/cnn)

## Images
![Demo](/docs/assets/demo.png)
![Training Data](/docs/assets/training_data.png)
