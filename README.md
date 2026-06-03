# inkling

inkling is a convolutional neural network used for classifying drawings, trained on Google's QuickDraw dataset with numpy/tensorflow/keras. It includes visualizations of training information using matplotlib/scikit-learn, a demo with gradio, an in-house data processing pipeline, and utilities for automating tedious processes (e.g. downloading datasets). It comes with a sample model, which supports 41 different categories (hand-picked by yours truly) at an accuracy of X%.

I created this project as a first step into machine learning. I used the QuickDraw dataset because it was easy to understand, familiar and fun!

## Quick Start

### Training
1. Create a Conda environment with `conda env create -f environment.yml`. Activate it with `conda activate main`.
2. Download the simplified dataset files corresponding to the categories listed in the Knowledge Base from the [Google QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) and place them in the data/raw directory. You can automate this with `python download_data.py`.
3. Process the data with `python process_data,py`.
3. Run train.py with `python train.py`.

### Demonstration
This project comes with a sample model, trained with the parameters used in the repo. To use the demonstration, run  `python demo.py`.

## Knowledge Base

### Configuration
To configure the project, you may change the values in these files:
* `shared.py` for data directory, raw data directory, processed data directory, model directory, categories, drawing count, validation fraction and image size
* `download_data.py` for endpoint, bucket name, dataset path and full API URL
* `process_data.py` for shards per category
* `train.py` for batch size and shuffle size
* `demo.py` for model name, model path, CSS file and no confidence threshold

### Drawings
inkling currently supports these drawing categories: apple, carrot, cat, house, umbrella, airplane, clock, cloud, star, tree, smiley face, table, pizza, book, computer, ice cream, floor lamp, key, pencil, flower, cake, snowflake, triangle, square, circle, hamburger, map, moon, pear, blueberry, bird, sock, zigzag, compass, cookie, fish, grapes, jail, lightning, leaf and snowman.

### Limitations
There are certain limitations with the current model architecture:
* There is no support for color, as the QuickDraw dataset only contains black and white images.
* The number of categories supported is purposefully small, as adding more categories decreases accuracy.

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
