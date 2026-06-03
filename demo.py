from shared import MODEL_DIR, IMAGE_SIZE, CATEGORIES
import random
import os.path as path
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import models
import gradio as gr

MODEL_NAME = "sample"
MODEL_PATH = path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
CSS_FILE = "./demo.css"

model = models.load_model(MODEL_PATH)

def generate_blank_canvas():
    return Image.new("L", (IMAGE_SIZE * 10, IMAGE_SIZE * 10), "white")

def generate_category():
    return random.choice(CATEGORIES)

def process_image(image):
    pillow_image = Image.fromarray(image)
    pillow_image = pillow_image.convert("L")
    pillow_image = pillow_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    np_image = np.array(pillow_image)
    np_image = 255 - np_image
    np_image = np_image / 255
    np_image = np_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return np_image

def predict(result):
    image = result["composite"]
    processed_image = process_image(image)

    logits = model.predict(processed_image)
    probabilities = tf.nn.softmax(logits[0]).numpy()
    predicted_index = np.argmax(probabilities)
    predicted_category = CATEGORIES[predicted_index]
    confidence = probabilities[predicted_index]

    if confidence >= 0.3:
        return f"{predicted_category} ({confidence:.2%})"
    else:
        return "No idea!"

def clear_all():
    return None, generate_blank_canvas(), None

with gr.Blocks() as demo:
    gr.Markdown("# Inkling")
    gr.Markdown("Inkling is a machine learning demo that tries to guess what you've drawn using a convolutional neural network trained on Google's QuickDraw dataset.")
    gr.Markdown("Generate a random category, sketch it, submit and inkling will try to classify your drawing into one of 41 categories. If it guesses wrong, either the model needs a bit more training or your drawing might be a bit *too* questionable. Don't worry; the model never sees the generated random category, so it's all fair :)")

    with gr.Row():
        with gr.Column(scale=1):
            category_output = gr.Text(
                label="Random category",
                interactive=False
            )
            generate_category_button = gr.Button("Generate")
            generate_category_button.click(fn=generate_category, outputs=category_output)
            prediction_output = gr.Text(
                label="Prediction"
            )

        with gr.Column(scale=2):
            sketchpad_input = gr.Sketchpad(
                label="Canvas",
                value=generate_blank_canvas(),
                image_mode="L",
                brush=gr.Brush(
                    default_size=8,
                    colors=["#000000"],
                    default_color="#000000",
                    color_mode="fixed"
                ),
                sources=[],
                buttons=[],
                transforms=[],
                layers=False
            )
            with gr.Row():
                submit_button = gr.Button("Submit")
                submit_button.click(fn=predict, inputs=sketchpad_input, outputs=prediction_output)
                clear_button = gr.Button("Clear")
                clear_button.click(fn=clear_all, outputs=[category_output, sketchpad_input, prediction_output])
        
demo.launch()