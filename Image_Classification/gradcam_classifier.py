import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.preprocessing import image

tf.get_logger().setLevel("ERROR")

model = MobileNetV2(weights="imagenet")

# MobileNetV2's final convolutional layer.
LAST_CONV_LAYER = "Conv_1"


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    prediction_index: int | None = None,
) -> np.ndarray:
    """Create a Grad-CAM heatmap for one prediction."""

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        last_conv_output, predictions = grad_model(img_array)

        if prediction_index is None:
            prediction_index = tf.argmax(predictions[0])

        predicted_class = predictions[:, prediction_index]

    gradients = tape.gradient(predicted_class, last_conv_output)

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    last_conv_output = last_conv_output[0]

    heatmap = last_conv_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)

    if maximum > 0:
        heatmap = heatmap / maximum

    return heatmap.numpy()


def save_gradcam_overlay(
    image_path: str,
    heatmap: np.ndarray,
    output_path: str = "gradcam_output.jpg",
    alpha: float = 0.4,
) -> None:
    """Overlay the Grad-CAM heatmap on the original image."""

    original_image = cv2.imread(image_path)

    if original_image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0]),
    )

    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        original_image,
        1 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    cv2.imwrite(output_path, overlay)


def classify_image(image_path: str) -> None:
    try:
        if not os.path.isfile(image_path):
            raise FileNotFoundError("The image file does not exist.")

        loaded_image = image.load_img(
            image_path,
            target_size=(224, 224),
        )

        image_array = image.img_to_array(loaded_image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        predictions = model.predict(image_array)
        decoded_predictions = decode_predictions(
            predictions,
            top=3,
        )[0]

        print("\nTop-3 Predictions for", image_path)

        for number, (_, label, score) in enumerate(
            decoded_predictions,
            start=1,
        ):
            print(f"{number}: {label} ({score:.2f})")

        predicted_index = int(np.argmax(predictions[0]))

        heatmap = make_gradcam_heatmap(
            image_array,
            model,
            LAST_CONV_LAYER,
            predicted_index,
        )

        output_path = "gradcam_output.jpg"

        save_gradcam_overlay(
            image_path,
            heatmap,
            output_path,
        )

        print(f"\nGrad-CAM image saved as: {output_path}")

    except Exception as error:
        print(f"Error processing '{image_path}': {error}")


if __name__ == "__main__":
    print("Image Classifier with Grad-CAM")
    print("Type 'exit' to quit.\n")

    while True:
        image_path = input("Enter image filename: ").strip()

        if image_path.lower() == "exit":
            print("Goodbye!")
            break

        classify_image(image_path)