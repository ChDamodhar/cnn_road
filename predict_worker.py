# predict_worker.py
import json
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image


def run_isolated_inference(img_path):
    # Determine the model file to load
    # Highly recommended to load the full saved model file (.h5 or .keras)
    model_file = "road_damage_cnn_model.h5"

    if not os.path.exists(model_file):
        # Alternative fallback check if named differently
        model_file = "best_model.h5"

    try:
        # This automatically maps the exact layers, shapes, and weights
        model = tf.keras.models.load_model(model_file, compile=False)
    except Exception as e:
        sys.stderr.write(
            f"Error loading model: {str(e)}. Make sure the full model file exists in the directory.\n"
        )
        sys.exit(1)

    # Extract target shape dynamically from the model's first input layer
    try:
        input_shape = model.input_shape  # E.g., (None, 224, 224, 3)
        target_w = input_shape[1] if input_shape[1] is not None else 128
        target_h = input_shape[2] if input_shape[2] is not None else 128
    except:
        target_w, target_h = 128, 128

    # Preprocess incoming image using the matching dimensions
    img = Image.open(img_path).convert("RGB").resize((target_w, target_h))
    img_array = np.array(img).astype("float32") / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # Run prediction
    predictions = model.predict(img_tensor, verbose=0)[0]

    # Print out purely clean JSON output for app.py to collect
    print(json.dumps(predictions.tolist()))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_isolated_inference(sys.argv[1])
    else:
        sys.stderr.write("No image file path provided to worker.\n")
        sys.exit(1)