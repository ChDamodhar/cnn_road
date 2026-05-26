# predict_worker.py
import json
import os
import sys
import urllib.request
import numpy as np
import tensorflow as tf
from PIL import Image


def run_isolated_inference(img_path):
    model_file = "road_damage_cnn_model.h5"

    # Automatically pull down the model binary asset if running on the cloud server
    if not os.path.exists(model_file):
        # REPLACE THIS URL with your exact GitHub Release Asset download link
        url = "https://github.com/ChDamodhar/cnn_road/releases/download/v1.0.0/road_damage_cnn_model.h5"
        try:
            sys.stderr.write("Downloading CNN model weights from release assets...\n")
            urllib.request.urlretrieve(url, model_file)
        except Exception as download_err:
            sys.stderr.write(
                f"Failed downloading model binary: {str(download_err)}\n"
            )
            sys.exit(1)

    try:
        model = tf.keras.models.load_model(model_file, compile=False)
    except Exception as e:
        sys.stderr.write(f"Error loading model structural layout: {str(e)}\n")
        sys.exit(1)

    # Extract target shape dynamically from the model configuration matrix
    try:
        input_shape = model.input_shape
        target_w = input_shape[1] if input_shape[1] is not None else 128
        target_h = input_shape[2] if input_shape[2] is not None else 128
    except:
        target_w, target_h = 128, 128

    # Process input image frame
    try:
        img = Image.open(img_path).convert("RGB").resize((target_w, target_h))
        img_array = np.array(img).astype("float32") / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)
    except Exception as img_err:
        sys.stderr.write(f"Image preprocessing matrix error: {str(img_err)}\n")
        sys.exit(1)

    # Run tensor calculation matrix array
    predictions = model.predict(img_tensor, verbose=0)[0]

    # Return pure JSON output string directly back to stdout
    print(json.dumps(predictions.tolist()))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_isolated_inference(sys.argv[1])
    else:
        sys.stderr.write("No image path parameter provided to subprocess.\n")
        sys.exit(1)