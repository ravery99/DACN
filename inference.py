# untuk:
# 1. load checkpoint
# 2. load image
# 3. resize 256x256
# 4. run model
# 5. simpan mask

import cv2
import tensorflow.compat.v1 as tf
import numpy as np
from pathlib import Path

tf.disable_v2_behavior()

from .actions import Actions
from .main import configure

BASE_DIR = Path(__file__).resolve().parent

def load_model():
    conf = configure()

    sess = tf.Session()

    model = Actions(sess, conf)
    model.reload(conf.test_epoch)

    return sess, model

def run(image_path: str):
    sess, model = load_model()

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = model.predict_image(image)

    output_path = BASE_DIR / "mask.png"
    cv2.imwrite(
        str(output_path),
        (mask * 255).astype("uint8")
    )

    print("Saved to:", output_path)
    sess.close()
       
if __name__ == "__main__":
    run(str(BASE_DIR / "test.BMP"))