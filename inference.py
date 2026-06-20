# untuk:
# 1. load checkpoint
# 2. load image
# 3. resize 256x256
# 4. run model
# 5. simpan mask

import cv2
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from actions import Actions
from main import configure

def load_model():
    conf = configure()
    # conf.test_epoch = 16   # TODO: ganti sesuai best epoch

    sess = tf.Session()

    model = Actions(sess, conf)
    model.reload(conf.test_epoch)

    return sess, model

def run(image_path):
    sess, model = load_model()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mask = model.predict_image(image)

    cv2.imwrite(
        "mask.png",
        (mask * 255).astype("uint8")
    )

    print(mask.shape)
    sess.close()
    
if __name__ == "__main__":
    run("test.BMP")