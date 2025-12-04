import numpy as np
import imageio.v2 as imageio   

def imsave(image, path):

    label_colours = [(0,0,0), (255,255,255)]

    # buat output RGB image
    images = np.ones(list(image.shape) + [3], dtype=np.uint8)

    for j_, j in enumerate(image):
        for k_, k in enumerate(j):
            if k < 2:
                images[j_, k_] = label_colours[int(k)]

    imageio.imwrite(path, images)
