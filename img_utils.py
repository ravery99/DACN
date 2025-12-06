import numpy as np
import imageio.v2 as imageio   

def imsave(image, path):
    """
    Convert 2-class mask (0/1) ke RGB dan simpan PNG.
    0 -> (0,0,0)
    1 -> (255,255,255)
    """
    label_colours = [(0,0,0), (255,255,255)]
    h, w = image.shape
    images = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            idx = int(image[i,j])
            if idx >= len(label_colours):
                idx = 0  # safety
            images[i,j] = label_colours[idx]

    imageio.imwrite(path, images)
