import numpy as np
import colorsys
from PySide6.QtGui import QImage

def numpyToQImage(data):
    height, width, channel = data.shape
    data = np.clip(data, 0, 255).astype(np.uint8)
    return QImage(data.data, width, height, channel * width, QImage.Format_RGB888)

def toGrayscale(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def hsvInterpolation(lightColor, darkColor, steps):
    palette = np.zeros((steps, 3), dtype=np.uint8)

    r1, g1, b1 = [x / 255.0 for x in lightColor]
    r2, g2, b2 = [x / 255.0 for x in darkColor]

    h1, s1, v1 = colorsys.rgb_to_hsv(r1, g1, b1)
    h2, s2, v2 = colorsys.rgb_to_hsv(r2, g2, b2)

    diff = h2 - h1
    if diff > 0.5:
        h1 += 1.0
    elif diff < -0.5:
        h2 += 1.0

    for i in range(steps):
        t = i / (steps - 1)

        h = (h1 + (h2 - h1) * t) % 1.0
        s = s1 + (s2 - s1) * t
        v = v1 + (v2 - v1) * t

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette[i] = [int(r * 255), int(g * 255), int(b * 255)]

    return palette