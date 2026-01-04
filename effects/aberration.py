from numba import jit
import numpy as np

class AberrationFilter:
    @jit(nopython=True)
    def applyAnaglyph(img, offset):
        if offset == 0:
            return img
        
        height, width, _ = img.shape
        outputImg = np.zeros_like(img)

        for y in range(height):
            for x in range(width):
                xRed = x - offset
                if xRed < 0:
                    r = 0

                outputImg[y, x, 0] = img[y, xRed, 0]

                xCyan = x + offset
                if xCyan >= width:
                    xCyan = width - 1

                outputImg[y, x, 1] = img[y, xCyan, 1]
                outputImg[y, x, 2] = img[y, xCyan, 2]
        
        return outputImg