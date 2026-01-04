import numpy as np
from numba import jit
import random

@jit(nopython=True)
def addNoise(oldPixel, noiseStr):
    if noiseStr > 0:
        noise = (random.random() - 0.5) * 2 * noiseStr
        oldPixel += noise
        oldPixel = min(255, max(0, oldPixel))

    return oldPixel

class FloydSteinbergDithering:
    @jit(nopython=True)
    def floydSteinbergDithering(grayImage, noiseStr, palette):
        height, width = grayImage.shape
        outputImg = np.zeros((height, width, 3), dtype=np.uint8)
        maxIdx = len(palette) - 1

        for y in range(height):
            for x in range(width):
                oldPixel = grayImage[y, x]
                oldPixel = addNoise(oldPixel, noiseStr)

                idx = int((oldPixel / 255) * (len(palette) - 1) + 0.5)

                if idx < 0: idx = 0
                if idx > maxIdx: idx = maxIdx

                color = palette[idx]
                outputImg[y, x, 0] = color[0]
                outputImg[y, x, 1] = color[1]
                outputImg[y, x, 2] = color[2]

                quantValue = idx * (255.0 / maxIdx)
                quantError = oldPixel - quantValue

                if x + 1 < width:
                    grayImage[y, x + 1] += quantError * 7 / 16
                if x - 1 >= 0 and y + 1 < height:
                    grayImage[y + 1, x - 1] += quantError * 3/16
                if y + 1 < height:
                    grayImage[y + 1, x] += quantError * 5 / 16
                if x + 1 < width and y + 1 < height:
                    grayImage[y + 1, x + 1] += quantError * 1 / 16

        return outputImg


    def imgFromGray(grayImage, noiseStr, palette):
        workCopy = grayImage.astype(np.float32)
        return FloydSteinbergDithering.floydSteinbergDithering(workCopy, noiseStr, palette)

class AtkinsonDithering:
    @jit(nopython=True)
    def atkinsonDithering(grayImage, noiseStr, palette):
        height, width = grayImage.shape
        outputImg = np.zeros((height, width, 3), dtype=np.uint8)
        maxIdx = len(palette) - 1

        for y in range(height):
            for x in range(width):
                oldPixel = grayImage[y, x]
                oldPixel = addNoise(oldPixel, noiseStr)

                idx = int((oldPixel / 255) * (len(palette) - 1) + 0.5)
                if idx < 0: idx = 0
                if idx > maxIdx: idx = maxIdx

                color = palette[idx]
                outputImg[y, x, 0] = color[0]
                outputImg[y, x, 1] = color[1]
                outputImg[y, x, 2] = color[2]

                quantValue = idx * (255.0 / maxIdx)
                quantError = oldPixel - quantValue

                if x + 1 < width:
                    grayImage[y, x + 1] += quantError * 1 / 8
                if x + 2 < width:
                    grayImage[y, x + 2] += quantError * 1 / 8
                if x - 1 >= 0 and y + 1 < height:
                    grayImage[y + 1, x - 1] += quantError * 1 / 8
                if y + 1 < height:
                    grayImage[y + 1, x] += quantError * 1 / 8
                if x + 1 < width and y + 1 < height:
                    grayImage[y + 1, x + 1] += quantError * 1 / 8
                if y + 2 < height:
                    grayImage[y + 2, x] += quantError * 1 / 8

        return outputImg

    def imgFromGray(grayImage, noiseStr, palette):
        workCopy = grayImage.astype(np.float32)
        return AtkinsonDithering.atkinsonDithering(workCopy, noiseStr, palette)