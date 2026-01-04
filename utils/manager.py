import numpy as np
from utils.preprocess import toGrayscale
from effects.dithering import FloydSteinbergDithering, AtkinsonDithering
from effects.aberration import AberrationFilter
from utils.preprocess import hsvInterpolation

class DitherControl:
    def __init__(self):
        pass

    def applyDithering(self, inputImage, method, noiseStr=0, lightColor=(255,255,255), darkColor=(0,0,0), steps=2):
        grayImg = toGrayscale(inputImage)
        palette = hsvInterpolation(darkColor, lightColor, steps)

        if method == "Original":
            return inputImage

        if method == "Floyd-Steinberg":
            return FloydSteinbergDithering.imgFromGray(grayImg, noiseStr, palette)
        
        if method == "Atkinson":
            return AtkinsonDithering.imgFromGray(grayImg, noiseStr, palette)

class AberationControl:
    def __init__(self):
        pass

    def applyAnaglyph(self, inputImage, offset):
        return AberrationFilter.applyAnaglyph(inputImage, offset)
        
        
    