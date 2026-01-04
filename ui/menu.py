from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QSlider,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QComboBox, QSizePolicy, QColorDialog, QCheckBox
)
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtCore import Qt

from PIL import Image
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

from utils.preprocess import numpyToQImage
from utils.manager import DitherControl, AberationControl

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()

        self.originalFull = None
        
        self.prevInput = None
        self.prevOutput = None

        self.lightColor = (226, 223, 195)
        self.darkColor = (41, 40, 29)

        self.initUI()

    def initUI(self):
        # load image button
        loadBtn = QPushButton("Load Image")
        loadBtn.clicked.connect(self.loadImage)

        # dithering method selection
        self.methodBox = QComboBox()
        self.methodBox.addItems(["Original", "Floyd-Steinberg", "Atkinson"])
        self.methodBox.currentIndexChanged.connect(self.applyEffect)

        # color picker
        self.lightColorBtn = QPushButton("Pick Light Color")
        self.lightColorBtn.setStyleSheet(f"background-color:{QColor(*self.lightColor).name()}; color: black;")
        self.lightColorBtn.clicked.connect(self.pickLightColor)

        self.darkColorBtn = QPushButton("Pick Dark Color")
        self.darkColorBtn.setStyleSheet(f"background-color:{QColor(*self.darkColor).name()}; color: white;")
        self.darkColorBtn.clicked.connect(self.pickDarkColor)

        # color steps
        self.stepsLabel = QLabel("Color Steps")
        self.stepsBox = QComboBox()
        self.stepsBox.addItems(["2", "4", "8", "16", "32", "64", "128", "256"])
        self.stepsBox.currentIndexChanged.connect(self.applyEffect)

        # noise strength slider
        self.noiseLabel = QLabel("Noise Strength: 0")
        self.noiseSlider = QSlider(Qt.Horizontal)
        self.noiseSlider.setMinimum(0)
        self.noiseSlider.setMaximum(50)
        self.noiseSlider.setValue(0)
        self.noiseSlider.valueChanged.connect(self.updateSliderLabel)
        self.noiseSlider.valueChanged.connect(self.applyEffect)

        # 3d aberration
        self.aberrationCheck = QCheckBox("Apply 3D Anaglyph")
        self.aberrationCheck.stateChanged.connect(self.applyEffect)

        self.slider3d = QSlider(Qt.Horizontal)
        self.slider3d.setMinimum(1)
        self.slider3d.setMaximum(10)
        self.slider3d.setValue(2)
        self.slider3d.valueChanged.connect(self.applyEffect)

        # widget layout
        controls = QHBoxLayout()
        controls.addWidget(loadBtn)
        controls.addWidget(self.methodBox)

        controls.addWidget(self.lightColorBtn)
        controls.addWidget(self.darkColorBtn)

        controls.addWidget(self.stepsLabel)
        controls.addWidget(self.stepsBox)

        controls.addWidget(self.noiseLabel)
        controls.addWidget(self.noiseSlider)

        controls.addWidget(self.aberrationCheck)
        controls.addWidget(self.slider3d)

        # output image display
        self.outputLabel = QLabel("Processed Image")
        self.outputLabel.setAlignment(Qt.AlignCenter)
        self.outputLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        imagesLayout = QHBoxLayout()
        imagesLayout.addWidget(self.outputLabel)

        # main layout
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(controls)
        mainLayout.addLayout(imagesLayout)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

    def pickLightColor(self):
        color = QColorDialog.getColor(
            initial=QColor(*self.lightColor),
            parent=self,
            options=QColorDialog.DontUseNativeDialog
            )

        if color.isValid():
            self.lightColor = (color.red(), color.green(), color.blue())
            self.lightColorBtn.setStyleSheet(f"background-color:{color.name()};")
            self.applyEffect()

    def pickDarkColor(self):
        color = QColorDialog.getColor(
            initial=QColor(*self.darkColor),
            parent=self,
            options=QColorDialog.DontUseNativeDialog
            )

        if color.isValid():
            self.darkColor = (color.red(), color.green(), color.blue())
            self.darkColorBtn.setStyleSheet(f"background-color:{color.name()};")
            self.applyEffect()

    def updateSliderLabel(self):
        value = self.noiseSlider.value()
        self.noiseLabel.setText(f"Noise Strength: {value}")

    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")

        if not filePath:
            return
        
        pilImg = Image.open(filePath).convert("RGB")

        self.originalFull = np.array(pilImg)

        pilImg.thumbnail((800, 800))
        
        self.prevInput = np.array(pilImg, dtype=np.float32)
        self.prevOutput = np.array(pilImg, dtype=np.float32)

        self.applyEffect()

    def applyEffect(self):
        if self.prevInput is None:
            return
        
        method = self.methodBox.currentText()
        noiseStr = self.noiseSlider.value()
        offset3d = self.slider3d.value()
        steps = int(self.stepsBox.currentText())

        self.prevOutput = DitherControl().applyDithering(
            self.prevInput,
            method,
            noiseStr,
            self.lightColor,
            self.darkColor,
            steps
            )
        
        if self.aberrationCheck.isChecked():
            self.prevOutput = AberationControl().applyAnaglyph(
                self.prevOutput,
                offset3d
            )


        self.updateDisplay()

    def updateDisplay(self):
        qImgPrevOutput = numpyToQImage(self.prevOutput)

        pixOutput = QPixmap.fromImage(qImgPrevOutput)

        wOutput = self.outputLabel.width()
        hOutput = self.outputLabel.height()

        if wOutput <= 0 or hOutput <= 0: return

        self.outputLabel.setPixmap(pixOutput.scaled(wOutput, hOutput, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.prevOutput is not None:
            self.updateDisplay()
