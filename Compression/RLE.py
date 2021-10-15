import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import SimpleITK as sitk
import math

comp = ComprimirHV(ortopanto,0)
print(comp)

def ComprimirHV(img, modo):
    # modos: 0 horizontal, 1 vertical

    fil, col = img.shape
    dtype = img.dtype
    listaRLE = [fil, col, modo, dtype]

    # Horizontal
    if modo == 0:
        value = img[0, 0]
        cant = 0
        for i in range(fil):
            for j in range(col):
                gris = img[i, j]
                if gris == value:
                    cant += 1
                else:
                    listaRLE.append(value)
                    listaRLE.append(cant)
                    value = gris
                    cant = 1
        listaRLE.append(value)
        listaRLE.append(cant)

    return listaRLE