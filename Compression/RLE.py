import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import SimpleITK as sitk
import math

image = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Compression/Img2.png', 0)


def ComprimirLRE_HV(img, modo):
    # modos: 0 horizontal, 1 vertical

    # Horizontal
    if modo == 0:
        modo = "Horizontal"
        fil, col = img.shape
        dtype = img.dtype
        listaRLE = [[fil, col, modo, dtype],[]]

        value = img[0, 0]
        cant = 0

        for i in range(fil):
            for j in range(col):
                gris = img[i, j]
                if gris == value:
                    cant += 1
                else:

                    """
                    listaRLE.append(value)
                    listaRLE.append(cant)
                    """
                    listaRLE[1].append([value,cant])

                    value = gris
                    cant = 1
        """
        listaRLE.append(value)
        listaRLE.append(cant)
        """
        listaRLE[1].append([value, cant])

    return listaRLE


comp = ComprimirLRE_HV(image, 0)
print(comp)
