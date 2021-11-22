import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import funciones as func

imagen1 = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Final Exercise GUI/Images/ALL_1.bmp', 0)

plt.imshow(imagen1, cmap="gray",vmin=0, vmax=255)
plt.title("La imagen Por Region Growing")
plt.show()