from tabulate import tabulate
from imageio import imread, imwrite
from skimage.transform import radon, iradon, rescale, rotate
from numpy.fft import fft2, fftshift, ifft2
from skimage.color import rgb2gray
from skimage import data, transform, color
from skimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.morphology import binary_hit_or_miss, binary_closing, binary_opening
from skimage.segmentation import active_contour
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set()
from PIL import Image
from copy import copy
import time
from skimage.measure import label, regionprops, regionprops_table
from skimage import measure


def binarizacion(im,umbral,umbral2):
  M = im.shape[0]
  N = im.shape[1]
  res = np.zeros([M,N])

  for i in range(len(im)):
    for j in range(len(im[0])):
      if im[i,j] > umbral and im[i,j]<umbral2:
        res[i,j] = 255
      else:
        res[i,j] = 0
  img_out = res.astype(np.uint8)
  plt.figure(figsize=(15,15))
  plt.subplot(121),plt.imshow(im,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(img_out,cmap='gray'),plt.title('Imagen Binarizada')
  plt.show()

  return img_out

def otsu(img):
  umbral, img_out = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  plt.figure(figsize=(10,10))
  plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(img_out,cmap='gray'),plt.title('Imagen Binarizada con Otsu')
  plt.show()
  return umbral, img_out

def ero(img,it):
  kernele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
  c1 = img.astype("uint8")
  img_out = cv2.erode(c1, kernele, iterations=it)
  plt.figure(figsize=(10,10))
  plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(img_out,cmap='gray'),plt.title('Imagen Erosionada')
  plt.show()
  return img_out

def dil(img,it):
  kernele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
  c1 = img.astype("uint8")
  img_out = cv2.dilate(c1, kernele, iterations=it)
  plt.figure(figsize=(10,10))
  plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(img_out,cmap='gray'),plt.title('Imagen Dilatada')
  plt.show()
  return img_out

def close(img, it):
  kernele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
  c1 = img.astype("uint8")
  img1 = cv2.dilate(c1, kernele, iterations=it)
  img_out = cv2.erode(img1, kernele, iterations=it)
  plt.figure(figsize=(10,10))
  plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(img_out,cmap='gray'),plt.title('Imagen Cerrada')
  plt.show()
  return img_out

def open(img, it):
  kernele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
  c1 = img.astype("uint8")
  img_out = cv2.erode(img, kernele, iterations=it)
  img1 = cv2.dilate(c1, kernele, iterations=it)
  plt.figure(figsize=(10,10))
  plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(img_out,cmap='gray'),plt.title('Imagen Abierta')
  plt.show()
  return img_out

def etiquetadoybb(img,img_original):
  c, labels = cv2.connectedComponents(img)

  #propiedades labels etiquetado
  regions = regionprops(labels)
  props = measure.regionprops_table(labels,img,properties=['label','area','equivalent_diameter','extent', 'centroid'])

  import pandas as pd
  df= pd.DataFrame(props)
  print(df)

  #conteo de linfocitos
  linfocitos = 0
  labels_lengthx = labels.shape[0]
  labels_lengthy = labels.shape[1]
  location = [] #ubicación

  for i in range (c-1):
    if (regions[i].area>15000):
      if (regions[i].extent > 0.65):
       linfocitos=linfocitos+1
       location.append(regions[i].centroid)

  print("Las linfocitos son: ", linfocitos)
  #print("El paciente presenta ", falciformes, "de ", total, "por lo tanto presenta un ", porcentaje, "%")
  print("Las posiciones son: ", location)

  fig, ax = plt.subplots(figsize=(10, 6))

  ax.imshow(img_original,cmap=plt.cm.nipy_spectral)

  #ESTO SIRVE PARA PONER EL CUADRADITO PERO HABRIA QUE LOGRAR SEPARAR MAS. Esto es indepe
  pos=np.zeros(shape=(1,c-1));

  #SACANDO LAS DE LOS BORDES>
  cuadrados = []
  for region in regionprops(labels):
    if (region.area>15000):
      if(region.extent > 0.65):
     # draw rectangles
        minr, minc, maxr, maxc = region.bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='white', linewidth=3)
        ax.add_patch(rect)
        cuadrados.append(rect)
  return cuadrados

def boundingBoxP(img, labels, c):
  ROI_number = 0

  regions = regionprops(labels)
  props = measure.regionprops_table(labels,img,properties=['label','area','equivalent_diameter','perimeter', 'centroid'])

  import pandas as pd
  df= pd.DataFrame(props)
  print(df)

  linfocitos = 0
  labels_lengthx = labels.shape[0]
  labels_lengthy = labels.shape[1]
  location = [] #ubicación

  for i in range (c-1):
    if (regions[i].area>15000):
      #if (abs(regions4[i].equivalent_diameter*3.14 - regions4[i].perimeter) < 500):
       linfocitos=linfocitos+1
       location.append(regions[i].centroid)

  print("Las linfocitos son: ", linfocitos)
  #print("El paciente presenta ", falciformes, "de ", total, "por lo tanto presenta un ", porcentaje, "%")
  print("Las posiciones de las falciformes son: ", location)

  cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  imagen2 = img.copy()
  original = imagen2.copy()

  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      cv2.rectangle(imagen2, (x, y), (x + w, y + h), (255,255,255), 5)  # parámetros, img, punto inicial, punto final, color, espesor
      ROI = original[y:y+h, x:x+w]

  plt.figure(figsize=(15,15))
  plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(imagen2,cmap='gray'),plt.title('Bounding Box')
  plt.show()
  return cnts,imagen2

def boundingBox(img):
  ROI_number = 0

  cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  imagen2 = img.copy()
  original = imagen2.copy()

  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      cv2.rectangle(imagen2, (x, y), (x + w, y + h), (255,255,255), 5)  # parámetros, img, punto inicial, punto final, color, espesor
      ROI = original[y:y+h, x:x+w]

  plt.figure(figsize=(15,15))
  plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(imagen2,cmap='gray'),plt.title('Bounding Box')
  plt.show()
  return cnts,imagen2

def Etiquetado2(imagen_binarizada, umbral):
  M = imagen_binarizada.shape[0]
  N = imagen_binarizada.shape[1]

  # Zero padding para ahorrar el problema con los bordes
  imagen_pad = np.zeros((M + 2, N + 2))
  imagen_pad[1:M + 1, 1:N + 1] = imagen_binarizada

  # Identificar las coordenadas de los píxeles que son iguales a 1 o 255
  Coordenadas = np.where((imagen_pad == 1) | (imagen_pad == 255))  # Blanco en imágen binaria o 8 bits
  Coordenadasx = Coordenadas[0]
  Coordenadasy = Coordenadas[1]
  etiquetas = np.zeros(imagen_pad.shape)

  # Defino una variable para enumerar las etiquetas y una lista donde guardo las etiquetas usadas
  e = 1
  eti = []

  for i in range(len(Coordenadasx)):

    # Saco las coordenadas de un primer pixel blanco
    x = Coordenadasx[i]
    y = Coordenadasy[i]

    if imagen_pad[x, y] == 1 or imagen_pad[x, y] == 255:
      # Defino una lista para guardar la etiqueta a la que pertenecen los vecinos etiquetados
      vecino_etiquetado = []
      vecino_no_etiquetado = []

      # Recorro la vecindad de 8 y pregunto si alguno de estos pertenece a una etiqueta dentro del diccionario
      for m in range(x - 1, x + 2):
        for n in range(y - 1, y + 2):
          if (imagen_pad[m, n] == 1 or imagen_pad[m, n] == 255) and ((m, n) != (x, y)):
            etiqueta_vecino = etiquetas[m, n]

            if etiqueta_vecino:  # Si pertenece lo agrego a vecino_etiquetado
              vecino_etiquetado.append(etiqueta_vecino)

      vecino_etiquetado = list(set(vecino_etiquetado))  # Ordeno de menor a mayor y elimino duplicados
      L = len(vecino_etiquetado)

      # Si vecino_etiquetado está vacío creo una nueva etiqueta y aumento en 1 la numeración para la próxima iteración
      if not L:
        etiquetas[x, y] = e
        eti = np.append(eti, e)
        e += 1

      # Si hay un solo vecino etiquetado agrego el pixel a esa etiqueta
      elif L == 1:
        etiquetas[x, y] = vecino_etiquetado[0]

      # Si hay más de uno agrego el pixel actual y todos los pixeles que pertenecen a otras etiquetas a la etiqueta de menor índice.
      # Ej: si tengo un vecino que pertenece a etiqueta 1 y otro a etiqueta 2 meto mi pixel actual y todos los pixeles en etiqueta 2 a etiqueta 1.
      elif L > 1:
        etiquetas[x, y] = vecino_etiquetado[0]

        for z in vecino_etiquetado[1::]:
          etiquetas[etiquetas == z] = vecino_etiquetado[0]

        eti = [n for n in eti if n not in vecino_etiquetado[1::]]

        eti_escalado = np.arange(1, len(eti) + 1)

        for (m, n) in zip(eti, eti_escalado):
          etiquetas[etiquetas == m] = n

        e = eti_escalado[-1] + 1
        eti = eti_escalado

  # Creo la imágen a color
  colores_usados = []  # Lista para evitar crear colores repetidos
  Imagen_Color = np.zeros((M, N, 3), np.uint8)

  etiquetas = list(np.ravel(etiquetas))
  etiquetas = [i for i in etiquetas if i != 0]

  # Por cada etiqueta creo un color aleatorio y corroboro que no esté repetido
  for e in eti:
    Repetido = True
    while Repetido:
      R = np.random.randint(0, 256)
      G = np.random.randint(0, 256)
      B = np.random.randint(0, 256)
      if (R, G, B) not in colores_usados:
        Repetido = False

    colores_usados.append((R, G, B))
    x = np.array([i for (i, j) in zip(Coordenadasx, etiquetas) if j == e])
    y = np.array([i for (i, j) in zip(Coordenadasy, etiquetas) if j == e])

    if len(x) > umbral:
      Imagen_Color[x - 1, y - 1, :] = [R, G, B]

  D = {'COORX': Coordenadasx, 'COORY': Coordenadasy, 'Etiquetas': eti, 'COOR_Etiquetas': etiquetas}

  return Imagen_Color, D

def Bounding_Box(imagen_original, imagen_binarizada, DIC, grosor, umbral):
  M = imagen_original.shape[0]
  N = imagen_original.shape[1]

  Imagen_BB = np.zeros((M + grosor * 2, N + grosor * 2, 3), np.uint8)
  Imagen_BB[grosor:M + grosor, grosor:N + grosor, 0] = imagen_original
  Imagen_BB[grosor:M + grosor, grosor:N + grosor, 1] = imagen_original
  Imagen_BB[grosor:M + grosor, grosor:N + grosor, 2] = imagen_original

  Coordenadasx = DIC['COORX']
  Coordenadasy = DIC['COORY']
  Eti = DIC['Etiquetas']
  Etiquetas = DIC['COOR_Etiquetas']

  colores_usados = []

  for e in Eti:
    Repetido = True
    while Repetido:
      R = np.random.randint(0, 256)
      G = np.random.randint(0, 256)
      B = np.random.randint(0, 256)
      if (R, G, B) not in colores_usados:
        Repetido = False
    colores_usados.append((R, G, B))
    x = np.array([i for (i, j) in zip(Coordenadasx, Etiquetas) if j == e])
    y = np.array([i for (i, j) in zip(Coordenadasy, Etiquetas) if j == e])

    if len(x) > umbral:
      Borde_superior = grosor + min(x) - 1
      Borde_inferior = grosor + max(x) - 1
      Borde_izquierdo = grosor + min(y) - 1
      Borde_derecho = grosor + max(y) - 1

      # Bordes = [[Borde_superior]*A + [Borde_inferior]*A + list(range(Borde_superior,Borde_inferior+1)) * 2,
      #        list(range(Borde_izquierdo,Borde_derecho+1)) * 2 + [Borde_izquierdo]*B + [Borde_inferior]*B]

      # Imagen_BB[Borde[0],Bordes[1],:] = [R,G,B]

      # Borde vertical izquierdo
      Imagen_BB[Borde_superior - grosor:Borde_inferior + grosor + 1, Borde_izquierdo - grosor:Borde_izquierdo + 1,
      :] = [R, G, B]
      # Borde vertical derecho
      Imagen_BB[Borde_superior - grosor:Borde_inferior + grosor + 1, Borde_derecho:Borde_derecho + grosor + 1, :] = [R,
                                                                                                                     G,
                                                                                                                     B]
      # Borde horizontal superior
      Imagen_BB[Borde_superior - grosor:Borde_superior + 1, Borde_izquierdo:Borde_derecho + 1, :] = [R, G, B]
      # Borde horizontal inferior
      Imagen_BB[Borde_inferior:Borde_inferior + grosor + 1, Borde_izquierdo:Borde_derecho + 1, :] = [R, G, B]

  Imagen_BB = Imagen_BB[grosor:M + grosor, grosor:N + grosor, :]

  return Imagen_BB

# IMPORTANTE
def kmeans(img):
  pixel_vals = img.reshape((-1))
  pixel_vals = np.float32(pixel_vals)

  # Definir criterio de corte = ( type, max_iter = 10 , epsilon = 1.0 )
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)

  # Definir centroides
  flags = cv2.KMEANS_RANDOM_CENTERS

  # Aplicamos kmeans
  compactness,labels,centers = cv2.kmeans(pixel_vals,3,None,criteria,100,flags)

  center = np.uint8(centers)
  img_kmeans = center[labels.flatten()]
  img_kmeans = img_kmeans.reshape((img.shape))

  center_blanco = np.max(center)

  plt.figure(figsize=(10,10))
  plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Imagen original')
  plt.subplot(122),plt.imshow(img_kmeans,cmap='gray'),plt.title('Imagen binarizada con Kmeans')
  plt.show()
  return img_kmeans, center_blanco




















