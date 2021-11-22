from tkinter import *
from tkinter import filedialog
import PIL
from PIL import ImageTk,Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageTk
import cv2

"""
Cosas de practica:

root = Tk()
# Definir un texto
myLabel = Label(root, text="Hello world")
myLabel.pack()

texto = Entry(root, width=50)  # Casilla para ingresar texto
texto.pack()


# Definir un boton
# Podemos cambiarle el tamaño
# Para que tenga alguna actividad tenes que definir una funcion!
def onClick():
    label = Label(root, text="Hola! " + texto.get())
    label.pack()


# El parametro command define la funcion que tiene asignada
myButton = Button(root, text="Enter your name", padx=50, pady=20, command=onClick)
myButton.pack()

exit_button = Button(root, text="Exit", command=root.quit)
exit_button.pack()

# Para cargar imagenes!


cv_img = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Final Exercise GUI/Images/ALL_1.bmp',0)

photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
#height, width = cv_img.shape
canvas = tkinter.Canvas(root)#, width = width, height = height)
canvas.pack()

# Add a PhotoImage to the Canvas
canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)


root.mainloop()
"""

def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input image

    path = filedialog.askopenfilename()
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image_inicial = cv2.imread(path)
        image = cv2.resize(image_inicial, (400, 300))

        """
            La funcion que diseñamos va aqui!
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert the images to PIL format...
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)
        # if the panels are None, initialize them

        """
            Desde aqui solo se posicionan las imagenes
        """
        imagen_1 = LabelFrame(root, text = "Imagen 1")
        imagen_1.pack(side="left", padx=30, pady=30)
        imagen_2 = LabelFrame(root, text="Imagen 2")
        imagen_2.pack(side="right", padx=30, pady=30)

        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(imagen_1,image=image)
            panelA.image = image
            panelA.pack(side="left", padx=30, pady=30)
            # while the second panel will store the edge map
            panelB = Label(imagen_2,image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=30, pady=30)
        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged


# Desde aca empieza la GUI
root = Tk()
root.title("Trabajo final de PIB")
root.geometry("800x500")

myLabel = Label(root, text="Trabajo final de PIB", font=("Arial", 50)) # Ese titulo lo podemos cambiar
myLabel.pack()

# Inicialmente no hay imagen
panelA = None
panelB = None

# El boton que nos lleva a cargar las imagenes
btn = Button(root, text="Seleccionar una imagen para detectar", command=select_image, font=("Arial", 25))
btn.pack(fill="both", expand="yes", padx="5", pady="5")

# Exit
exit_button = Button(root, text="Exit", command=root.quit)
exit_button.pack(side="bottom",fill="both", expand="yes")

root.mainloop()