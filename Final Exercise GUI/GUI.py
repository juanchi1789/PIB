from tkinter import *
import tkinter
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
	# open a file chooser dialog and allow the user to select an input
	# image
	path = tkFileDialog.askopenfilename()
    # ensure a file path was selected
    if len(path) > 0:
    # load the image from disk, convert it to grayscale, and detect
    # edges in it
    image = cv2.imread(path)
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

    if panelA is None or panelB is None:
        # the first panel will store our original image
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side="left", padx=10, pady=10)
        # while the second panel will store the edge map
        panelB = Label(image=edged)
        panelB.image = edged
        panelB.pack(side="right", padx=10, pady=10)
    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image)
        panelB.configure(image=edged)
        panelA.image = image
        panelB.image = edged

# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()