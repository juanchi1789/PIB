from tkinter import *

root = Tk()
# Definir un texto
myLabel = Label(root, text="Hello world")
myLabel.pack()

texto = Entry(root, width = 50)# Casilla para ingresar texto
texto.pack()

# Definir un boton
# Podemos cambiarle el tamaño
# Para que tenga alguna actividad tenes que definir una funcion!
def onClick():
    label = Label(root, text = "Hola! " + texto.get())
    label.pack()

# El parametro command define la funcion que tiene asignada
myButton = Button(root, text="Enter your name", padx=50, pady=20, command=onClick)
myButton.pack()
root.mainloop()
