import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Loading the pre-trained model
model = load_model("animal_classifier_model.h5")

# Initializing the GUI
top = tk.Tk()
top.geometry("800x600")
top.title("Animal Classifier")
top.configure(background="#CDCDCD")

# Initializing the labels
label_age = Label(top, background="#CDCDCD", font=("arial", 15, "bold"))
label_diet = Label(top, background="#CDCDCD", font=("arial", 15, "bold"))
label_count = Label(top, background="#CDCDCD", font=("arial", 15, "bold"))
sign_image = Label(top)

# Function to predict attributes of uploaded image
def predict_attributes(file_path):
    try:
        global label_age, label_diet, label_count
        image = Image.open(file_path)
        image = image.resize((48, 48,))  # Resize image to fit model input size
        image = np.array(image)  # Convert PIL image to numpy array
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize image

        # Predicting age, diet, and count
        pred_age, pred_diet, pred_count = model.predict(image)

        # Age prediction (binary)
        age = "Adult" if pred_age[0][0] > 0.5 else "Child"

        # Diet prediction (binary)
        diet = "Carnivore" if pred_diet[0][0] > 0.5 else "Herbivore"

        # Count prediction (2 outputs: Herbivores and Carnivores count)
        herbivores_count = int(np.round(pred_count[0][0]))
        carnivores_count = int(np.round(pred_count[0][1]))

        # Update labels with predictions
        label_age.config(foreground="#011638", text=f"Predicted Age: {age}")
        label_diet.config(foreground="#011638", text=f"Predicted Diet: {diet}")
        label_count.config(foreground="#011638", text=f"Predicted Herbivores: {herbivores_count}, Carnivores: {carnivores_count}")

    except Exception as e:
        print("Error:", e)

# Function to display uploaded image and initiate prediction
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label_age.config(text=" ")
        label_diet.config(text=" ")
        label_count.config(text=" ")
        predict_attributes(file_path)

    except Exception as e:
        print("Error:", e)

# Button to upload an image
upload_button = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload_button.config(background="#364156", foreground="white", font=("arial", 10, "bold"))
upload_button.pack(side="bottom", pady=50)

# Packing labels and image display
sign_image.pack(side="bottom", expand=True)
label_age.pack()
label_diet.pack()
label_count.pack()

# Label for heading
heading = Label(top, text="Animal Classifier", pady=20, font=("arial", 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Start GUI main loop
top.mainloop()
