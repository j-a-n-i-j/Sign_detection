import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import Image, ImageTk
import numpy as np

# Load the saved model
model = load_model('asl_model_with_errors.h5')

# Dictionary to map class indices to letters
class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
                8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
                15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 
                22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 
                28: 'space'}

# Function to load and predict the image
def predict_image(filepath):
    img = Image.open(filepath).resize((64, 64))  # Resize image to match model input shape
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class]

# Function to handle file upload and prediction
def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        result_label.config(text="Predicting...")
        prediction = predict_image(file_path)
        result_label.config(text=f"Predicted ASL Sign: {prediction}")
        img = Image.open(file_path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

# Setting up the GUI
root = tk.Tk()
root.title("ASL Sign Language Prediction")

upload_button = tk.Button(root, text="Upload Image", command=upload_file)
upload_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

img_label = tk.Label(root)
img_label.pack()

root.mainloop()

