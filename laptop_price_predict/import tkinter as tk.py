import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
from laptop_price_prediction_final import predict_price  # Import the prediction function

# Loading the trained model and the X_train data
model = joblib.load('laptop_price_model.pkl')
X_train = joblib.load('X_train.pkl')  # Load the X_train data

# Defining options for categorical variables (these should match your dataset's categories)
company_options = ['HP', 'Dell', 'Apple', 'Asus', 'Chuwi', 'Fujitsu', 'Google', 'Lenovo', 'MSI', 'Razer']
opsys_options = ['Windows', 'macOS', 'Linux']
processor_options = ['i3', 'i5', 'i7', 'i9']
gpu_options = ['Intel HD Graphics 520', 'NVIDIA GeForce GTX 1050', 'AMD Radeon RX 580', 'Intel UHD Graphics 630']

# Creating the main window
window = tk.Tk()
window.title("Laptop Price Prediction")
window.geometry("400x600")  # Set a fixed size for the window

# Framing for input fields
input_frame = tk.Frame(window)
input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Function to create labeled entry widgets
def create_label_entry(frame, label_text, entry_widget, row, col):
    label = tk.Label(frame, text=label_text)
    label.grid(row=row, column=col, sticky="w", padx=5, pady=5)
    entry_widget.grid(row=row, column=col+1, sticky="ew", padx=5, pady=5)

# Inches (Numeric input)
inches_entry = tk.Entry(input_frame)
create_label_entry(input_frame, "Inches", inches_entry, 0, 0)

# Ram (Numeric input)
ram_entry = tk.Entry(input_frame)
create_label_entry(input_frame, "Ram (GB)", ram_entry, 1, 0)

# Storage (Dropdown)
storage_entry = ttk.Combobox(input_frame, values=['500GB HDD', '1TB HDD', '256GB SSD', '512GB SSD', '1TB SSD'])
create_label_entry(input_frame, "Storage", storage_entry, 2, 0)

# Company (Dropdown)
company_entry = ttk.Combobox(input_frame, values=company_options)
create_label_entry(input_frame, "Company", company_entry, 3, 0)

# OpSys (Dropdown)
opsys_entry = ttk.Combobox(input_frame, values=opsys_options)
create_label_entry(input_frame, "Operating System", opsys_entry, 4, 0)

# Processor (Dropdown)
processor_entry = ttk.Combobox(input_frame, values=processor_options)
create_label_entry(input_frame, "Processor", processor_entry, 5, 0)

# GPU (Dropdown)
gpu_entry = ttk.Combobox(input_frame, values=gpu_options)
create_label_entry(input_frame, "GPU", gpu_entry, 6, 0)

# Frame for result display
result_frame = tk.Frame(window)
result_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Label to display the predicted price
result_label = tk.Label(result_frame, text="Predicted Price: $0.00", font=("Helvetica", 14))
result_label.grid(row=0, column=0, pady=20)

# Function to trigger prediction
def on_predict():
    try:
        # Get the input data from the user
        input_data = {
            'Inches': float(inches_entry.get()) if inches_entry.get() else 0,
            'Ram': float(ram_entry.get()) if ram_entry.get() else 0,
            '500GB HDD': 1 if storage_entry.get() == '500GB HDD' else 0,
            '1TB HDD': 1 if storage_entry.get() == '1TB HDD' else 0,
            '256GB SSD': 1 if storage_entry.get() == '256GB SSD' else 0,
            '512GB SSD': 1 if storage_entry.get() == '512GB SSD' else 0,
            '1TB SSD': 1 if storage_entry.get() == '1TB SSD' else 0,
            'Company_HP': 1 if company_entry.get() == 'HP' else 0,
            'Company_Dell': 1 if company_entry.get() == 'Dell' else 0,
            'Company_Apple': 1 if company_entry.get() == 'Apple' else 0,
            'Company_Asus': 1 if company_entry.get() == 'Asus' else 0,
            'Company_Chuwi': 1 if company_entry.get() == 'Chuwi' else 0,
            'Company_Fujitsu': 1 if company_entry.get() == 'Fujitsu' else 0,
            'Company_Google': 1 if company_entry.get() == 'Google' else 0,
            'Company_Lenovo': 1 if company_entry.get() == 'Lenovo' else 0,
            'Company_MSI': 1 if company_entry.get() == 'MSI' else 0,
            'Company_Razer': 1 if company_entry.get() == 'Razer' else 0,
            'OpSys_Windows': 1 if opsys_entry.get() == 'Windows' else 0,
            'OpSys_macOS': 1 if opsys_entry.get() == 'macOS' else 0,
            'OpSys_Linux': 1 if opsys_entry.get() == 'Linux' else 0,
            'Processor_i3': 1 if processor_entry.get() == 'i3' else 0,
            'Processor_i5': 1 if processor_entry.get() == 'i5' else 0,
            'Processor_i7': 1 if processor_entry.get() == 'i7' else 0,
            'Processor_i9': 1 if processor_entry.get() == 'i9' else 0,
            'Intel HD Graphics 520': 1 if gpu_entry.get() == 'Intel HD Graphics 520' else 0,
            'NVIDIA GeForce GTX 1050': 1 if gpu_entry.get() == 'NVIDIA GeForce GTX 1050' else 0,
            'AMD Radeon RX 580': 1 if gpu_entry.get() == 'AMD Radeon RX 580' else 0,
            'Intel UHD Graphics 630': 1 if gpu_entry.get() == 'Intel UHD Graphics 630' else 0,
        }

        # Make prediction using the imported function
        predicted_price_log = predict_price(input_data, model, X_train)
        predicted_price = np.exp(predicted_price_log)  # Reverse log transformation

        # Update the result label with the predicted price
        result_label.config(text=f"Predicted Price: ${predicted_price:.2f}")
        
    except ValueError:
        result_label.config(text="Error: Invalid input. Please try again.")

# Function to erase all input fields
def erase_inputs():
    inches_entry.delete(0, tk.END)
    ram_entry.delete(0, tk.END)
    storage_entry.set('')
    company_entry.set('')
    opsys_entry.set('')
    processor_entry.set('')
    gpu_entry.set('')
    result_label.config(text="Predicted Price: $0.00")

# Add buttons to trigger prediction and erase input
predict_button = tk.Button(window, text="Predict Price", command=on_predict, font=("Helvetica", 12), bg="#4CAF50", fg="white")
predict_button.grid(row=2, column=0, pady=20)

erase_button = tk.Button(window, text="Erase Inputs", command=erase_inputs, font=("Helvetica", 12), bg="#FF6347", fg="white")
erase_button.grid(row=3, column=0, pady=10)

# Configure grid weights to make the layout responsive
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)

# Run the GUI
window.mainloop()
