import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the dataset
dataset_path = r"C:\Users\STRANGERS\Desktop\Semester 7 - 2024\EC9640-Artificial Intelligence\Project\p1\Tamil_Grammar_Correction_Dataset.xlsx"
data = pd.read_excel(dataset_path)

# Data Preprocessing
# Encode the error type labels
le = LabelEncoder()
data['Error Type'] = le.fit_transform(data['Type of Error'])

# Split data into training and testing sets
X = data['Sentence']  # Input sentences
y = data['Corrected Sentence']  # Corrected sentences (ground truth)
error_types = data['Error Type']  # Type of error (for classification)

# Train-test split
X_train, X_test, y_train, y_test, error_types_train, error_types_test = train_test_split(X, y, error_types, test_size=0.2, random_state=42)

# TF-IDF Vectorizer for text feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model for Error Type Prediction
error_type_model = LogisticRegression()
error_type_model.fit(X_train_tfidf, error_types_train)

# Predict the error type on the test set
error_type_predictions = error_type_model.predict(X_test_tfidf)

# Calculate accuracy of error type classification
error_type_accuracy = accuracy_score(error_types_test, error_type_predictions)
print(f"Error Type Classification Accuracy: {error_type_accuracy * 100:.2f}%")

# Load a pre-trained transformer model for sequence-to-sequence (e.g., T5 or BART)
model_name = "t5-base"  # Changed to t5-base for better performance
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to predict the corrected sentence using the model
def correct_sentence_with_model(sentence):
    # Simplified and clearer prompt
    prompt = f"Correct the grammar of this sentence: {sentence}"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Print tokenized input for debugging
    print(f"Tokenized Input: {inputs}")

    # Generate the output from the model
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    
    # Decode the output from the model
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the model output for debugging
    print(f"Generated Output: {corrected_sentence}")

    # If output is not meaningful, return a default message
    if not corrected_sentence or corrected_sentence == sentence:
        corrected_sentence = "No correction made or error in model output."
    
    return corrected_sentence

# Function to check error type and correct sentence
def check_grammar_gui(sentence):
    # Predict error type
    sentence_tfidf = vectorizer.transform([sentence])
    error_type_prediction = error_type_model.predict(sentence_tfidf)[0]
    error_type = le.inverse_transform([error_type_prediction])[0]
    
    # Correct the sentence
    corrected_sentence = correct_sentence_with_model(sentence)
    
    return error_type, corrected_sentence

# GUI Application
def create_gui():
    root = tk.Tk()
    root.title("Tamil Grammar Checker")

    # Input area
    tk.Label(root, text="Enter Tamil Sentence:", font=("Arial", 14)).pack(pady=10)
    input_text = tk.Text(root, height=5, width=60, font=("Arial", 12))
    input_text.pack(pady=10)

    # Output area
    output_text = tk.StringVar()
    output_label = tk.Label(root, textvariable=output_text, font=("Arial", 12), fg="blue", wraplength=500, justify="left")
    output_label.pack(pady=10)

    # Accuracy display label
    accuracy_label = tk.Label(root, text="Model Accuracy: 0.00%", font=("Arial", 12), fg="green")
    accuracy_label.pack(pady=10)

    def check_sentence():
        user_sentence = input_text.get("1.0", tk.END).strip()
        if not user_sentence:
            messagebox.showerror("Error", "Please enter a sentence.")
            return
        
        # Predict error type and corrected sentence
        error_type, corrected_sentence = check_grammar_gui(user_sentence)
        
        # Display the corrected sentence and error type
        output_text.set(f"Error Type: {error_type}\nCorrected Sentence: {corrected_sentence}")
        
        # Calculate accuracy dynamically for this specific input sentence
        predicted_corrected_sentence = corrected_sentence.strip()
        expected_corrected_sentence = y_test.iloc[X_test[X_test == user_sentence].index[0]]  # Get the corresponding ground truth
        accuracy = 100 if predicted_corrected_sentence == expected_corrected_sentence else 0
        
        accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}%")

    def refresh():
        input_text.delete("1.0", tk.END)
        output_text.set("")  # Reset the output area to empty string
        accuracy_label.config(text="Model Accuracy: 0.00%")  # Reset the accuracy to default

    def exit_program():
        root.quit()

    # Button for checking the grammar
    tk.Button(root, text="Check Grammar", font=("Arial", 12), command=check_sentence).pack(pady=10)
    
    # Button for refreshing the input
    tk.Button(root, text="Refresh", font=("Arial", 12), command=refresh).pack(pady=10)
    
    # Button for exiting the program
    tk.Button(root, text="Exit", font=("Arial", 12), command=exit_program).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
