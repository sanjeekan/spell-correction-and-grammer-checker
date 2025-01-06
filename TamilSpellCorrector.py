import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from symspellpy import SymSpell, Verbosity
from collections import Counter
from itertools import tee
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize the spell checker
def initialize_spell_checker(dictionary_path):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    try:
        with open(dictionary_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    sym_spell.create_dictionary_entry(parts[0], int(parts[1]))
    except FileNotFoundError:
        raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")
    return sym_spell

# Build the bigram model
def generate_bigrams(tokens):
    a, b = tee(tokens)
    next(b, None)
    return Counter(zip(a, b))

# Perform combined spell checking
def spell_check_combined(text, sym_spell, bigram_model):
    if not text.strip():
        return "", [], {}

    words = text.split()
    corrected_text = []
    corrections = []
    suggestions_dict = {}

    for i, word in enumerate(words):
        suggestions = sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)
        if suggestions:
            corrected_word = suggestions[0].term
            corrected_text.append(corrected_word)

            if corrected_word != word:
                corrections.append((word, corrected_word))
                suggestions_dict[word] = [s.term for s in suggestions[:3]]
        else:
            corrected_text.append(word)

    return " ".join(corrected_text), corrections, suggestions_dict

# Calculate accuracy
def calculate_accuracy(corrected_text, ground_truth):
    corrected_words = corrected_text.split()
    ground_truth_words = ground_truth.split()
    matcher = SequenceMatcher(None, corrected_words, ground_truth_words)
    return round(matcher.ratio() * 100, 2)

# Plot accuracy graph
def plot_accuracy_graph(accuracy):
    """Improved plot for accuracy visualization."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Accuracy"], [accuracy], color="#4CAF50")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Spell Checker Accuracy", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.bar_label(ax.containers[0], fmt="%.2f%%", label_type="edge", fontsize=10, fontweight="bold")

    for widget in accuracy_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=accuracy_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# GUI Functions
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                input_text.delete(1.0, tk.END)
                input_text.insert(tk.END, file.read())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

def perform_spell_check():
    text = input_text.get(1.0, tk.END).strip()
    ground_truth = ground_truth_text.get(1.0, tk.END).strip()

    if not text:
        messagebox.showwarning("Input Error", "Please enter or load text to check.")
        return
    if not ground_truth:
        messagebox.showwarning("Input Error", "Please provide ground truth text.")
        return

    try:
        corrected_text, corrections, suggestions_dict = spell_check_combined(text, sym_spell, bigram_model)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, corrected_text)

        corrections_list.delete(0, tk.END)
        for original, corrected in corrections[:5]:
            suggestions = ", ".join(suggestions_dict.get(original, []))
            corrections_list.insert(tk.END, f"{original} -> {corrected} (Suggestions: {suggestions})")

        accuracy = calculate_accuracy(corrected_text, ground_truth)
        accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
        plot_accuracy_graph(accuracy)
    except Exception as e:
        messagebox.showerror("Error", f"Spell checking failed: {e}")

def save_corrected_text():
    corrected_text = output_text.get(1.0, tk.END).strip()
    if not corrected_text:
        messagebox.showwarning("Save Error", "No corrected text to save.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(corrected_text)
            messagebox.showinfo("Success", "Corrected text saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

def refresh_fields():
    input_text.delete(1.0, tk.END)
    ground_truth_text.delete(1.0, tk.END)
    output_text.delete(1.0, tk.END)
    corrections_list.delete(0, tk.END)
    accuracy_label.config(text="Accuracy: N/A")
    for widget in accuracy_frame.winfo_children():
        widget.destroy()

# Main Application
if __name__ == "__main__":
    try:
        dictionary_path = r"C:\Users\STRANGERS\Desktop\Semester 7 - 2024\EC9640-Artificial Intelligence\Project\p1\tamilWords_formatted .txt"
        sym_spell = initialize_spell_checker(dictionary_path)

        corpus = """
        இலங்கையின் வரலாறு மிகப்பெரியது. 
        அதில் உள்ள பல்வேறு பண்பாட்டு வளங்கள் உலகளாவிய முறையில் புகழ்பெற்றவை.
        இது பல மொழிகளையும், மதங்களையும், பாரம்பரியங்களையும் உள்ளடக்கியது.
        """
        tokens = corpus.split()
        bigram_model = generate_bigrams(tokens)
    except Exception as e:
        print(f"Failed to initialize resources: {e}")
        exit(1)

    root = tk.Tk()
    root.title("Tamil Spell Checker")
    root.geometry("800x600")

    style = ttk.Style()
    style.configure("TFrame", padding=10)
    style.configure("TLabel", font=("Arial", 12), padding=5)
    style.configure("TButton", font=("Arial", 10))

    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    input_frame = ttk.LabelFrame(main_frame, text="Input Text")
    input_frame.pack(fill="x", padx=10, pady=5)
    input_text = tk.Text(input_frame, height=5, wrap=tk.WORD)
    input_text.pack(fill="x", padx=5, pady=5)

    ground_truth_frame = ttk.LabelFrame(main_frame, text="Ground Truth Text")
    ground_truth_frame.pack(fill="x", padx=10, pady=5)
    ground_truth_text = tk.Text(ground_truth_frame, height=5, wrap=tk.WORD)
    ground_truth_text.pack(fill="x", padx=5, pady=5)

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill="x", padx=10, pady=5)
    ttk.Button(button_frame, text="Load Text", command=browse_file).pack(side="left", padx=5)
    ttk.Button(button_frame, text="Check Spelling", command=perform_spell_check).pack(side="left", padx=5)
    ttk.Button(button_frame, text="Save Corrected Text", command=save_corrected_text).pack(side="left", padx=5)
    ttk.Button(button_frame, text="Refresh", command=refresh_fields).pack(side="left", padx=5)

    output_frame = ttk.LabelFrame(main_frame, text="Corrected Text")
    output_frame.pack(fill="x", padx=10, pady=5)
    output_text = tk.Text(output_frame, height=5, wrap=tk.WORD)
    output_text.pack(fill="x", padx=5, pady=5)

    corrections_frame = ttk.LabelFrame(main_frame, text="Corrections (Top 5)")
    corrections_frame.pack(fill="x", padx=10, pady=5)
    corrections_list = tk.Listbox(corrections_frame, height=5)
    corrections_list.pack(fill="x", padx=5, pady=5)

    accuracy_label = ttk.Label(main_frame, text="Accuracy: N/A", font=("Arial", 14, "bold"))
    accuracy_label.pack(pady=10)

    accuracy_frame = ttk.Frame(main_frame)
    accuracy_frame.pack(pady=5)

    root.mainloop()
