import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from PIL import Image, ImageTk
from tkinter import *
from PIL import Image as PILImage, ImageTk


nltk.download('punkt')
nltk.download('stopwords')

# Global variables
data = None
vectorizer = None
X_train, X_test, y_train, y_test = None, None, None, None

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

# Function to load dataset
def load_dataset():
    global data
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        data = pd.read_csv(file_path)

        if 'Category' not in data.columns or 'Message' not in data.columns:
            messagebox.showerror("Error", "Dataset must contain 'Category' and 'Message' columns!")
            data = None
            return

        messagebox.showinfo("Success", "Dataset loaded successfully!")

        # Display dataset in Treeview
        display_dataset()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

# Function to display dataset in the Treeview
def display_dataset():
    if data is None:
        return

    # Clear the existing data in the Treeview
    for row in tree.get_children():
        tree.delete(row)

    # Define the columns in the Treeview
    tree["columns"] = list(data.columns)
    tree["show"] = "headings"  # Hide the default tree column

    for col in data.columns:
        tree.column(col, width=150, anchor="w")
        tree.heading(col, text=col)

    # Insert rows into the Treeview
    for index, row in data.iterrows():
        tree.insert("", "end", values=list(row))

def train_classifier():
    global data, vectorizer, X_train, X_test, y_train, y_test
    if data is None:
        messagebox.showerror("Error", "Dataset not loaded!")
        return

    try:
        # Clear previous results
        vectorizer = None
        X_train, X_test, y_train, y_test = None, None, None, None

        # Preprocess data
        data['Message'] = data['Message'].fillna("").astype(str)

        # Drop rows where 'Category' or 'Message' is missing or empty
        data = data[data['Category'].notna()]
        data = data[data['Message'].notna() & (data['Message'] != '')]

        # Ensure 'Category' is mapped to valid labels (ham: 0, spam: 1)
        valid_labels = {'ham': 0, 'spam': 1}
        data['Category'] = data['Category'].map(valid_labels)
        data = data[data['Category'].notna()]  # Remove invalid categories

        # Apply text preprocessing
        data['processed_text'] = data['Message'].apply(preprocess_text)

        # Drop rows with empty processed text (i.e., empty after preprocessing)
        data = data[data['processed_text'].notna()]

        # Ensure that there are no missing values in the features or target
        if data['processed_text'].isnull().any() or data['Category'].isnull().any():
            messagebox.showerror("Error", "Missing values found in the data!")
            return

        # Convert text to features using CountVectorizer
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['processed_text'])

        y = data['Category']

        # Ensure X and y have the same length before splitting
        if X.shape[0] != y.shape[0]:
            messagebox.showerror("Error", "Mismatch in number of samples between features and target!")
            return

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Selected classifiers
        selected_classifiers = []
        if var_nb.get():
            selected_classifiers.append(("Naive Bayes", MultinomialNB()))
        if var_svm.get():
            selected_classifiers.append(("SVM", SVC(kernel='linear', probability=True)))
        if var_lr.get():
            selected_classifiers.append(("Logistic Regression", LogisticRegression(max_iter=1000)))
        if var_rf.get():
            selected_classifiers.append(("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)))
        if var_knn.get():
            selected_classifiers.append(("KNN", KNeighborsClassifier(n_neighbors=5)))

        if not selected_classifiers:
            messagebox.showerror("Error", "Please select at least one classifier!")
            return

        # Training selected classifiers
        results = {}
        for name, clf in selected_classifiers:
            print(f"Training {name}...")
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)
            fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)

            results[name] = {
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "classification_report": report,
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": roc_auc
            }

        # Plotting results
        plot_results(results)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to train classifiers: {str(e)}")
        print(f"Error during training: {str(e)}")  # Debugging log


# Function to plot results
def plot_results(results):
    # Plotting the comparison of accuracies
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot for accuracy comparison
    ax[0].bar(results.keys(), [r["accuracy"] for r in results.values()], color='skyblue')
    ax[0].set_title('Accuracy Comparison')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0, 1)

    # ROC Curve comparison
    for name, result in results.items():
        ax[1].plot(result['fpr'], result['tpr'], label=f"{name} (AUC = {result['roc_auc']:.2f})")

    ax[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax[1].set_title('ROC Curve Comparison')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # Display Confusion Matrices
    for name, result in results.items():
        cm = result['confusion_matrix']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


# Login Screen
def login():
    def verify_credentials():
        username = entry_username.get()
        password = entry_password.get()

        # Sample login credentials check (replace with your own logic)
        if username == "lg3" and password == "svs":
            login_window.destroy()  # Close the login window
            show_main_window()  # Show the main window
        else:
            messagebox.showerror("Login Failed", "Invalid credentials!")

    # Create the login window
    login_window = tk.Tk()
    login_window.title("Login")

    # Make the login window full screen
    login_window.state("zoomed")

    # Add background image
    def load_background_image():
        try:
            # Load image using PIL
            image_path = r"C:\Users\cheve\PycharmProjects\Data Mining\Project\project.png"
            img = PILImage.open(image_path)  # Corrected usage of PILImage.open()
            img = img.resize(
                (login_window.winfo_screenwidth(), login_window.winfo_screenheight()))  # Resize image to fit screen
            background_image = ImageTk.PhotoImage(img)

            # Create a label to hold the image and place it in the root window
            background_label = tk.Label(login_window, image=background_image)
            background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Cover the entire window

            # Ensure the label is behind all other widgets
            background_label.lower()

            # Keep a reference to the image to avoid garbage collection
            login_window.background_image = background_image

        except FileNotFoundError:
            messagebox.showerror("Error", "Failed to load background image.")

    # Call the background loading function
    load_background_image()

    # Add a project title at the top of the login page
    project_title = tk.Label(
        login_window,
        text="Comparative Analysis of Classifiers for Spam Detection: Performance and Accuracy Evaluation",
        font=("Helvetica", 20, "bold"),
        bg="#00008B",  # Navy blue background
        fg="#FFFFFF",  # White text
        wraplength=900,  # Wrap text for readability
        justify="center",
        padx=20,
        pady=10
    )
    project_title.pack(pady=20)

    # Add other UI elements on top of the canvas
    label_username = tk.Label(login_window, text="Username", font=("Arial", 16), bg="#ADD8E6", fg="#00008B")
    label_username.pack(pady=20)
    entry_username = tk.Entry(login_window, font=("Arial", 14))
    entry_username.pack(pady=10)

    label_password = tk.Label(login_window, text="Password", font=("Arial", 16), bg="#ADD8E6", fg="#00008B")
    label_password.pack(pady=20)
    entry_password = tk.Entry(login_window, show="*", font=("Arial", 14))
    entry_password.pack(pady=10)

    btn_login = tk.Button(login_window, text="Login", command=verify_credentials, font=("Arial", 16), bg="#5CB85C", fg="#FFFFFF")
    btn_login.pack(pady=20)

    login_window.mainloop()


# Main Window
# Main Window with colors
def show_main_window():
    global tree, var_nb, var_svm, var_lr, var_rf, var_knn
    root = tk.Tk()
    root.title("Spam Detection with Multiple Algorithms")
    root.state("zoomed")

    # Frame for controls
    frame_controls = tk.Frame(root, bg="#f0f8ff")  # Light blue background
    frame_controls.pack(side="top", fill="both", pady=20)

    btn_load = tk.Button(
        frame_controls,
        text="Load Dataset",
        command=load_dataset,
        width=25,
        bg="#87CEEB",  # Sky blue background
        fg="white",    # White text
        font=("Arial", 12, "bold"),
    )
    btn_load.pack(pady=10)

    label_select_classifiers = tk.Label(
        frame_controls,
        text="Select Classifiers to Train:",
        bg="#f0f8ff",  # Same background as the frame
        fg="#000080",  # Navy blue text
        font=("Arial", 14, "bold"),
    )
    label_select_classifiers.pack(pady=10)

    # Create Checkboxes for classifier selection (inline layout)
    var_nb = tk.BooleanVar()
    var_svm = tk.BooleanVar()
    var_lr = tk.BooleanVar()
    var_rf = tk.BooleanVar()
    var_knn = tk.BooleanVar()

    frame_checkboxes = tk.Frame(frame_controls, bg="#f0f8ff")  # Light blue background
    frame_checkboxes.pack(pady=5)

    tk.Checkbutton(
        frame_checkboxes,
        text="Naive Bayes",
        variable=var_nb,
        bg="#f0f8ff",  # Checkbox background
        fg="#000000",  # Black text
        font=("Arial", 12),
    ).pack(side="left", padx=10)
    tk.Checkbutton(
        frame_checkboxes,
        text="SVM",
        variable=var_svm,
        bg="#f0f8ff",
        fg="#000000",
        font=("Arial", 12),
    ).pack(side="left", padx=10)
    tk.Checkbutton(
        frame_checkboxes,
        text="Logistic Regression",
        variable=var_lr,
        bg="#f0f8ff",
        fg="#000000",
        font=("Arial", 12),
    ).pack(side="left", padx=10)
    tk.Checkbutton(
        frame_checkboxes,
        text="Random Forest",
        variable=var_rf,
        bg="#f0f8ff",
        fg="#000000",
        font=("Arial", 12),
    ).pack(side="left", padx=10)
    tk.Checkbutton(
        frame_checkboxes,
        text="KNN",
        variable=var_knn,
        bg="#f0f8ff",
        fg="#000000",
        font=("Arial", 12),
    ).pack(side="left", padx=10)

    btn_train = tk.Button(
        frame_controls,
        text="Train Selected Classifiers",
        command=train_classifier,
        width=25,
        bg="#4682B4",  # Steel blue background
        fg="white",    # White text
        font=("Arial", 12, "bold"),
    )
    btn_train.pack(pady=10)

    btn_exit = tk.Button(
        frame_controls,
        text="Exit",
        command=root.quit,
        width=25,
        bg="#B22222",  # Firebrick red background
        fg="white",
        font=("Arial", 12, "bold"),
    )
    btn_exit.pack(pady=10)

    frame_dataset = tk.Frame(root, bg="#f8f8ff")  # Ghost white background
    frame_dataset.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

    tree = ttk.Treeview(frame_dataset)
    tree.pack(fill="both", expand=True)

    # Styling the Treeview
    style = ttk.Style()
    style.configure("Treeview", background="#ffffff", foreground="black", rowheight=25, fieldbackground="#f8f8ff")
    style.map("Treeview", background=[("selected", "#ADD8E6")])  # Light blue for selected rows

    scrollbar = ttk.Scrollbar(frame_dataset, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()


# Start the login process
if __name__ == "__main__":
    login()  # Display the login window first
