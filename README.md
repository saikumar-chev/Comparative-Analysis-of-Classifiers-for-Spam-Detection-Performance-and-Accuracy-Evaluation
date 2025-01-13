# Comparative Analysis of Classifiers for Spam Detection: Performance and Accuracy Evaluation

## Overview
This project evaluates the performance of multiple machine learning classifiers for spam detection. It utilizes a GUI-based system built with Python's Tkinter framework for dataset loading, preprocessing, training, and visualization of results.

## Features
- **Login System**: Secure access to the application.
- **Dataset Loading**: Import CSV files with `Category` and `Message` columns.
- **Text Preprocessing**: Tokenization, stopword removal, and feature extraction using `CountVectorizer`.
- **Classifier Selection**: Train multiple classifiers including:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
- **Performance Metrics**:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC Curve and AUC
- **Visualization**: Compare results through bar charts, ROC curves, and heatmaps.

## Prerequisites
Make sure you have the following installed:
- Python 3.7+
- Required Python libraries:
  ```bash
  pip install pandas scikit-learn matplotlib seaborn nltk pillow
  ```

## How to Run
1. Clone the repository:
   ```bash
   git clone <[repository-url](https://github.com/saikumar-chev/Comparative-Analysis-of-Classifiers-for-Spam-Detection-Performance-and-Accuracy-Evaluation/tree/03597ef3de4cb5e8d8747e657ab9281580f591c0/Project)>
   cd <project>
   ```
2. Run the `temp.py` script:
   ```bash
   python temp.py
   ```

3. Login using the following credentials:
   - Username: `lg3`
   - Password: `svs`

4. Load a dataset (CSV format with `Category` and `Message` columns).
5. Select the classifiers to train and evaluate.
6. View and analyze the results.

## Dataset Requirements
The dataset should contain:
- **Category**: Labels as `ham` (0) or `spam` (1).
- **Message**: Text data to classify.

Example:
| Category | Message             |
|----------|---------------------|
| ham      | Hello, how are you? |
| spam     | You won $1000!      |

## File Structure
```
.
├── main.py                 # Main application script
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
├── assets/
│   └── project.png         # Login background image
└── datasets/
    └── sample.csv          # Example dataset
```

## Performance Visualization
- **Accuracy Comparison**: A bar chart showing the accuracy of each classifier.
- **Confusion Matrix**: Heatmaps for each classifier's predictions.
- **ROC Curve**: AUC values for classifier comparison.

## Screenshots
### Login Page:
![image](https://github.com/user-attachments/assets/0b9f7625-864a-4fd7-b1a8-0c1347037d21)

### Dataset Viewer:
![image](https://github.com/user-attachments/assets/637104cd-7d69-4208-b929-6a6e3b50abca)

### Main Window:
![image](https://github.com/user-attachments/assets/566af064-178f-4a9b-a6a7-4438211fa6ff)

### Results:
**Accuracy & ROC Curve Comparision**
![image](https://github.com/user-attachments/assets/26bd2344-7e8c-4eef-9ff0-755da520ebb0)
**Confusion Matrix**
![image](https://github.com/user-attachments/assets/16fef664-ae01-4b6f-ab40-b837c58754f8) ![image](https://github.com/user-attachments/assets/c8b97eab-708f-459f-b7b1-dfcc4b0eb129) ![image](https://github.com/user-attachments/assets/f6bd96a4-3f9c-41b5-922b-f953d206d351)

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `Tkinter`: GUI development
  - `Pandas`: Data manipulation
  - `Scikit-learn`: Machine learning
  - `Matplotlib` & `Seaborn`: Data visualization
  - `NLTK`: Text preprocessing

## Contributors
- **Your Name** - [Your GitHub Profile](https://github.com/yourprofile)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
