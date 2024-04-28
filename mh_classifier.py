from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def clean_dataset():
    df = pd.read_csv('Student Mental health.csv')

    # Clean column names
    df.columns = ['Gender', 'Age', 'Course', 'Year_of_Study', 'CGPA', 'Marital_Status', 'Depression', 'Anxiety',
                  'Panic_Attack', 'Treatment']

    # Remove null values
    df = df.dropna()

    # Preprocessing and cleaning data
    df['Year_of_Study'] = df['Year_of_Study'].astype(str).str.extract('(\d+)').astype(int)

    le = LabelEncoder()
    cat_cols = ['Gender', 'Marital_Status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
    df['Year_of_Study'] = df['Year_of_Study'].astype(str).str.extract('(\d+)').astype(int)
    le_course = LabelEncoder()

    df['Course'] = le_course.fit_transform(df['Course'])
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    def calculate_cgpa(x):
        if isinstance(x, str):
            first, second = x.split('-')
            return (float(first.strip()) + float(second.strip())) / 2
        else:
            return x

    df['CGPA'] = df['CGPA'].apply(calculate_cgpa)

    # Define X and y
    X = df[['Marital_Status', 'Course', 'CGPA', 'Anxiety', 'Panic_Attack']]
    y = df['Depression']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_classifier(model_name, train_x, train_y):
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Naive Bayes": GaussianNB(),
        "K-nearest Neighbour": KNeighborsClassifier()
    }
    model = models[model_name]
    model.fit(train_x, train_y)
    return model


def test_classifier(model_name, test_x, test_y):
    model = train_classifier(model_name, train_x, train_y)

    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    report = classification_report(test_y, predictions)

    print(model_name)
    print(f"Accuracy: {accuracy}")
    print(report)

    return predictions, accuracy, report


def predict_with_metrics(classifier_name, test_x):
    predictions, accuracy, report = test_classifier(classifier_name, test_x, test_y)
    return predictions, accuracy, report


train_x, test_x, train_y, test_y = clean_dataset()

classifier_name = input("Enter the name of the classifier (e.g., 'Naive Bayes', 'Logistic Regression'): ")
predicted_values, accuracy, classification_report = predict_with_metrics(classifier_name, test_x)
