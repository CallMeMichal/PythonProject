import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandastable import Table
import matplotlib.pyplot as plt
import pickle

model = None
df = None

def train_model():
    global df
    try:
        # Krok 1: Pobranie danych
        data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                        'thal', 'target']
        df = pd.read_csv(data_url, header=None, names=data_columns, na_values='?')

        # Krok 2: Obsługa brakujących danych
        df = df.dropna()

        # Krok 3: Podział danych na zbiór treningowy i testowy
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Krok 4: Trenowanie modelu
        global model
        model = LogisticRegression(solver='saga', max_iter=10000)
        model.fit(X_train, y_train)

        # Krok 5: Testowanie modelu
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_label.config(text=f"Dokładność modelu: {accuracy}", fg="green")

    except ValueError:
        accuracy_label.config(text="", fg="red")

def test_model():
    try:
        global model, df
        if model is None:
            raise ValueError("Najpierw trenuj model")

        global df
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred_train = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)

        y_pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        accuracy_label.config(text=f"Dokładność treningowa: {train_accuracy}, Dokładność testowa: {test_accuracy}", fg="green")

    except ValueError as e:
        accuracy_label.config(text=str(e), fg="red")

def predict():
    try:
        global model
        if model is None:
            raise ValueError("Najpierw trenuj model")

        # Krok 6: Predykcja na nowych danych
        new_data = {'age': float(age_entry.get()), 'sex': float(sex_entry.get()), 'cp': float(cp_entry.get()),
                    'trestbps': float(trestbps_entry.get()), 'chol': float(chol_entry.get()), 'fbs': float(fbs_entry.get()),
                    'restecg': float(restecg_entry.get()), 'thalach': float(thalach_entry.get()),
                    'exang': float(exang_entry.get()), 'oldpeak': float(oldpeak_entry.get()),
                    'slope': float(slope_entry.get()), 'ca': float(ca_entry.get()), 'thal': float(thal_entry.get())}
        new_data_df = pd.DataFrame([new_data])
        prediction = model.predict(new_data_df)
        prediction_label.config(text=f"Wynik predykcji: {prediction}", fg="green")

    except ValueError as e:
        prediction_label.config(text=str(e), fg="red")

def browse_data():
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                    'thal', 'target']
    df = pd.read_csv(data_url, header=None, names=data_columns)
    df = df.dropna()


    browse_window = tk.Toplevel(window)
    browse_window.title("Przeglądanie danych")

    frame = tk.Frame(browse_window)
    frame.pack(fill="both", expand=True)

    table = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
    table.show()

def save_model():
    try:
        global model
        if model is None:
            raise ValueError("Najpierw trenuj model")

        # Zapisz model do pliku
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)

        accuracy_label.config(text="Model został zapisany", fg="green")

    except ValueError as e:
        accuracy_label.config(text=str(e), fg="red")

def load_model():
    try:
        global model

        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)

        accuracy_label.config(text="Model został wczytany", fg="green")

    except FileNotFoundError:
        accuracy_label.config(text="Nie znaleziono pliku z modelem", fg="red")
    except Exception as e:
        accuracy_label.config(text=str(e), fg="red")

def visualize_data():
    global df
    if df is None:
        return

    # wie cechy do wizualizacji
    feature1 = 'age'
    feature2 = 'chol'

    # tworzenie wykresu
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature1], df[feature2], c=df['target'])
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Wizualizacja danych: {feature1} vs {feature2}')
    plt.colorbar(label='target')
    plt.show()

window = tk.Tk()
window.title("Projekt Michal Tulej - Klasyfikacja Choroby Serca")

age_label = tk.Label(window, text="Wiek:")
age_label.pack()
age_entry = tk.Entry(window)
age_entry.pack()

sex_label = tk.Label(window, text="Płeć(1 = mężczyzna; 0 = kobieta):")
sex_label.pack()
sex_entry = tk.Entry(window)
sex_entry.pack()

cp_label = tk.Label(window, text="Ból w klatce piersiowej:")
cp_label.pack()
cp_entry = tk.Entry(window)
cp_entry.pack()

trestbps_label = tk.Label(window, text="Ciśnienie krwi:")
trestbps_label.pack()
trestbps_entry = tk.Entry(window)
trestbps_entry.pack()

chol_label = tk.Label(window, text="Cholesterol:")
chol_label.pack()
chol_entry = tk.Entry(window)
chol_entry.pack()

fbs_label = tk.Label(window, text="Poziom cukru we krwi:")
fbs_label.pack()
fbs_entry = tk.Entry(window)
fbs_entry.pack()

restecg_label = tk.Label(window, text="Elektrokardiograficzne wyniki spoczynkowe:")
restecg_label.pack()
restecg_entry = tk.Entry(window)
restecg_entry.pack()

thalach_label = tk.Label(window, text="Maksymalne osiągnięte tętno:")
thalach_label.pack()
thalach_entry = tk.Entry(window)
thalach_entry.pack()

exang_label = tk.Label(window, text="Dławica wysiłkowa:")
exang_label.pack()
exang_entry = tk.Entry(window)
exang_entry.pack()

oldpeak_label = tk.Label(window, text="Depresja ST indukowana wysiłkiem:")
oldpeak_label.pack()
oldpeak_entry = tk.Entry(window)
oldpeak_entry.pack()

slope_label = tk.Label(window, text="Nachylenie odcinka ST:")
slope_label.pack()
slope_entry = tk.Entry(window)
slope_entry.pack()

ca_label = tk.Label(window, text="Liczba głównych naczyń:")
ca_label.pack()
ca_entry = tk.Entry(window)
ca_entry.pack()

thal_label = tk.Label(window, text="Tętno:")
thal_label.pack()
thal_entry = tk.Entry(window)
thal_entry.pack()

train_button = tk.Button(window, text="Trenuj Model", command=train_model)
train_button.pack(side=tk.LEFT, padx=5)

test_button = tk.Button(window, text="Testuj Model", command=test_model)
test_button.pack(side=tk.LEFT, padx=5)

predict_button = tk.Button(window, text="Predykcja", command=predict)
predict_button.pack(side=tk.LEFT, padx=5)

browse_button = tk.Button(window, text="Przeglądaj Dane", command=browse_data)
browse_button.pack(side=tk.LEFT, padx=5)

save_model_button = tk.Button(window, text="Zapisz Model", command=save_model)
save_model_button.pack(side=tk.LEFT, padx=5)

load_model_button = tk.Button(window, text="Wczytaj Model", command=load_model)
load_model_button.pack(side=tk.LEFT, padx=5)

visualize_button = tk.Button(window, text="Wizualizuj Dane", command=visualize_data)
visualize_button.pack(side=tk.LEFT, padx=5)

accuracy_label = tk.Label(window, text="")
accuracy_label.pack()

prediction_label = tk.Label(window, text="")
prediction_label.pack()

window.mainloop()
