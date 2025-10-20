import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Ejemplo: dataset de carros
data = {
    "Brand": ["Toyota", "BMW", "Ford", "Daewoo", np.nan],
    "FuelType": ["Petrol", "Diesel", "CNG", "Diesel", "Petrol"],
    "Mileage": [15000, 30000, 900000, 25000, 27000],  # un valor atípico (900000)
    "EngineSize": [1.8, 3.0, np.nan, 2.0, 1.6],
    "Transmission": ["Manual", "Automatic", "Manual", "Automatic", "Manual"],
    "Price": [20000, 35000, 5000, 15000, 18000]
}

df = pd.DataFrame(data)
print("Dataset original:")
print(df)

# -------------------------------
# 1. Limpieza de datos
# -------------------------------

# Imputar valores faltantes en EngineSize con la mediana
df["EngineSize"].fillna(df["EngineSize"].median(), inplace=True)

# Eliminar filas con NA en Brand
df.dropna(subset=["Brand"], inplace=True)

# Tratar valores atípicos en Mileage (ejemplo: eliminar > 200000 km)
df = df[df["Mileage"] < 200000]

# Agrupar categorías poco frecuentes en Brand
df["Brand"] = df["Brand"].replace({"Daewoo": "Other"})

print("\nDespués de limpieza:")
print(df)

# -------------------------------
# 2. Codificación de variables categóricas
# -------------------------------

# One Hot Encoding para FuelType
df = pd.get_dummies(df, columns=["FuelType"], drop_first=True)

# Label Encoding para Transmission
le = LabelEncoder()
df["Transmission"] = le.fit_transform(df["Transmission"])

print("\nDespués de codificación:")
print(df)

# -------------------------------
# 3. Escalamiento / Normalización
# -------------------------------
scaler = StandardScaler()
df[["Mileage", "EngineSize"]] = scaler.fit_transform(df[["Mileage", "EngineSize"]])

print("\nDespués de escalamiento:")
print(df)

# -------------------------------
# 4. Selección de variables relevantes
# -------------------------------
X = df.drop("Price", axis=1)
y = df["Price"]

# Seleccionar las 3 variables más importantes según ANOVA F-test
selector = SelectKBest(score_func=f_regression, k=3)
X_new = selector.fit_transform(X, y)

print("\nVariables seleccionadas más relevantes:")
print(X.columns[selector.get_support()])
