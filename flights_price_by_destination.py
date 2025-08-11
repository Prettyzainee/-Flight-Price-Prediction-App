# flights_price_by_destination.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1️⃣ Load dataset
df = pd.read_csv("airlines_flights_data.csv")
print("Dataset shape:", df.shape)
print(df.head())

df.drop(columns=['index', 'Unnamed: 13', 'flight', 'Unnamed: 12'], inplace=True)
print(df.head())

df.dropna(inplace=True)
print(df.head())
# 2️⃣ Target and features
target_column = "price"  # Change if your column name is different
# We'll keep destination + other possible useful columns
features = df.drop(columns=[target_column])
y = df[target_column]

# 3️⃣ Identify categorical and numerical columns
categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 4️⃣ Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# 5️⃣ Prepare train/test data
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Transform data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 6️⃣ Build neural network model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_transformed.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')  # Price prediction
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7️⃣ Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_transformed, y_train,
    validation_data=(X_test_transformed, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 8️⃣ Save model and preprocessor
save_model(model, "flight_price_model.h5")
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ Model saved as flight_price_model.h5")
print("✅ Preprocessor saved as preprocessor.pkl")
