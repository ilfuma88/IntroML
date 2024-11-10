import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

# # Load the dataset
# data = pd.read_csv('raw data.csv')

# %% Load dataset
filename = "raw data.csv"
df = pd.read_csv(filename, sep=";")
# Remove columns and rows with all NaN values
df_cleaned = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

print(df_cleaned.head())
# %% Encode player names as integer labels
# Encode player names to numerical values for machine learning
players = df_cleaned.iloc[:, 0]  # Assuming first column contains player names
player_codes = players.astype('category').cat.codes
K = player_codes.max() + 1  # Number of unique players

# Create one-hot encoding for players and append to dataset
player_encoding = np.zeros((player_codes.size, K))
player_encoding[np.arange(player_codes.size), player_codes] = 1

# Combine one-hot encoded players with the rest of the dataset (without player names)
X_raw = np.hstack((player_encoding, df_cleaned.iloc[:, 1:].values))

# Separate numeric features for further processing
features = df_cleaned.columns[1:]  # Skipping player names column
df_numeric = df_cleaned[features]
AllAttributeNames = players.unique().tolist() + features.tolist()



# # Separate features and target
# X = data.iloc[:, 1:].values  # Features (statistics of matches)
# y = data.iloc[:, 0].values   # Target (player names)

# # Encode target labels as integers for compatibility with sklearn models
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# %% STANDARDIZE DATA ULTRA IMPORTANTE
N, M = X_raw.shape  # Set dimensions of the dataset
# Standardize data prima di tutto

X_tosstandard = X_raw[:, 7:]
X_stand_mean = X_tosstandard - np.ones((N, 1)) * X_tosstandard.mean(axis=0) #########solo media
X_noplayers_standard = X_stand_mean * (1 / np.std(X_stand_mean, 0)) #########solo deviazione standard
X_combined_standard = np.hstack((X_raw[:, :7], X_noplayers_standard))
### qua è standardizzato anche plus minus
# print(X_combined_standard)
# %% Finalize Data Matrix for Analysis

# X = X_combined_standard[:, final_selected_features]
X = X_combined_standard[:, :-1]
y = X_combined_standard[:, -1]

N, M = X.shape  # Set dimensions of the final dataset

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + AllAttributeNames
M = M + 1

# %%


# Define models
baseline_model = DummyClassifier(strategy='most_frequent')

logistic_model = LogisticRegression(
    multi_class='multinomial', 
    solver='lbfgs', 
    max_iter=1000, 
    random_state=42
)

ann_model = MLPClassifier(
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
