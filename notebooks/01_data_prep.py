import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(input_path: str):
    # Define correct column names
    col_names = [
        "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
        "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
    ]
    
    # 1. Load dataset safely
    df = pd.read_csv(input_path, names=col_names, header=0, quotechar='"')
    print("✅ Data loaded. Shape:", df.shape)

    # 🔹 Drop rows where target is missing
    df = df.dropna(subset=["Survived"])

    # 2. Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin"])  # Drop Cabin

    # 3. Drop irrelevant columns
    df = df.drop(columns=["Name", "Ticket"])  # keep PassengerId
    print("✅ Dropped irrelevant columns.")

    # 4. Encode categorical variables
    encoder = LabelEncoder()
    for col in ["Sex", "Embarked"]:
        df[col] = encoder.fit_transform(df[col].astype(str))

    print("✅ Categorical variables encoded.")

    # 5. Ensure numeric dtypes
    df = df.apply(pd.to_numeric, errors="coerce")
    print("✅ All columns converted to numeric.")

    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "..", "data", "titanic.csv")
    input_csv = os.path.normpath(input_csv)

    processed_df = preprocess_data(input_csv)
    print("\n🔹 Preview of preprocessed data:")
    print(processed_df.head())
    print("\n🔹 Data types:")
    print(processed_df.dtypes)
