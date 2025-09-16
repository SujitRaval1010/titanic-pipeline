import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_data(input_path: str):
    # Define correct column names
    col_names = [
        "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
        "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
    ]
    
    # 1. Load dataset
    df = pd.read_csv(input_path, names=col_names, header=0, quotechar='"')
    logging.info(f"✅ Data loaded. Shape: {df.shape}")

    # Drop rows where target is missing
    df = df.dropna(subset=["Survived"])

    # 2. Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin"])  # Drop Cabin

    # 3. Drop irrelevant columns
    df = df.drop(columns=["Name", "Ticket"])
    logging.info("✅ Dropped irrelevant columns.")

    # 4. Encode categorical variables
    encoder = LabelEncoder()
    for col in ["Sex", "Embarked"]:
        df[col] = encoder.fit_transform(df[col].astype(str))
    logging.info("✅ Categorical variables encoded.")

    # 5. Ensure numeric dtypes
    df = df.apply(pd.to_numeric, errors="coerce")
    logging.info("✅ All columns converted to numeric.")

    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv = os.path.join(script_dir, "..", "data", "titanic.csv")
    clean_csv = os.path.join(script_dir, "..", "data", "titanic_clean.csv")
    train_csv = os.path.join(script_dir, "..", "data", "titanic_train.csv")
    test_csv = os.path.join(script_dir, "..", "data", "titanic_test.csv")

    # Preprocess
    processed_df = preprocess_data(raw_csv)

    # Save full cleaned dataset
    processed_df.to_csv(clean_csv, index=False)
    logging.info(f"✅ Clean dataset saved at {clean_csv}")

    # Split into train/test
    train, test = train_test_split(processed_df, test_size=0.2, random_state=42)
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    logging.info(f"✅ Train saved at {train_csv} | Test saved at {test_csv}")

    # Preview
    print("\n🔹 Preview of cleaned data:")
    print(processed_df.head())
    print("\n🔹 Data types:")
    print(processed_df.dtypes)
