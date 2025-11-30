import numpy as np
import os
import pandas as pd

def load_data(path="data/train.csv"):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = os.path.join(root_dir, path)
    df = pd.read_csv(full_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop obvious duplicates on stable identifiers where available
    if "passengerid" in df.columns:
        df = df.drop_duplicates(subset=["passengerid"])

    # Fix common spelling/format issues (example: embarked values)
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].astype(str).str.strip().str.upper()
        df["embarked"] = df["embarked"].replace({"SOUTHAMPTON":"S", "CHERBOURG":"C", "QUEENSTOWN":"Q"})

    # Handle missing values
    # Age: median by Pclass & Sex group
    if "age" in df.columns:
        df["age"] = df.groupby(["pclass","sex"])["age"].transform(lambda s: s.fillna(s.median()))

    # Fare: median by Pclass
    if "fare" in df.columns:
        df["fare"] = df.groupby(["pclass"])["fare"].transform(lambda s: s.fillna(s.median()))

    # Embarked: mode fill
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].replace({"nan": np.nan})
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # Cabin: high missing -> derive 'has_cabin' as feature, drop cabin
    if "cabin" in df.columns:
        df["has_cabin"] = df["cabin"].notna().astype(int)
        df = df.drop(columns=["cabin"])

    # Title feature from name
    if "name" in df.columns:
        df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.")
        common = {"Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Rare","Countess":"Rare",
                  "Sir":"Rare","Jonkheer":"Rare","Don":"Rare","Dona":"Rare","Dr":"Rare","Rev":"Rare","Col":"Rare","Major":"Rare","Capt":"Rare"}
        df["title"] = df["title"].replace(common)
        df["title"] = df["title"].fillna("Rare")

    # Family size features
    for col in ["sibsp","parch"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if set(["sibsp","parch"]).issubset(df.columns):
        df["family_size"] = df["sibsp"] + df["parch"] + 1
        df["is_alone"] = (df["family_size"] == 1).astype(int)

    # Ticket formatting clean
    if "ticket" in df.columns:
        df["ticket_prefix"] = df["ticket"].astype(str).str.replace(r"[\./]", " ", regex=True).str.extract(r"^([A-Za-z]+)").fillna("NONE")
        df = df.drop(columns=["ticket"])

    # Drop leakage columns not needed
    drop_cols = [c for c in ["name"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df