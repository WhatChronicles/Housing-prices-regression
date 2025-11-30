import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def eda_section(df: pd.DataFrame):
    st.subheader("EDA")
    # Basic overview
    st.write("Shape:", df.shape)
    st.write(df.dtypes)
    st.write(df.describe(include="all"))

    # Bar chart: Survived count
    if "survived" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="survived", ax=ax)
        ax.set_title("Survival counts")
        st.pyplot(fig)

    # Boxplot: Age by Pclass
    if set(["age","pclass"]).issubset(df.columns):
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="pclass", y="age", ax=ax)
        ax.set_title("Age distribution by passenger class")
        st.pyplot(fig)

    # Bar: Sex distribution
    if "sex" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="sex", ax=ax)
        ax.set_title("Sex distribution")
        st.pyplot(fig)

    # Relevant plot: Survival rate by Sex and Pclass
    if set(["sex","pclass","survived"]).issubset(df.columns):
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="sex", y="survived", hue="pclass", ax=ax, estimator=lambda x: sum(x)/len(x))
        ax.set_title("Survival rate by sex and class")
        st.pyplot(fig)