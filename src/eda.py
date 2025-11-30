import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def eda_section(df: pd.DataFrame):
    st.subheader("EDA")
    st.write("Shape:", df.shape)
    st.write(df.dtypes)
    st.write(df.describe(include="all"))

    if "survived" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="survived", ax=ax)
        ax.set_title("Survival counts")
        st.pyplot(fig)

    if set(["age","pclass"]).issubset(df.columns):
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="pclass", y="age", ax=ax)
        ax.set_title("Age distribution by passenger class")
        st.pyplot(fig)

    if "sex" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="sex", ax=ax)
        ax.set_title("Sex distribution")
        st.pyplot(fig)

    if set(["sex","pclass","survived"]).issubset(df.columns):
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="sex", y="survived", hue="pclass", ax=ax, estimator=lambda x: sum(x)/len(x))
        ax.set_title("Survival rate by sex and class")
        st.pyplot(fig)