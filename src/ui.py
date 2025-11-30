import streamlit as st
import pandas as pd

def user_input_form():
    st.subheader("Enter passenger features to predict survival")
    pclass = st.selectbox("Pclass", [1,2,3], index=2)
    sex = st.selectbox("Sex", ["male","female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
    sibsp = st.number_input("Siblings/Spouses Aboard (sibsp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents/Children Aboard (parch)", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=7.25, step=0.5)
    embarked = st.selectbox("Embarked", ["S","C","Q"])
    has_cabin = st.selectbox("Has cabin", [0,1])
    title = st.selectbox("Title", ["Mr","Mrs","Miss","Master","Rare"])
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    ticket_prefix = st.selectbox("Ticket prefix", ["NONE","PC","A","SC"])

    df = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked,
        "has_cabin": has_cabin,
        "title": title,
        "family_size": family_size,
        "is_alone": is_alone,
        "ticket_prefix": ticket_prefix
    }])
    return df