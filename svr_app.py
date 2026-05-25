import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

st.title("Support Vector Regressor")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.dataframe(df.head())

    target = st.selectbox("Select Target", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("R2 Score")
    st.write(r2_score(y_test, y_pred))

    st.subheader("MAE")
    st.write(mean_absolute_error(y_test, y_pred))