import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fpdf import FPDF
import io

# Streamlit UI
st.title("Hotel Revenue Management System")

# File Upload Section
st.sidebar.header("Upload Revenue Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(" ", "_")  # Normalize column names
    st.write("### Detected Columns in Uploaded File:")
    st.write(df.columns.tolist())
    
    required_columns = {'occupancy_rate', 'competitor_rate', 'room_price', 'revenue_generated'}
    df.columns = df.columns.str.lower()
    
    if not required_columns.issubset(df.columns):
        st.error(f"CSV file is missing required columns: {required_columns - set(df.columns)}")
        st.stop()
    st.success("File uploaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Display Data Sample
st.write("### Hotel Data Sample")
st.dataframe(df.head())

# Train ML Model
def train_model(df):
    X = df[['occupancy_rate', 'competitor_rate']]
    y = df['room_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, mae, rmse

model, mae, rmse = train_model(df)
st.write(f"### Model Performance\n- MAE: {mae:.2f}\n- RMSE: {rmse:.2f}")

# User Input for Price Prediction
st.sidebar.header("Predict Room Price")
occupancy_input = st.sidebar.slider("Occupancy Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
competitor_rate_input = st.sidebar.slider("Competitor Room Rate ($)", min_value=50, max_value=500, step=5)

def predict_price(model, occupancy, competitor_rate):
    return model.predict(np.array([[occupancy, competitor_rate]]))[0]

if st.sidebar.button("Predict Price"):
    predicted_price = predict_price(model, occupancy_input, competitor_rate_input)
    st.sidebar.success(f"Predicted Room Price: ${predicted_price:.2f}")

# Data Visualization
st.write("### Occupancy vs Revenue")
fig, ax = plt.subplots()
sns.scatterplot(x=df['occupancy_rate'], y=df['revenue_generated'], ax=ax)
ax.set_xlabel("Occupancy Rate (%)")
ax.set_ylabel("Revenue Generated ($)")
st.pyplot(fig)

# Generate PDF Report
def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Hotel Revenue Report", ln=True, align='C')
    pdf.ln(10)
    
    for index, row in df.head(10).iterrows():
        pdf.cell(200, 10, f"Room Price: {row['room_price']}, Occupancy: {row['occupancy_rate']}%, Revenue: {row['revenue_generated']}", ln=True)
    
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

pdf_data = generate_pdf(df)
st.download_button(label="Download Revenue Report as PDF", data=pdf_data, file_name="Revenue_Report.pdf", mime="application/pdf")