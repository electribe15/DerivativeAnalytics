from fastapi import FastAPI
import streamlit as st
import pandas as pd
from src.database.db_connection import get_compliance_data

app = FastAPI()

def load_compliance_data():
    # Fetch compliance data from the database
    data = get_compliance_data()
    return data

def display_dashboard(data):
    st.title("ESG Compliance Monitoring Dashboard")
    
    # Display metrics
    st.header("Compliance Metrics")
    st.metric(label="Overall Compliance Score", value=data['overall_score'])
    
    # Display detailed compliance data
    st.header("Detailed Compliance Data")
    st.dataframe(data['detailed_compliance'])

def main():
    data = load_compliance_data()
    display_dashboard(data)

if __name__ == "__main__":
    main()