import streamlit as st
import joblib
import numpy as np
from utils import process_new

## Load the model

model=joblib.load('forst_model.pkl')

import pandas as pd

def categorize_loan_amount(loan_amount):
    bins = [0, 150, 300, float('inf')]
    labels = ['Low', 'Mid', 'High']
    
    # Use pd.cut to categorize the loan amount
    category = pd.cut([loan_amount], bins=bins, labels=labels, right=False)[0]
    
    return category


def LoanApproval():
    ## Title
    st.title('Loan Approval Predictor')
    st.markdown('---')

    ## input fields
    
    ## input fields
    Gender=st.selectbox("What is your gender?",options=['Male','Female'])
    Married=st.selectbox('Are you Married?',options=['Yes','No'])
    Dependents=st.selectbox('How Many Dependents do you have?',options=[0,1,2,3])
    Education=st.selectbox('What is your education level?',options=['Graduate','Not Graduate'])
    Self_Employed=st.selectbox('Are you Self Employed?',options=['Yes','No'])
    ApplicantIncome=st.number_input('What is your income?')
    CoapplicantIncome=st.number_input('What is your Co-Applicant income?')
    LoanAmount=st.number_input('What is the amount you want to loan?')
    Loan_Amount_Term=st.number_input('How Long you want the loan term to be in months?')
    Credit_History=st.selectbox('What is your credit history',options=[1.0,0.0])
    Property_Area=st.selectbox('Where is the property Area located?',options=['Semiurban','Urban','Rural'])
    LoanAmount_Category=categorize_loan_amount(LoanAmount)

   
    st.markdown('---')
    if st.button('Classify whether the loan will be accepted or not'):
        new_data=np.array([Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,LoanAmount_Category])

        X_processed=process_new(x_new=new_data)

    ## Predict
        y_pred=model.predict(X_processed)
        if y_pred == 1:
           y_pred ='The Loan is Accepted'
        else:
            y_pred='The Loan is Not Accepted'
    ## Display
        st.success(f'{y_pred}')
    
    return None



if __name__=='__main__':
    LoanApproval()