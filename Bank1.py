import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Predicting Customers who might terminate their Credit Card Services")
st.markdown( "A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction. The model is trained with the Credit Card customers dataset on Kaggle")

data_url = "Bank_Churners_Clean.csv"
data_unclean_url = "BankChurners.csv"

# Creating a side bar for users to explore
st.sidebar.markdown("## Side Bar")
st.sidebar.markdown("Use this panel to explore the dataset, create viz, and make predictions.")

df = pd.read_csv(data_url)
df_unclean = pd.read_csv(data_unclean_url)

# Showing the original raw data
if st.checkbox("Here, you can check the raw data", False):
    st.subheader('Raw data')
    st.write(df_unclean)
st.markdown("The dataset has 10,127 entries with 20 features.")


st.title('Quick  Exploration')
st.sidebar.subheader('Quick  Exploration')
st.markdown("Tick the box on the side panel to explore the dataset.")


if st.sidebar.checkbox('Basic info'):
    if st.sidebar.checkbox('Dataset Quick Look'):
        st.subheader('Dataset Quick Look:')
        st.write(df.head(10))
    if st.sidebar.checkbox("Show Columns"):
        st.subheader('Show Columns List')
        all_columns = df.columns.to_list()
        st.write(all_columns)

    if st.sidebar.checkbox('Statistical Description'):
        st.subheader('Statistical Data Descripition')
        st.write(df.describe())
    if st.sidebar.checkbox('Missing Values'):
        st.subheader('Missing values')
        st.write(df.isnull().sum())
st.markdown(
"""
Observations
- Attrition_Flag is the target feature, which tells us whether the credit card account is closed or not.
- There are no null values in the dataset. There are 10 integer data types, 5 float data types and 6 object data types.
- The other features are either objects or numbers, and need to be explored further. They describe various characteristics of the credit card accounts.
"""
)


# Visualization part
st.title('Explore Data with Visualization')
st.markdown('Tick the box on the side panel to create your own Visualization and explore the data.')
st.sidebar.subheader('Explore with Visualization')
if st.sidebar.checkbox('Data Visualization'):

    if st.sidebar.checkbox('Count Plot'):
        st.subheader('Count Plot')
        st.info("If error, please adjust column name on side panel.")
        column_count_plot = st.sidebar.selectbox(
            "Choose a column to plot count. Try Selecting Categorical Columns (e.g. Gender) ", df.columns)
        hue_opt = st.sidebar.selectbox("Optional categorical variables (countplot hue). Try Selecting Attrition Flag ",
                                       df.columns.insert(0, None))

        fig = sns.countplot(x=column_count_plot, data=df, hue=hue_opt)
        st.pyplot()

    if st.sidebar.checkbox('Histogram | Distplot'):
        st.subheader('Histogram | Distplot')
        st.info("If error, please adjust column name on side panel.")
        if st.checkbox('Dist plot'):
            column_dist_plot = st.sidebar.selectbox(
                "Optional categorical variables (countplot hue)", df.columns)
        fig = sns.distplot(df[column_dist_plot])
        st.pyplot()

    if st.sidebar.checkbox('Boxplot'):
        st.subheader('Boxplot')
        st.info("If error, please adjust column name on side panel.")
        column_box_plot_X = st.sidebar.selectbox("X (Choose a column). Try Selecting Attrition Flag:",
                                                 df.columns.insert(0, None))
        column_box_plot_Y = st.sidebar.selectbox("Y (Choose a column - only numerical). Try Selecting Total_Trans_Ct",
                                                 df.columns)

        fig = sns.boxplot(x=column_box_plot_X, y=column_box_plot_Y, data=df, palette="Set3")
        st.pyplot()

    if st.sidebar.checkbox('Correlation Map'):
        st.subheader('Correlation Map')
        st.info("If error, please adjust column name on side panel.")

        plt.figure(figsize=(25, 20), dpi=200)
        fig = sns.heatmap(df.corr(), annot=True, vmin=-0.5, vmax=1, cmap='coolwarm', linewidths=0.75)
        st.pyplot()

st.markdown(
"""
Observations
- Most customers have exactly a 36-month, i.e., 3 year relationship with the bank. This is an outlier value, that may have occured as 3 years is a rounded figure, and most customers willing to end their relationship with the bank around that time tend to round it to 3 years.
- The credit limit has a skewed normal distribution with mean 2,500. There are a few outliers with a high credit limit near 35,000.
- A majority of the customers have stuck to their spending habits. The change in the number of transactions has more or less remained the same through quarters (0 to 1.5). However, there is an outlier of 3.5.
- Most customers have made a total of between 40-80 transactions in the last 12 months.
- Customers who have attrited have a greater tendency to have a credit limit of 0 before they shut their account.
- Across different credit limits, those with higher transaction amounts are less likley to churn.
- The age of the customer has a strong postitive correlation with the period of relationship with the bank. This is obvious as older people are more likely to be with the bank for a longer time.
"""
)

# Give prediction based on user input
st.title('Explore Model')
st.markdown('Tick the box on the side panel to explore different model scores, or predict if the customer will terminate their services or not.')
st.sidebar.subheader('Explore Model')
if st.sidebar.checkbox('Features'):
    def user_input_features():
        Total_Relationship_Count = st.sidebar.slider('Total Relationship Count', 0, 10)
        Months_Inactive_12_mon = st.sidebar.slider('Months Inactive 12_month', 0, 12)
        Contacts_Count_12_mon = st.sidebar.slider('Contacts Count 12_month', 0, 10)
        Total_Revolving_Bal = st.sidebar.slider('Total Revolving Balance', 0, 2600)
        Total_Amt_Chng_Q4_Q1 = st.sidebar.slider('Total Amount Change_Q4_Q1', 0, 4)
        Total_Trans_Amt = st.sidebar.slider('Total Transaction Amount', 0, 19000)
        Total_Trans_Ct = st.sidebar.slider('Total Transaction Count', 0, 140)
        Total_Ct_Chng_Q4_Q1 = st.sidebar.slider('Total Change Transaction Count_Q4_Q1', 0, 5)
        Avg_Utilization_Ratio = st.sidebar.slider('Average Card Utilization Ratio', 0, 5)
        data = {'Total Relationship Count': Total_Relationship_Count,
                'Months Inactive 12_month': Months_Inactive_12_mon,
                'Contacts Count 12_month': Contacts_Count_12_mon,
                'Total Revolving Balance': Total_Revolving_Bal,
                'Total Amount Change_Q4_Q1': Total_Amt_Chng_Q4_Q1,
                'Total Transaction Amount': Total_Trans_Amt,
                'Total Transaction Count': Total_Trans_Ct,
                'Total Change Transaction Count_Q4_Q1': Total_Ct_Chng_Q4_Q1,
                'Average Card Utilization Ratio': Avg_Utilization_Ratio}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Reads in saved classification model
    load_rfc = pickle.load(open('random_forest.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_rfc.predict(input_df)
    prediction_proba = load_rfc.predict_proba(input_df)

    st.subheader('Prediction')
    customer_types = np.array(['Existing Customer', 'Attrited Customer'])
    st.write(customer_types[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

if st.sidebar.checkbox('Choose Model'):
    st.sidebar.selectbox('Choose a model to check their performance',['Random Forest Classifier', 'Logistic Regression','Support Vector Machines'])
    if ('Random Forest Classifier'):
        st.subheader("Classification Report")
        df['Attrition_Flag'] = df['Attrition_Flag'].replace({"Existing Customer": 0, "Attrited Customer": 1})
        df = df[
            ['Attrition_Flag', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal', 'Avg_Utilization_Ratio',
             'Total_Trans_Amt',
             'Total_Relationship_Count', 'Total_Amt_Chng_Q4_Q1', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon']]
        X = df.drop('Attrition_Flag', axis=1)
        y = df['Attrition_Flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        # Oversample the train dataset
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        rfc = RandomForestClassifier()
        rfc.fit(X_res, y_res)
        predictions = rfc.predict(X_test)

        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.table(report_df)


        # plot the confusion matrix
        st.subheader("Confusion Matrix")
        fig5 = plt.figure()
        conf_matrix = confusion_matrix(rfc.predict(X_test), y_test)
        sns.heatmap(conf_matrix, annot=True, fmt = 'g')
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        st.pyplot(fig5)

st.title('Inferences')
st.markdown(
"""
- The first model applied here was logistic regression without an sampling and that didn't give satisfactory results as the data seems highly imbalanced(number of observations belonging to one group or class is significantly higher than those belonging to the other classes). 
Thus, in the next step Logistic Regression was analyzed after oversampling the data. Oversampling helps me to upsample the number of attrition samples to match them with the regular customer sample size. This data transformation gives the model a better chance of catching small details for the high score.
This gave improved results but still not that accurate.
- The second model used was Random Forest. Here, the algorithm splits the dataset into independent random subsamples and selects subsamples of features (draw with replacement), and fits a decision tree classfier to each subsample. It then averages the estimates to improve the prediction accuracy.
This had the best accuracy of almost 96.12%.
- SVM algorithm itries to find a hyperplane in an N-dimensional space that distinctly classifies the data points. The dimension of the hyperplane depends upon the number of features. If the number of input features is two, then the hyperplane is just a line. If the number of input features is three, then the hyperplane becomes a 2-D plane. It becomes difficult to imagine when the number of features exceeds three. 
The third model SVM was somehow the least accurate one.
"""
)

st.title('Conclusion')
st.markdown(
"""
- There are 16.07pc of customers who have terminated the services.
- The proportion of attrited customers by gender: there are 14.4% more male than female who have attrited. 
- Customers who have left are highly educated - A high proportion of education level of attrited customer is Graduate level (29.9%), followed by Post-Graduate level (18.8%)
- A high proportion of marital status of customers who have left are Married (43.6%), followed by Single (41.1%) compared to Divorced (7.4%) and Unknown (7.9%) status - marital status of the attributed customers are highly clustered in Married status and Single
- One can see from the proportion of income category of attrited customer, it is highly concentrated around 60K - 80K income (37.6%), followed by Less than 40K income (16.7%) compare to attrited customers with higher annual income of 80K-120K(14.9%) and over 120K+ (11.5%). I assume that customers with higher income doesn't likely leave their credit card services than meddle-income customers
"""
)

