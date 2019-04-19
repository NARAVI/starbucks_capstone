# starbucks_capstone
udacity_ds_capstone

Medium Link-https://medium.com/@ravikishore268/starbucks-capstone-challenge-7de0013df12e

github link- https://github.com/NARAVI/starbucks_capstone
## Project Overview

The Starbucks project was my capstone project for the Udacity Data Scientist Nanodegree. This data was provided by Starbucks to simulate their customers and transactions to find out  whether a customer will respond to an offer..

The different files that were used to create this project are located in the Profile Portfolio and Transcript Data.zip file. Here is a quick overview of each file:

1. Profile - Dimensional data about each person, including their age, salary, and gender. There is one unique customer for each record.
2. Portfolio - Information about the promotional offers that are possible to receive, and basic information about each one including the promotional type, duration of promotion, reward, and how the promotion was distributed to customers
3. Transcript - Records show the different steps of promotional offers that a customer received. The different values of receiving a promotion are receiving, viewing, and completing. You also see the different transactions that a person made in the time since he became a customer. With all records, you see the day that they intereacted with Starbucks and the amount that it is worth.

Profile Portfolio and Transcript Data

Three JSON files that show profiles of customers, promotional deals that are offered, and the transaction history of customers.

portfolio.json

id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - UNKNOWN
channels (list of strings)
profile.json

age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income
transcript.json

event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record
Starbucks Capstone Project.ipynb

## Files

1. Starbucks_Capstone_notebook.ipynb
    * Jupyter notebook that performs three tasks:
        * Combines offer portfolio, customer demographic, and customer transaction data
        * Generates training customer demographic data visualizations and computes summary statistics
        * Generates logistic regression, random forest, & gradient boosting models
2. clean_data.py
    * Python software that combines offer portfolio, customer demographic, and customer transaction data
3. exploratory_data_analysis.py
    * Generates training customer demographic data visualizations and computes summary statistics
4. LICENSE
    * Repository license file
        * .gitignore
    * Describes files and/or directories that should not be checked into revision control
5. README.md
    * Markdown file that summarizes this repository

## Python Libraries Used

* Python Data Analysis Library
* Numpy
* Matplotlib
* seaborn: Statistical Data Visualization
* re: Regular expression operations
* os — Miscellaneous operating system interfaces
* scikit-learn: Machine Learning in Python
* Joblib: running Python functions as pipeline jobs



Jupyter Notebook that shows the start to finish of the this project. This starts with data cleanup and exploration, top level statsitics of customers and their transactions, and fitting the model and its prediction.
