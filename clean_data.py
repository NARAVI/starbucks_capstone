from datetime import datetime
import numpy as np
import pandas as pd
import re
import os
import progressbar
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


def update_column_name(dataframe,
                       old_column_name,
                       new_column_name):
    """ Updates a Pandas DataFrame column name
    INPUT:
        dataframe: Pandas DataFrame object
        old_column_name: String that stores the old column name
        new_column_name: String that stores the new column name
    OUTPUT:
        column_names: np.array that stores the updated Pandas DataFrame
                      column names"""
    column_names = dataframe.columns.values
    
    select_data = np.array([elem == old_column_name for elem in column_names])

    column_names[select_data] = new_column_name
        
    return column_names


def clean_portfolio(data_dir="./data"):
    """ 
    Transforms a DataFrame containing offer ids and meta data about 
    each offer (duration, type, etc.)
    INPUT:
        (Optional) data_dir: String that stores the full path to the
                             data directory
    OUTPUT:
        portfolio: DataFrame containing offer ids and meta data about 
                   each offer (duration, type, etc.)
    """
    portfolio = pd.read_json(os.path.join(data_dir, 'portfolio.json'),
                             orient='records',
                             lines=True)
    
    # Change the name of the 'id' column to 'offerid'
    columns = update_column_name(portfolio,
                                 'id',
                                 'offerid')
    
    # Change the name of the 'duration' column to 'durationdays'
    portfolio.columns = update_column_name(portfolio,
                                           'duration',
                                           'durationdays')

    # Remove underscores from column names
    portfolio.columns = [re.sub('_', '', elem) for elem in columns]

    # Initialize a list that stores the desired output DataFrame 
    # column ordering
    column_ordering = ['offerid',
                       'difficulty',
                       'durationdays',
                       'reward']

    # One hot encode the 'offertype' column
    offertype_df = pd.get_dummies(portfolio['offertype'])

    column_ordering.extend(offertype_df.columns.values)

    # One hot encode the 'channels' columns
    ml_binarizerobj = MultiLabelBinarizer()
    ml_binarizerobj.fit(portfolio['channels'])

    channels_df =\
        pd.DataFrame(ml_binarizerobj.transform(portfolio['channels']),
        columns=ml_binarizerobj.classes_)

    column_ordering.extend(channels_df.columns.values)

    # Replace the 'offertype' and 'channels' columns
    portfolio = pd.concat([portfolio, offertype_df, channels_df], axis=1)

    portfolio = portfolio.drop(columns=['offertype', 'channels'])

    # Return the "cleaned" portfolio data
    return portfolio[column_ordering]


def convert_to_datetime(elem):
    """Converts a string to a datetime object
    
    INPUT:
        elem: String that stores a date in the %Y%m%d format
    OUTPUT:
        datetimeobj: Datetime object"""
    return datetime.strptime(str(elem), '%Y%m%d')


def clean_profile(data_dir = "./data"):
    """ Transforms a DataFrame that contains demographic data for each 
    customer
    
    INPUT:
        (Optional) data_dir: String that stores the full path to the
                             data directory
    
    OUTPUT:
        profile: DataFrame that contains demographic data for each 
                 customer
    """
    profile = pd.read_json('data/profile.json',
                           orient='records',
                           lines=True)

    # Remove customers with N/A income data
    profile = profile[profile['income'].notnull()]

    # Remove customers with unspecified gender
    profile = profile[profile['gender'] != 'O']
    profile = profile.reset_index(drop=True)

    # Change the name of the 'id' column to 'customerid'
    profile.columns = update_column_name(profile,
                                         'id',
                                         'customerid')

    # Initialize a list that describes the desired DataFrame column
    # ordering
    column_ordering = ['customerid',
                       'gender',
                       'income']

    # Transform the 'became_member_on' column to a datetime object
    profile['became_member_on'] =\
        profile['became_member_on'].apply(convert_to_datetime)

    # One hot encode a customer's membership start year
    profile['membershipstartyear'] =\
        profile['became_member_on'].apply(lambda elem: elem.year)

    membershipstartyear_df = pd.get_dummies(profile['membershipstartyear'])
    column_ordering.extend(membershipstartyear_df.columns.values)

    # One hot encode a customer's age range
    min_age_limit = np.int(np.floor(np.min(profile['age'])/10)*10)
    max_age_limit = np.int(np.ceil(np.max(profile['age'])/10)*10)

    profile['agerange'] =\
        pd.cut(profile['age'],
               (range(min_age_limit,max_age_limit + 10, 10)),
               right=False)

    profile['agerange'] = profile['agerange'].astype('str')

    agerange_df = pd.get_dummies(profile['agerange'])
    column_ordering.extend(agerange_df.columns.values)

    # Transform a customer's gender from a character to a number
    binarizerobj = LabelBinarizer()
    profile['gender'] = binarizerobj.fit_transform(profile['gender'])

    gender_integer_map = {}
    for elem in binarizerobj.classes_:
        gender_integer_map[elem] = binarizerobj.transform([elem])[0,0]

    # Appened one hot encoded age range and membership start year variables
    profile = pd.concat([profile,
                         agerange_df,
                         membershipstartyear_df], axis=1)

    # Drop depcreated columns
    profile = profile.drop(columns=['age',
                                    'agerange',
                                    'became_member_on',
                                    'membershipstartyear'])

    # Return a DataFrame with "clean" customer profile data
    return profile[column_ordering], gender_integer_map

def clean_transcript(profile,
                     data_dir = './data'):
    """ Transforms a DataFrame that contains records for transactions, offers
    received, offers viewed, and offers completed
    INPUT:
        profile: DataFrame that contains demographic data for each 
                 customer
        (Optional) data_dir: String that stores the full path to the
                             data directory
    OUTPUT:
        offer_data: DataFrame that describes customer offer data
        transaction: DataFrame that describes customer transactions
    """
    transcript = pd.read_json(os.path.join(data_dir,
                                           'transcript.json'),
                              orient='records',
                              lines=True)

    # Change the name of the 'person' column to 'customerid'
    transcript.columns = update_column_name(transcript,
                                            'person',
                                            'customerid')

    # Remove customer id's that are not in the customer profile DataFrame
    select_data = transcript['customerid'].isin(profile['customerid'])
    transcript = transcript[select_data]

    percent_removed = 100 * (1 - select_data.sum() / select_data.shape[0])
    print("Percentage of transactions removed: %.2f %%" % percent_removed)

    # Convert from hours to days
    transcript['time'] /= 24.0
    
    # Change the name of the 'time' column to 'timedays'
    transcript.columns = update_column_name(transcript,
                                            'time',
                                            'timedays')

    # Select customer offers
    pattern_obj = re.compile('^offer (?:received|viewed|completed)')

    h_is_offer = lambda elem: pattern_obj.match(elem) != None

    is_offer = transcript['event'].apply(h_is_offer)

    offer_data = transcript[is_offer].copy()
    offer_data = offer_data.reset_index(drop=True)

    # Initialize a list that describes the desired output DataFrame
    # column ordering
    column_order = ['offerid', 'customerid', 'timedays']

    # Create an offerid column
    offer_data['offerid'] =\
        offer_data['value'].apply(lambda elem: list(elem.values())[0])

    # Transform a column that describes a customer offer event
    pattern_obj = re.compile('^offer ([a-z]+$)')

    h_transform = lambda elem: pattern_obj.match(elem).groups(1)[0]

    offer_data['event'] = offer_data['event'].apply(h_transform)

    # One hot encode customer offer events
    event_df = pd.get_dummies(offer_data['event'])
    column_order.extend(event_df.columns.values)

    # Create a DataFrame that describes customer offer events
    offer_data = pd.concat([offer_data, event_df], axis=1)
    offer_data.drop(columns=['event', 'value'])
    offer_data = offer_data[column_order]

    # Select customer transaction events
    transaction = transcript[is_offer == False]
    transaction = transaction.reset_index(drop=True)

    # Transform customer transaction event values
    transaction['amount'] =\
        transaction['value'].apply(lambda elem: list(elem.values())[0])

    # Create a DataFrame that describes customer transactions
    transaction = transaction.drop(columns=['event', 'value'])
    column_order = ['customerid', 'timedays', 'amount']
    transaction = transaction[column_order]

    return offer_data, transaction


