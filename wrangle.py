" In my wrangle py"


from cgi import test
from lib2to3.pgen2.pgen import DFAState
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import env
from pydataset import data
import scipy
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')



def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_zillow_data():
    """bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips"""
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql("""SELECT parcelid,
             bedroomcnt,
             bathroomcnt,
             calculatedfinishedsquarefeet,
             taxvaluedollarcnt,
             yearbuilt,
             taxamount,
             fips
          FROM properties_2017
          JOIN propertylandusetype USING(propertylandusetypeid) 
          WHERE (propertylandusetypeid = 261) OR (propertylandusetypeid = 279) """, get_connection('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)
        #changing it csv because its a csv

        # Return the dataframe to the calling code
        return df   




def clean_zillow(df):
    """Prepares acquired zillow data for exploration"""
    # Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # Drop all rows with any Null values, assign to df, and verify.
    df = df.dropna()
    # Change all column data tyes to int64, reassign to df, and verify.
    df = df.astype('int')
    # getting rid of zero values in bedroomcnt and bathroomcnt
    df = df[(df.bedroomcnt != 0) & (df.bathroomcnt != 0)]
    return df


# def split_zillow_data(df):
#     '''
#     take in a DataFrame and return train, validate, and test DataFrames; stratify on species.
#     return train, validate, test DataFrames.
#     '''
    
#     # splits df into train_validate and test using train_test_split() stratifying on species to get an even mix of each species
#     train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    
#     # splits train_validate into train and validate using train_test_split() stratifying on species to get an even mix of each species
#     train, validate = train_test_split(train_validate, 
#                                        test_size=.3, 
#                                        random_state=123, 
#                                        stratify=train_validate.species)
#     return train, validate, test


# def prep_zillow(df):
#     '''Prepares acquired Iris data for exploration'''
    
#     # drop column using .drop(columns=column_name)
#     df = df.drop(columns='species_id')
    
#     # remame column using .rename(columns={current_column_name : replacement_column_name})
#     df = df.rename(columns={'species_name':'species'})
    
#     # create dummies dataframe using .get_dummies(column_name,not dropping any of the dummy columns)
#     dummy_df = pd.get_dummies(df['species'], drop_first=False)
    
#     # join original df with dummies df using .concat([original_df,dummy_df], join along the index)
#     df = pd.concat([df, dummy_df], axis=1)
    
#     # split data into train/validate/test using split_data function
#     train, validate, test = split_iris_data(df)
    
#     return train, validate, test
