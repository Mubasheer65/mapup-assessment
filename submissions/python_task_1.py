#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


def generate_car_matrix(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Create a pivot table with id_1 as index, id_2 as columns, and car as values
    car_matrix = pd.pivot_table(df, values='car', index='id_1', columns='id_2', fill_value=0)

    # Set the diagonal values to 0 using numpy
    np.fill_diagonal(car_matrix.values, 0)

    return car_matrix

file_path = "C:\\Users\\md musheeruddin\\Downloads\\dataset-1.csv"
result_df = generate_car_matrix(file_path)
print(result_df)



# In[9]:


def get_type_count(df):
    # Add a new categorical column 'car_type' based on values of the 'car' column
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'])

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))

    return type_count

file_path = "C:\\Users\\md musheeruddin\\Downloads\\dataset-1.csv"
df = pd.read_csv(file_path)
result_dict = get_type_count(df)
print(result_dict)


# In[10]:


def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean value
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

file_path = "C:\\Users\\md musheeruddin\\Downloads\\dataset-1.csv"
df = pd.read_csv(file_path)
result_list = get_bus_indexes(df)
print(result_list)


# In[13]:


def filter_routes(df):
    # Calculate the average value of the 'truck' column for each 'route'
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' column is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

file_path = "C:\\Users\\md musheeruddin\\Downloads\\dataset-1.csv"
df = pd.read_csv(file_path)
result_list = filter_routes(df)
print(result_list)


# In[14]:


def multiply_matrix(result_df):
    modified_df = result_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

modified_result_df = multiply_matrix(result_df)
print(modified_result_df)


# In[26]:


import pandas as pd

def verify_time_completeness(df):
    try:
        # Combine 'startDay' and 'startTime' columns to create a 'start_timestamp' column
        df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

        # Combine 'endDay' and 'endTime' columns to create an 'end_timestamp' column
        df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    except pd.errors.OutOfBoundsDatetime as e:
        print(f"Error: {e}")
        return pd.Series(False, index=df.set_index(['id', 'id_2']).index)

    # Group by (id, id_2) and check if timestamps cover a full 24-hour period and span all 7 days
    result_series = df.groupby(['id', 'id_2']).apply(lambda group:
        ((group['start_timestamp'].min() <= pd.to_datetime('00:00:00')) &
         (group['end_timestamp'].max() >= pd.to_datetime('23:59:59')) &
         set(group['start_timestamp'].dt.day_name()) == set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
    )

    return result_series

# Example usage:
file_path = "C:\\Users\\md musheeruddin\\Downloads\\dataset-2.csv"
df = pd.read_csv(file_path)
result_bool_series = verify_time_completeness(df)
print(result_bool_series)


# In[ ]:





# In[ ]:




