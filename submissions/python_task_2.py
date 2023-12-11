#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import networkx as nx
from datetime import datetime, time, timedelta


# In[1]:


def calculate_distance_matrix(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Create a graph to represent the toll locations and distances
    G = nx.Graph()

    # Add edges and distances to the graph
    for index, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])

    # Create a dictionary to store cumulative distances
    distance_matrix = {}

    # Calculate cumulative distances
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 == node2:
                distance_matrix[(node1, node2)] = 0
            elif G.has_edge(node1, node2):
                # If edge exists, use the known distance
                distance_matrix[(node1, node2)] = G[node1][node2]['distance']
            else:
                # If no direct edge, find the shortest path and sum the distances
                shortest_path = nx.shortest_path_length(G, source=node1, target=node2, weight='distance')
                distance_matrix[(node1, node2)] = shortest_path

    # Convert the distance dictionary to a DataFrame
    distance_df = pd.DataFrame(distance_matrix.values(), index=pd.MultiIndex.from_tuples(distance_matrix.keys()), columns=['Distance']).unstack()

    return distance_df

csv_file_path = "C:\\Users\\md musheeruddin\\Downloads\\dataset-3.csv"
result_df = calculate_distance_matrix(csv_file_path)

print(result_df)


# In[3]:


def unroll_distance_matrix(distance_matrix_df):
    # Create an empty DataFrame to store unrolled distance matrix
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate through the upper triangle of the distance matrix
    for i in range(len(distance_matrix_df.columns)):
        for j in range(i + 1, len(distance_matrix_df.columns)):
            id_start = distance_matrix_df.columns[i][1]
            id_end = distance_matrix_df.columns[j][1]
            distance = distance_matrix_df.iloc[i, j]

            # Append the values to the unrolled DataFrame
            unrolled_df = unrolled_df.append({'id_start': id_start, 'id_end': id_end, 'distance': distance}, ignore_index=True)

    return unrolled_df

result_df_unrolled = unroll_distance_matrix(result_df)

print(result_df_unrolled)


# In[5]:


def find_ids_within_ten_percentage_threshold(unrolled_df, reference_value):
    # Filter rows for the specified reference value
    reference_rows = unrolled_df[unrolled_df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_avg_distance = reference_rows['distance'].mean()

    # Calculate the threshold values (10% above and below the average)
    threshold_upper = reference_avg_distance * 1.1
    threshold_lower = reference_avg_distance * 0.9

    # Filter rows within the threshold range
    within_threshold = unrolled_df[(unrolled_df['distance'] >= threshold_lower) & (unrolled_df['distance'] <= threshold_upper)]

    # Extract unique values from the 'id_start' column and sort them
    result_ids = sorted(within_threshold['id_start'].unique())

    return result_ids

reference_value = 1 
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(result_df_unrolled, reference_value)

print(result_ids_within_threshold)


# In[6]:


def calculate_toll_rate(unrolled_df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Add columns for each vehicle type with their respective toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        unrolled_df[vehicle_type] = unrolled_df['distance'] * rate_coefficient

    return unrolled_df

result_df_with_toll_rates = calculate_toll_rate(result_df_unrolled)

print(result_df_with_toll_rates)


# In[12]:


def calculate_time_based_toll_rates(toll_rate_df):
    # Define time ranges for weekdays and weekends
    weekday_time_ranges = [(time(0, 0), time(10, 0)), (time(10, 0), time(18, 0)), (time(18, 0), time(23, 59, 59))]
    weekend_time_range = (time(0, 0), time(23, 59, 59))

    # Create a list to store rows of the resulting DataFrame
    result_rows = []

    # Iterate over unique (id_start, id_end) pairs
    unique_pairs = toll_rate_df[['id_start', 'id_end']].drop_duplicates()
    for _, pair in unique_pairs.iterrows():
        for day in range(7):  # 0 represents Monday, 1 for Tuesday, and so on
            for start_time, end_time in weekday_time_ranges if day < 5 else [weekend_time_range]:
                start_datetime = datetime.combine(datetime.today(), start_time) + timedelta(days=day)
                end_datetime = datetime.combine(datetime.today(), end_time) + timedelta(days=day)

                # Filter rows for the current (id_start, id_end) pair and time interval
                filtered_rows = toll_rate_df[
                    (toll_rate_df['id_start'] == pair['id_start']) &
                    (toll_rate_df['id_end'] == pair['id_end']) &
                    (toll_rate_df['start_time'] >= start_datetime.time()) &
                    (toll_rate_df['end_time'] <= end_datetime.time())
                ]

                # Apply discount factors based on the time interval
                discount_factor = 0.7 if day >= 5 else 0.8 if start_time < time(10, 0) or (start_time >= time(18, 0) and end_time <= time(23, 59, 59)) else 1.2

                # Calculate time-based toll rates and append to the result_rows list
                for _, row in filtered_rows.iterrows():
                    result_rows.append({
                        'id_start': pair['id_start'],
                        'id_end': pair['id_end'],
                        'start_day': row['start_day'],
                        'end_day': row['end_day'],
                        'start_time': start_time,
                        'end_time': end_time,
                        'time_based_rate': row['distance'] * discount_factor
                    })

    result_df = pd.DataFrame(result_rows)

    return result_df

result_df_time_based = calculate_time_based_toll_rates(result_df_with_toll_rates)

print(result_df_time_based)


# In[ ]:




