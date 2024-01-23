#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import csv
import ast
from sklearn.metrics.pairwise import euclidean_distances
import mysql.connector
from decouple import config
from dotenv import load_dotenv
import os

# Function Definitions
load_dotenv()
dbase = os.getenv("dbase")
duser = os.getenv("duser")
dpw = os.getenv("dpw")
dip = os.getenv("dip")

# Define a function to open database connection
def open_db_connection(db_name, user, password, host):
    # Try to connect to the database
    try:
        # Create a connection object with the name mydb
        mydb = mysql.connector.connect(database=db_name, user=user, password=password, host=host)
        # Print a success message
        print("Opened connection to database:", db_name)
        # Return the connection object
        return mydb
    # Handle any exceptions
    except mysql.connector.Error as e:
        # Print an error message
        print("Failed to open connection to database:", e)
        # Return None
        return None

# Define a function to close database connection
def close_db_connection(mydb, db_name):
    # Check if the connection object exists
    if mydb:
        # Try to close the connection
        try:
            # Close the connection
            mydb.close()
            # Print a success message
            print("Closed connection to database:", db_name)
        # Handle any exceptions
        except mysql.connector.Error as e:
            # Print an error message
            print("Failed to close connection to database:", e)



def defineDB():
   mydb=open_db_connection(
       dbase,
       duser,
       dpw,
       dip,
   )
   return mydb




def search_data(email):
    mydb = defineDB()
    mycursor = mydb.cursor()
    mycursor.execute(f"SELECT * FROM User WHERE email = '{email}'")
    rows = mycursor.fetchall()
    mycursor.close()
    close_db_connection(mydb, "User")
    return rows

def user_input(rows, categories):
    user = [rows[0][2], rows[0][4], "17-25", "student", categories]
    tutors = {'Math': 0, 'Science': 0, 'Social': 0, 'Technology': 0, 'Music': 0, 'Arts': 0, 'Multimedia': 0, 'Language': 0}
    if categories in tutors:
        tutors[categories] = 1
    lst = ast.literal_eval(rows[0][7])
    user.extend(tutors.values())
    user.extend(lst)
    return user

def insert_row_to_csv(row, file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
    lines.insert(1, row)
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lines)

def read_csv_to_dataframe(file_path):
    return pd.read_csv(file_path)

def drop_columns(df, columns):
    df.drop(columns, axis=1, inplace=True)

def convert_to_float32(df):
    return np.array(df, dtype=np.float32)

def load_model_and_predict(model_path, data):
    loaded_model = tf.saved_model.load(model_path)
    inference_fn = loaded_model.signatures["serving_default"]
    return inference_fn(tf.constant(data))["dense_2"]

def one_hot_encode(df, column):
    return pd.get_dummies(df, columns=[column]).astype(int)

def concat_dataframes(df1, df2):
    return pd.concat([df1, df2], axis=1)

def predict_pca(match_array, model_path):
    loaded_model = tf.saved_model.load(model_path)
    new_data_tf = tf.convert_to_tensor(match_array, dtype=tf.float32)
    pca_value = loaded_model.apply_pca(new_data_tf)
    return pca_value.numpy()

def get_guides_idx_filtered(user_id, user_df):
    user_destination = user_df.loc[user_id, 'Categories']
    filtered_users = user_df[(user_df['Categories'] == user_destination)]
    user_indices = [idx for idx in filtered_users.index if idx != user_id]
    return user_indices

def matchmaking(user_id, match_array, user_df):
    filtered_user_indices = get_guides_idx_filtered(user_id, user_df)
    if len(filtered_user_indices) == 0:
        return []

    user_indices = filtered_user_indices + [user_id]
    pca_data = predict_pca(match_array[user_indices], "pca_matching/4/")
    distances = euclidean_distances([pca_data[user_indices.index(user_id)]], pca_data)
    normalized_distances = 1 - distances / np.max(distances)
    scores = normalized_distances.flatten() * 100
    matches = sorted(zip(filtered_user_indices, scores), key=lambda x: x[1], reverse=True)

    match_results = []
    for match_index, score in matches:
        match_info = {
            "Nama": user_df.iloc[match_index]['Names'],
            "id": str(match_index),  # Adjusted to match_index, assuming this is equivalent to id-1
            "accuracy": f"{score:.2f}%"
        }
        match_results.append(match_info)

    return match_results

def get_tutor_data(match_results):
    mydb = defineDB()
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM Tutor")
    rows = mycursor.fetchall()
    mycursor.close()
    close_db_connection(mydb, "Tutor")

    # Filter rows based on matched IDs
    filtered_data = []
    for match in match_results:
        for row in rows:
            if str(row[0]) == match['id']:
                tutor_info = {
                    "id": str(row[0]),
                    "UserId": str(row[0]),
                    "Nama": row[1],
                    "hasPenis": row[2],
                    "AgesRanges": row[3],
                    "Specialization": row[4],
                    "Categories": row[5],
                    "AboutMe": row[6],
                    "SkillsAndExperiences": row[7],
                    "picture": row[8],
                    "price": row[9],
                    "accuracy": match['accuracy']
                }
                filtered_data.append(tutor_info)
                break

    response = {
        "error": "false",
        "message": "successfully getting the match XD",
        "data": filtered_data
    }

    return response


def delete_user_row_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
    del lines[1]
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lines)

def main_workflow(email, category):
    user_data = search_data(email)
    row = user_input(user_data, category)
    file_path = 'data/clean_data.csv'

    # Insert a new user row into the CSV
    insert_row_to_csv(row, file_path)

    # Process the data for matchmaking
    clean_df = read_csv_to_dataframe(file_path)
    match_df = clean_df.copy()
    drop_columns(match_df, ['Names', 'Age Ranges', 'Specialization', 'Categories'])
    persona_test_result = match_df.iloc[:, -25:]
    data_input = convert_to_float32(persona_test_result)
    model_path = '../personality_classification/persona_model/1'
    predictions = load_model_and_predict(model_path, data_input)
    predicted_clusters = np.argmax(predictions, axis=1)
    persona_type = pd.DataFrame({'Persona type': predicted_clusters})
    persona_type = one_hot_encode(persona_type, 'Persona type')
    match_df = concat_dataframes(match_df.iloc[:, :-25], persona_type)
    clean_df = concat_dataframes(clean_df.iloc[:, :-25], persona_type)
    match_array = match_df.values

    # Perform matchmaking and capture the results
    match_results = matchmaking(0, match_array, clean_df)

    # Delete the newly added user row from the CSV
    delete_user_row_from_csv(file_path)

    return get_tutor_data(match_results)
