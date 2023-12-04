import pandas as pd
import numpy as np

def read_data(file):
    data = pd.read_csv(file, sep=',')
    return data


def create_corpus(file):
    print("Creating corpus...")
    data = read_data(file)
    # data_train.columns = ['coders_classification', 'text', 'date', 'id', 'id_user', 'participant']

    # bf_scores = pd.read_json("bigfive/bigfive_scores.json")

    # data_train = data_train.merge(bf_scores, on="dev")
    data = data.drop(columns=['Unnamed: 0'])

    return data