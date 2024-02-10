from sklearn.model_selection import GridSearchCV
# import joblib
import pickle
import os


# execution = "without_grid_search_c"


def perform_grid_search(model, params, cv, X_train_selected, y_train_cv):
    grid_search = GridSearchCV(model, param_grid=params, cv=cv, refit = True, n_jobs=-1)
    grid_search.fit(X_train_selected, y_train_cv.astype('int').values.ravel())

    return grid_search

def get_random_state():
    return 42


def save_df_to_csv(df, dataset, inst):

    if not os.path.exists('best_models/feb/'):
        os.makedirs('best_models/feb/')

    filename = 'best_models/feb/' + dataset['train'] + '_' + str(inst) + '_instances.csv'

    df.to_csv(filename, index=None, sep=',', mode='w')


def save_best_model(result_df, dataset, inst, sca):
    filename = ''
    result_df = result_df.reset_index()
    models = result_df.head(10)

    for index, row in models.iterrows():
        m = row['Model']

        filename = 'best_models/feb/model_' + str(index + 1) + '_' + dataset['train'] + '_' + str(inst) + '_instances_' + sca + '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(m, file)