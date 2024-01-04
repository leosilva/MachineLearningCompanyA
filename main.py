import read_data as rd
import vectorizer as vt
from datetime import datetime, timezone
import init_config as init
import pandas as pd
import balance_corpus as ba
import classifiers.classifiers as cl
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_validate, cross_val_score
import utils as ut
import numpy as np
import time
import gc
import argparse
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random


import warnings
warnings.filterwarnings('ignore')


# Generate a random integer based on current time
random.seed(int(time.time()))


def run(is_test, is_balance, which_models):
    utc_dt = datetime.now(timezone.utc)
    print("GENERAL ANALYSIS starting at {}".format(utc_dt.astimezone().isoformat()))

    param_dict = init.get_general_parameters(is_test)

    for d in param_dict['datasets']:
        d_train_name = d['train']
        d_val_name = d['val']
        for inst in param_dict['smote_instances']:
            result_map = init.get_result_map()
            for sca in param_dict['scales']:
                filename_train = d_train_name + '_' + str(inst) + '_instances_' + sca + '.csv'
                filename_val = d_val_name + '_' + str(inst) + '_instances_' + sca + '.csv'

                print(filename_val)

                # read files and create corpus
                data_train = rd.create_corpus('/Users/leosilva/Documents/Estudo/Doutorado/Coimbra/2019-2020/Disciplinas/Thesis/Papers/Logique_Sistemas_Research/dataset/' + filename_train)
                data_val = rd.create_corpus('/Users/leosilva/Documents/Estudo/Doutorado/Coimbra/2019-2020/Disciplinas/Thesis/Papers/Logique_Sistemas_Research/dataset/' + filename_val)

                features = param_dict['features']

                # count = 0
                for feat in features:
                    X = pd.DataFrame(data_train[feat])
                    y = pd.DataFrame(data_train['daysFromDevOrFixToTest_categories'])
                    # y = pd.DataFrame(data_train['storyPoints'])

                    X_val = pd.DataFrame(data_val[feat])
                    y_val = pd.DataFrame(data_val['daysFromDevOrFixToTest_categories'])
                    # y_val = pd.DataFrame(data_val['storyPoints'])

                    models = cl.get_models(which_models)

                    gc.collect()

                    print("Running models...")
                    for item in models.items():
                        hyperparams = item[1]
                        model = item[0]
                        model_name = model.__class__.__name__

                        gc.collect()

                        balance = 'smote'

                        print("Folding with KFold...")
                        for f in param_dict['folds']:
                            random_state = random.randint(1, 1000)

                            # adjusting random_state for algorythm
                            if 'random_state' in hyperparams.keys():
                                hyperparams['random_state'] = [random_state]

                            cv = StratifiedKFold(n_splits=f, shuffle=True, random_state=random_state)

                            print("Train test split...")
                            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                                test_size=0.2,
                                                                                random_state=random_state)

                            gc.collect()

                            # print("x has nan: ", X.isnull().values.any())
                            # print("y has nan: ", y.isnull().values.any())
                            # print(y.value_counts())

                            # for train_index, test_index in cv.split(X, y):
                            #     X_train_cv = X.iloc[train_index]
                            #     X_test_cv = X.iloc[test_index]
                            #     y_train_cv = y.iloc[train_index]
                            #     y_test_cv = y.iloc[test_index]

                            start = time.time()
                            # print("Running model {}".format(model_name))
                            # print(X_train.shape)
                            # print(y_train.shape)

                            gc.collect()

                            # grid_search = ut.perform_grid_search(model, hyperparams, cv, X_train, y_train)
                            model = model.fit(X_train, y_train.astype('int').values.ravel())
                            end = time.time()
                            # best_model = grid_search.best_estimator_
                            print("Training time: {}".format(end - start))

                            gc.collect()

                            start = time.time()
                            metrics = cross_validate(estimator=model,
                                                     X=X_train,
                                                     y=y_train.astype('int').values.ravel(),
                                                     cv=cv,
                                                     scoring=init.get_default_scoring(),
                                                     error_score="raise")

                            end = time.time()
                            print("Cross validate time: {}".format(end - start))

                            test_predictions = model.predict(X_test)
                            class_report_dict_test = classification_report(y_test.astype('int'), test_predictions, output_dict=True)

                            val_predictions = model.predict(X_val)
                            class_report_dict_val = classification_report(y_val.astype('int'), val_predictions,
                                                                      output_dict=True)

                            result_map["Algorithm"].append(model_name)
                            result_map["Train Accuracy"].append(round(np.mean(metrics['test_accuracy']), 4))
                            result_map["Train Precision"].append(round(np.mean(metrics['test_precision']), 4))
                            result_map["Train Recall"].append(round(np.mean(metrics['test_recall']), 4))
                            result_map["Train F1 Score"].append(round(np.mean(metrics['test_f1_score']), 4))

                            result_map["Test Accuracy"].append(round(np.mean(class_report_dict_test['accuracy']), 4))
                            result_map["Test Precision"].append(round(np.mean(class_report_dict_test['weighted avg']['precision']), 4))
                            result_map["Test Recall"].append(round(np.mean(class_report_dict_test['weighted avg']['recall']), 4))
                            result_map["Test F1 Score"].append(round(np.mean(class_report_dict_test['weighted avg']['f1-score']), 4))

                            result_map["Val. Accuracy"].append(round(np.mean(class_report_dict_val['accuracy']), 4))
                            result_map["Val. Precision"].append(
                                round(np.mean(class_report_dict_val['weighted avg']['precision']), 4))
                            result_map["Val. Recall"].append(
                                round(np.mean(class_report_dict_val['weighted avg']['recall']), 4))
                            result_map["Val. F1 Score"].append(
                                round(np.mean(class_report_dict_val['weighted avg']['f1-score']), 4))

                            result_map["AUC"].append(round(np.mean(metrics['test_roc_auc']), 4))

                            result_map["Bal. Strategy"].append(balance)
                            result_map["SMOTE Instances"].append(inst)
                            result_map["Scale"].append(sca)
                            result_map["Features"].append(feat)
                            result_map["Folds"].append(f)
                            # result_map["Hyper Params."].append(grid_search.best_params_)
                            result_map["Model"].append(model)
                            pd.set_option('display.max_colwidth', None)
                            pd.set_option('display.max_columns', None)
                            temp_df = pd.DataFrame(result_map)
                            print(temp_df.tail(1).T)

                            gc.collect()

            result_df = pd.DataFrame(result_map)
            result_df.sort_values(by='Val. F1 Score', ascending=False, inplace=True)

            ut.save_df_to_csv(result_df, d, inst)
                    # ut.save_best_model(result_df, d, inst, sca, count)

                    # count = count + 1

    utc_dtf = datetime.now(timezone.utc)
    print("GENERAL ANALYSIS ending at {}".format(utc_dtf.astimezone().isoformat()))

    utc_diff = utc_dtf - utc_dt
    minutes = divmod(utc_diff.seconds, 60)
    print('Time spent: ', minutes[0], 'minutes', minutes[1], 'seconds')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--is_test", help="Is the script running in TEST mode?")
    parser.add_argument("-m", "--models", help="Which models should run?")

    # 'svc', 'logistic-regression', 'random-forest',
    # 'multinomial-nb', 'bernoulli-nb', 'gaussian-nb', 'complement-nb',
    # 'decision-tree', 'mlp-classifier', 'knn'

    parser.add_argument("-b", "--is_balance",
                        help="Do you wish the script to perform balance strategies (SMOTE, UnderSampling, etc) for the dataset?")

    args = parser.parse_args()
    config = vars(args)
    is_test = config['is_test']
    models = config['models']
    is_balance = config['is_balance']

    if "," in models:
        models = models.split(',')
    else:
        models = [models]

    run(is_test=is_test, which_models=models, is_balance=is_balance)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
