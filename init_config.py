import sklearn.metrics as me

def get_general_parameters(is_test):
    if is_test == 'True':
        print("Executing in TEST mode...")
        return {
            # "ngram": [(1,2)],
            "balance": [''],
            # "percentage_features": [10],
            # "feature_selection": ['none'],
            "folds": [5, 5, 5]
        }
    else:
        print("Executing in PROD mode...")
        return {
            "features": [
<<<<<<< HEAD
                # PARA Y DAYS FROM DEV TO TEST
                # ['storyPoints', 'polarity_mean', 'O', 'C', 'E', 'A', 'N'],
                ['storyPoints', 'polarity_mean', 'stress_mean', 'O', 'C', 'E', 'A', 'N'],
                # ['storyPoints', 'polarity', 'O', 'C', 'E', 'A', 'N'],
                # ['storyPoints', 'polarity', 'stress', 'O', 'C', 'E', 'A', 'N'],
=======
                # PARA Y DAYS FROM DEV TO TEST, TODAS OS BIG FIVE
                # ['storyPoints', 'polarity_mean', 'O', 'C', 'E', 'A', 'N'],
                # ['storyPoints', 'polarity_mean', 'stress_mean', 'O', 'C', 'E', 'A', 'N'],
                # ['storyPoints', 'polarity', 'O', 'C', 'E', 'A', 'N'],
                # ['storyPoints', 'polarity', 'stress', 'O', 'C', 'E', 'A', 'N'],

                # PARA Y DAYS FROM DEV TO TEST, COM C, A e N
                # ['storyPoints', 'polarity_mean', 'C', 'A', 'N'],
                # ['storyPoints', 'polarity_mean', 'stress_mean', 'C', 'A', 'N'],
                # ['storyPoints', 'polarity', 'C', 'A', 'N'],
                # ['storyPoints', 'polarity', 'stress', 'C', 'A', 'N'],

                # PARA Y DAYS FROM DEV TO TEST, SOMENTE COM N
                # ['storyPoints', 'polarity_mean', 'N'],
                # ['storyPoints', 'polarity_mean', 'stress_mean', 'N'],
                # ['storyPoints', 'polarity', 'N'],
                # ['storyPoints', 'polarity', 'stress', 'N'],

                # PARA Y DAYS FROM DEV TO TEST, SOMENTE COM A
                # ['storyPoints', 'polarity_mean', 'A'],
                # ['storyPoints', 'polarity_mean', 'stress_mean', 'A'],
                # ['storyPoints', 'polarity', 'A'],
                # ['storyPoints', 'polarity', 'stress', 'A'],

                # PARA Y DAYS FROM DEV TO TEST, SOMENTE COM C
                ['storyPoints', 'polarity_mean', 'C'],
                ['storyPoints', 'polarity_mean', 'stress_mean', 'C'],
                ['storyPoints', 'polarity', 'C'],
                ['storyPoints', 'polarity', 'stress', 'C'],
>>>>>>> a2fd7ef7a398ba105ed3e3d2b8a8421eb102d722

                # PARA Y STORY POINTS
                # ['polarity_mean', 'O', 'C', 'E', 'A', 'N'],
                # ['polarity_mean', 'stress_mean', 'O', 'C', 'E', 'A', 'N'],
                # ['polarity', 'O', 'C', 'E', 'A', 'N'],
                # ['polarity', 'stress', 'O', 'C', 'E', 'A', 'N']
            ],
            # "ngram": [(1,2)],
            # "balance": ['over', 'under', 'mixed', 'smote'],
            # "balance": [''], # NAO ESTA EXECUTANDO O SMOTE, ISTO EH APENAS PARA SALVAR NO ARQUIVO DE RESULTADO
            # "percentage_features": [5, 10], #, 40, 50, 60, 70, 80, 90, 95],
            # "feature_selection": ['kbest'], #, 'percentile'],
            "folds": [10],
            "datasets": [
                {
                    'train': 'dataset_survey_tasks_train_1_5_5',
                    'val': 'dataset_survey_tasks_test_1_5_5'
                },
                {
                    'train': 'dataset_survey_tasks_train_2_5_5',
                    'val': 'dataset_survey_tasks_test_2_5_5'
                }
            ],
            "smote_instances": [
<<<<<<< HEAD
                # 1000,
                # 3000,
=======
                250,
                500,
                1000,
                3000,
>>>>>>> a2fd7ef7a398ba105ed3e3d2b8a8421eb102d722
                5000
            ],
            "scales": [
                'no_scaled',
                'MinMaxScaler',
                'StandardScaler'
            ]
        }


def get_result_map():
    result_map = {
        "Algorithm": [],
        "Train Accuracy": [],
        "Train Precision": [],
        "Train Recall": [],
        "Train F1 Score": [],
        "Test Accuracy": [],
        "Test Precision": [],
        "Test Recall": [],
        "Test F1 Score": [],
        "Val. Accuracy": [],
        "Val. Precision": [],
        "Val. Recall": [],
        "Val. F1 Score": [],
        "AUC": [],
        # "Ngram": [],
        # "Vect. Strategy": [],
        "Bal. Strategy": [],
        "SMOTE Instances": [],
        "Scale": [],
        "Features": [],
        "Folds": [],
        # "Feat. Selec. Strategy": [],
        # "Hyper Params.": [],
        "Model": []
    }
    return result_map


def get_default_scoring():
    scoring = {
        'roc_auc': me.make_scorer(me.roc_auc_score, needs_proba=True, multi_class='ovr'),
        'accuracy': me.make_scorer(me.accuracy_score),
        'precision': me.make_scorer(me.precision_score, average='weighted'),
        'recall': me.make_scorer(me.recall_score, average='weighted'),
        'f1_score': me.make_scorer(me.f1_score, average='macro'),
    }
    return scoring
