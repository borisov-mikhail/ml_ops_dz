import pandas as pd
from catboost import CatBoostClassifier


def import_model():
    
    model = CatBoostClassifier()
    model.load_model('./models/main_model.cbm')
    return model


def make_pred(dt, path_to_file):
    
    print('Importing pretrained model...')
    model = import_model()

    # Define optimal threshold
    model_th = 0.49

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': (model.predict_proba(dt)[:, 1] > model_th) * 1
    })
    print('Prediction complete!')

    # Return proba for positive class
    return submission