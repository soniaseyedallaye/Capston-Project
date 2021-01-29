import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict

########################################
# Begin database stuff
DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    label = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)
# End database stuff
########################################
########################################
# Unpickle the previously-trained model
with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################
########################################
# Input validation functions

def check_request(request):
    """
        Validates that our request is well formatted

        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """

    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        return False, error

    if "observation" not in request:
        error = "Field `observation` missing from request: {}".format(request)
        return False, error

    return True, ""


def check_valid_column(observation):
    """
        Validates that our observation only has valid columns

        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """

    valid_columns = {"Type", "Part of a policing operation", "Latitude", "Longitude", "Gender", "Legislation", "Object of search", "Age range", "Officer-defined ethnicity", "Removal of more than just outer clothing", "station", "hour", "month", "day_of_week"}
    keys = set(observation['observation'].keys())

    if len(valid_columns - keys) > 0:
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error

    if len(keys - valid_columns) > 0:
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error

    if len(keys - valid_columns) == 0:
        return True, ""


def check_column_types(observation):
    column_types = {
        "Type": object,
        "Part of a policing operation": bool,
        "Latitude": float,
        "Longitude": float,
        "Gender": object,
        "Legislation": object,
        "Object of search": object,
        "Age range": object,
        "Officer-defined ethnicity": object,
        "Removal of more than just outer clothing": bool,
        "station": object,
        "hour": int,
        "month": int,
        "day_of_week": object,
    }

    for col, type_ in column_types.items():
        if not isinstance(observation['observation'][col], type_):
            error = "Field {} is {}, while it should be {}".format(col, type(observation['observation'][col]), type_)
            return False, error
    return True, ""


def check_numerical_values(observation):

    valid_range_map = {"hour": list(range(0, 24)),"month": list(range(1, 13))}

    for key, item in valid_range_map.items():
     if key in observation['observation']:
        value = observation['observation'][key]
        if value not in item:
            error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                key, value, ",".join(["'{}'".format(v) for v in item]))
            return False, error
     elif key not in observation['observation']:
        error = "{} is not in the observation".format(key)
        return False, error

    return True, ""


def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """

    valid_category_map = {
        "Type": ['Person search', 'Person Vehicle search', 'Vehicle search'],
        "Part of a policing operation": [True, False],
        "Gender": ['Male', 'Female'],
        "Legislation": ['Misuse of Drugs Act 1971 (section 23)',
                        'Police and Criminal Evidence Act 1984 (section 1)',
                        'Criminal Justice and Public Order Act 1994 (section 60)',
                        'Firearms Act 1968 (section 47)',
                        'Criminal Justice Act 1988 (section 139B)',
                        'Poaching Prevention Act 1862 (section 2)',
                        'Psychoactive Substances Act 2016 (s36(2))',
                        'Wildlife and Countryside Act 1981 (section 19)',
                        'Police and Criminal Evidence Act 1984 (section 6)',
                        'Aviation Security Act 1982 (section 27(1))',
                        'Customs and Excise Management Act 1979 (section 163)',
                        'Crossbows Act 1987 (section 4)',
                        'Psychoactive Substances Act 2016 (s37(2))',
                        'Protection of Badgers Act 1992 (section 11)',
                        'Public Stores Act 1875 (section 6)',
                        'Conservation of Seals Act 1970 (section 4)',
                        'Deer Act 1991 (section 12)'],
        "Object of search": ['Controlled drugs',
                             'Offensive weapons',
                             'Stolen goods',
                             'Article for use in theft',
                             'Evidence of offences under the Act',
                             'Anything to threaten or harm anyone',
                             'Articles for use in criminal damage',
                             'Firearms',
                             'Fireworks',
                             'Psychoactive substances',
                             'Detailed object of search unavailable',
                             'Game or poaching equipment',
                             'Evidence of wildlife offences',
                             'Goods on which duty has not been paid etc.',
                             'Crossbows',
                             'Seals or hunting equipment'],
        "Age range": ['18-24', '10-17', '25-34', 'over 34', 'under 10'],

        "Officer-defined ethnicity": ['White', 'Black', 'Asian', 'Other', 'Mixed'],
        "Removal of more than just outer clothing": [True, False],
        "station": ['merseyside',
                    'essex',
                    'thames-valley',
                    'west-yorkshire',
                    'hampshire',
                    'hertfordshire',
                    'kent',
                    'south-yorkshire',
                    'surrey',
                    'avon-and-somerset',
                    'btp',
                    'lancashire',
                    'west-mercia',
                    'devon-and-cornwall',
                    'staffordshire',
                    'nottinghamshire',
                    'northumbria',
                    'sussex',
                    'north-wales',
                    'lincolnshire',
                    'leicestershire',
                    'greater-manchester',
                    'cheshire',
                    'norfolk',
                    'dyfed-powys',
                    'bedfordshire',
                    'humberside',
                    'city-of-london',
                    'northamptonshire',
                    'suffolk',
                    'warwickshire',
                    'gloucestershire',
                    'derbyshire',
                    'dorset',
                    'durham',
                    'north-yorkshire',
                    'cumbria',
                    'cleveland',
                    'wiltshire',
                    'cambridgeshire',
                    'gwent'],

        "day_of_week": ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    }

    for key, valid_categories in valid_category_map.items():
        value = observation['observation'][key]
        if value not in valid_categories:
            error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
            return False, error

    return True, ""


def check_latitude(observation):
    latitude = observation['observation'].get("Latitude")

    if not latitude:
        error = "Field `Latitude` missing"
        return False, error

    if latitude < 49 or latitude > 58:
        error = "Field `Latitude` is not between 49 and 58"
        return False, error

    return True, ""


def check_longitude(observation):
    longitude = observation['observation'].get("Longitude")

    if not longitude:
        error = "Field `Longitude` missing"
        return False, error

    if longitude < -9 or longitude > 2:
        error = "Field `Longitude` is not between -9 and 2"
        return False, error

    return True, ""


# End input validation functions
########################################
########################################
# Begin webserver stuff
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()

    # verification routines
    valid_columns_ok, error = check_valid_column(obs_dict)
    if not valid_columns_ok:
        response = {'error': error}
        return jsonify(response)

    valid_types_ok, error = check_column_types(obs_dict)
    if not valid_types_ok:
        response = {'error': error}
        return jsonify(response)

    valid_numerical_ok, error = check_numerical_values(obs_dict)
    if not valid_numerical_ok:
        response = {'error': error}
        return jsonify(response)

    valid_categorical_ok, error = check_categorical_values(obs_dict)
    if not valid_categorical_ok:
        response = {'error': error}
        return jsonify(response)

    valid_latitude_ok, error = check_latitude(obs_dict)
    if not valid_latitude_ok:
        response = {'error': error}
        return jsonify(response)

    valid_longitude_ok, error = check_longitude(obs_dict)
    if not valid_longitude_ok:
        response = {'error': error}
        return jsonify(response)

    # read data
    _id = obs_dict['observation_id']
    obs = pd.DataFrame([obs_dict['observation']], columns=columns).astype(dtypes)

    # compute prediction
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'observation_id': _id, 'proba': proba, 'prediction': bool(prediction)}

    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs_dict = request.get_json()

    try:
        p = Prediction.get(Prediction.observation_id == obs_dict['observation_id'])
        p.label = obs_dict['label']
        p.save()

        response = obs_dict

        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs_dict['observation_'
                                                                          'id'])
        return jsonify({'error': error_msg})


if __name__ == "__main__":
    app.run()
