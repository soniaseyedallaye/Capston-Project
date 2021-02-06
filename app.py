import json
import pickle
import datetime
import dateutil
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
import re
from playhouse.shortcuts import model_to_dict
########################################
# Begin database stuff
DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    #proba = FloatField()
    prediction = TextField()
    outcome = IntegerField(null=True)

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

def check_valid_column(observation):
    """
        Validates that our observation only has valid columns

        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """

    valid_columns = {"observation_id","Type", "Date","Part of a policing operation", "Latitude", "Longitude", "Gender", "Legislation", "Object of search", "Age range", "Officer-defined ethnicity", "station"}

    keys = set(observation.keys())

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
        "observation_id" : object,
        "Type": object,
        "Date" : object,
        "Part of a policing operation": bool,
        "Latitude": float,
        "Longitude": float,
        "Gender": object,
        "Legislation": object,
        "Object of search": object,
        "Age range": object,
        "Officer-defined ethnicity": object,
        "station": object,
    }


    for col, type_ in column_types.items():
        if not isinstance(observation[col], type_):
            error = "Field {} is {}, while it should be {}".format(col, type(observation[col]), type_)
            return False, error
    return True, ""


#def check_numerical_values(observation):

    #valid_range_map = {"hour": list(range(0, 24)),"month": list(range(1, 13))}

    #for key, item in valid_range_map.items():
     #if key in observation['observation']:
        #value = observation['observation'][key]
        #if value not in item:
           # error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
            #    key, value, ",".join(["'{}'".format(v) for v in item]))
            #return False, error
     #elif key not in observation['observation']:
      #  error = "{} is not in the observation".format(key)
       # return False, error

    #return True, ""


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
                        'Deer Act 1991 (section 12)'
                        ,'Other'],
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
                             'Seals or hunting equipment','Other'],
        "Age range": ['18-24', '10-17', '25-34', 'over 34', 'under 10'],

        "Officer-defined ethnicity": ['White', 'Black', 'Asian', 'Other', 'Mixed'],
        #"Removal of more than just outer clothing": [True, False],
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
                    'gwent', 'Other'],

        #"day_of_week": ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    }

    for key, valid_categories in valid_category_map.items():
        value = observation[key]
        if value not in valid_categories:
            error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
            return False, error

    return True, ""


def check_latitude(observation):
    latitude = observation.get("Latitude")

    if not latitude:
        error = "Field `Latitude` missing"
        return False, error

    if latitude < 49 or latitude > 58:
        error = "Field `Latitude` is not between 49 and 58"
        return False, error

    return True, ""


def check_longitude(observation):
    longitude = observation.get("Longitude")

    if not longitude:
        error = "Field `Longitude` missing"
        return False, error

    if longitude < -9 or longitude > 2:
        error = "Field `Longitude` is not between -9 and 2"
        return False, error

    return True, ""
def getDateTimeFromISO8601String(s):
    d = dateutil.parser.isoparse(s)
    return d
regex = r'^(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]+)?(Z|[+-](?:2[0-3]|[01][0-9]):[0-5][0-9])?$'
match_iso8601 = re.compile(regex).match
def check_date(observation):
    date = observation.get("Date")
    try:
        if match_iso8601(date) is not None:
            return True,""
    except ValueError:
        pass
        error= "ERROR: Date '{}' is not in correct ISO8601String format".format(date)
        return False,error


# End input validation functions
########################################
########################################
# Begin webserver stuff
app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def should_search():
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

    valid_date_ok, error = check_date(obs_dict)
    if not valid_date_ok:
        response = {'error': error}
        return jsonify(response)

    # read data
    _id = obs_dict['observation_id']
    obs_dict.pop('observation_id')
    _iso = obs_dict['Date']
    obs_dict.pop('Date')
    date_iso = getDateTimeFromISO8601String(_iso)
    hour = date_iso.hour
    day = date_iso.day
    month = date_iso.month
    obs_dict['hour'] = hour
    obs_dict['month'] = month
    obs_dict['day'] = day


    obs = pd.DataFrame([obs_dict], columns=columns).astype(dtypes)

    # compute prediction
    #proba = pipeline.predict_proba(obs)[0,1]
    prediction = pipeline.predict(obs)[0]
    response = {'outcome': bool(prediction)}

    p = Prediction(
        observation_id=_id,
        observation=request.data,
        #proba=proba,
        prediction=prediction,)
    try:
       p.save()
       return jsonify(response)
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        #response["error"] = error_msg
        #print(error_msg)
        DB.rollback()
        #{'error': error_msg}
        return jsonify({'error': error_msg})



@app.route('/search_result/', methods=['POST'])
def search_result():
    obs_dict = request.get_json()

    try:
        p = Prediction.get(Prediction.observation_id == obs_dict['observation_id'])
        p.outcome = obs_dict['outcome']
        #p.outcome = Prediction.prediction
        #response = obs_dict
        p.save()
        #obs_dict['outcome'] =p
        obs_dict['predicted_outcome'] = p.prediction
        return jsonify(obs_dict)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs_dict['observation_id'])
        return jsonify({'error': error_msg})
if __name__ == "__main__":
    app.run()
