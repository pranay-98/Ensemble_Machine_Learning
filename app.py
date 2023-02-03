#!/usr/bin/env python3

from flask import Flask,request 
import pandas as pd
import pickle
import numpy as np
from pandas.io.json import json_normalize

app = Flask(__name__) 


@app.route('/load_model',methods=['POST'])
def load_model():
    req_data = request.get_json()
    test_data_subset = pd.DataFrame.from_dict(json_normalize(req_data), orient='columns')
    
    
    columns_to_drop=pd.read_csv("/model/columns_to_drop.csv")
    
    
    columns_to_Retain = set(test_data_subset.columns.values) - set(columns_to_drop.colnames.values)
    test_data_selected_columns = test_data_subset[columns_to_Retain]
    
    #select the categorical columns from the dataframe
    column_datatypes = test_data_selected_columns.dtypes
    categorical_columns = list(column_datatypes[column_datatypes=="object"].index.values)
    
    
    for cf1 in categorical_columns:
        filename = "/model/"+cf1+".sav"
        le = pickle.load(open(filename, 'rb'))
        
       
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        test_data_selected_columns[cf1]=test_data_selected_columns[cf1].apply(lambda x: le_dict.get(x, -1))
    
    test_data_id = test_data_selected_columns['id']
    test_data_selected_columns = test_data_selected_columns.drop('id',axis=1)

       
    Column_datatypes= test_data_selected_columns.dtypes
    Integer_columns = list(Column_datatypes.where(lambda x: x =="int64").dropna().index.values)
    
    
    test_data_selected_columns[Integer_columns] = test_data_selected_columns[Integer_columns].astype('category',copy=False)
    
    
    tuned_model = pickle.load(open("/model/tunedmodel_rf", 'rb'))
    Y_test_predict = tuned_model.predict(test_data_selected_columns)
    
    
    output = pd.DataFrame()
    output['id']=test_data_id
    output['predict_loss']=Y_test_predict
    
    output=output.to_json(orient='records')    
    return output

if __name__ =='__main__':
    app.run(debug=True,port=4000)