import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc # garbage collection

import os
# print(os.listdir("../input"))

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)

### load data ############################################
train = pd.read_csv('analysisData.csv') 
test = pd.read_csv('scoringData.csv') 

### data statistics ############################################
def data_stat(data):
    types = pd.Series({c:str(data[c].dtype) for c in data.columns}) # variable type
    total = data.isnull().sum() # the number of null values
    percent = 100 * data.isnull().sum() / data.isnull().count() # the percentage of null values
    n_unique = data.nunique() # the number of unique values
    
    tt = np.transpose(pd.concat([types, total, percent, n_unique], axis=1, keys=['Types', '# of nulls', '% of nulls', '# of uniques']))
    return pd.concat([data.head(), tt], axis = 0)

stat_train = data_stat(train)
stat_test = data_stat(test)
stat_train

### feature engineering ###############################################
droplist_torf = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'is_location_exact', 'has_availability',
                 'requires_license', 'instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture', 
                 'require_guest_phone_verification'] # true / false
droplist_ca01 = ['host_response_time'] # categorical
droplist_ca02 = ['host_response_rate'] # categorical
droplist_ca03 = ['calendar_updated'] # categorical
droplist_ca04 = ['room_type', 'bed_type', 'cancellation_policy', 'property_type'] # categorical
droplist_date = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review'] # datetime
droplist_lo01 = ['neighbourhood_group_cleansed', 'neighbourhood_cleansed'] # location
droplist_lo02 = ['zipcode'] # location
droplist_it01 = ['host_verifications'] # list of items
droplist_it02 = ['amenities'] # list of items
droplist_tx01 = ['name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction',
                 'house_rules', 'host_name', 'host_about'] # text processing

droplist_processed = (droplist_torf
                      + droplist_ca01 + droplist_ca02 + droplist_ca03 + droplist_ca04
                      + droplist_date 
                      + droplist_lo01 + droplist_lo02
                      + droplist_it01 + droplist_it02
                      + droplist_tx01)

def feature_engineering(data):
    prefix_new_vari = 'm_'
    
    # true / false
    for col in droplist_torf: 
        data[prefix_new_vari+col] = data[col].map({'t':1, 'f':0})
        
    # categorical
    for col in droplist_ca01: 
        data[prefix_new_vari+col] = data[col].map({'within an hour':4, 'within a few hours':3, 'within a day':2, 'a few days or more':1})
    for col in droplist_ca02: 
        data[prefix_new_vari+col] = data[col].str.extract(r'(\d+)%', expand=False).astype(float)
    for col in droplist_ca03: 
        nn = data[col].str.extract(r'(\d+) [a-zA-Z]+', expand = False).astype(float).fillna(0)
        aa = data[col].str.contains(r'a ') * 1.0
        yy = data[col].str.contains(r'yesterday') * 1.0
        dd = data[col].str.contains(r'day') * 1
        ww = data[col].str.contains(r' week') * 7
        mm = data[col].str.contains(r' month') * (365/12.0)
        vv = data[col].str.contains(r'never') * 2500.0
        data[prefix_new_vari+col] = (nn + aa + yy) * (dd + ww + mm) + vv
    for col in droplist_ca04: 
        data[prefix_new_vari+col] = data[col].map({v:i+1 for (i,v) in enumerate(train[[col]+target].groupby(col).mean().sort_values(by = target).index)})
    
    # date / time
    for col in droplist_date: 
        dtime = pd.to_datetime(data[col])
        dd = dtime.dt.year
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_year'] = dd
        dd = dtime.dt.month
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_month'] = dd
        dd = dtime.dt.day
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_day'] = dd
        dd = dtime.dt.hour
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_hour'] = dd
        dd = dtime.dt.weekofyear
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_weekofyear'] = dd
        dd = dtime.dt.weekday
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_weekday'] = dd
        dd = (dtime.dt.weekday >=5).astype(int)
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_weekend'] = dd
            
    # location
    for col in droplist_lo01: 
        data[prefix_new_vari+col] = data[col].map({v:i+1 for (i,v) in enumerate(train[[col]+target].groupby(col).mean().sort_values(by = target).index)})
    for col in droplist_lo02: 
        if str(data[col].dtype) == 'object':
            data[prefix_new_vari+col] = data[col].str.extract(r'(\d+)', expand = False).astype(float)
            data.loc[data[col] == '10003-8623', prefix_new_vari+col] = 10003
            data.loc[data[col] == '11249\r\n11249', prefix_new_vari+col] = 11249
            data.loc[data[col] == '11103-3233', prefix_new_vari+col] = 11103
            data.loc[data[col] == '11413-3220', prefix_new_vari+col] = 11413
            data.loc[data[col] == '1m', prefix_new_vari+col] = np.nan
        else:
            data[prefix_new_vari+col] = data[col].copy()
        
    # items
    for col in droplist_it01: 
        tt = data[col].str.replace('[\[\]\' ]','').str.get_dummies(',') # https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
        data[(prefix_new_vari + col + '_') + tt.columns] = tt
        data[prefix_new_vari + col + '_num'] = tt.sum(axis = 1)
    for col in droplist_it02: 
        tt = data[col].str.replace(', toilet',' toilet').str.replace('[{}"]','').str.get_dummies(',') # https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
        data[(prefix_new_vari + col + '_') + tt.columns] = tt
        data[prefix_new_vari + col + '_num'] = tt.sum(axis = 1)
        
    # text
    for col in droplist_tx01:
        dd = data[col].str.len() # number of characters
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_nchar'] = dd
        dd = data[col].str.count('\\S+') # number of words
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_nword'] = dd
#         dd = data[col].str.count("[A-Za-z,;'\"\\s]+[^.!?]*[.?!]") # number of sentences TOO SLOW !!!
#         if len(dd.unique()) > 1: data[prefix_new_vari+col+'_nsntc'] = dd
        dd = data[col].str.count('\.') # number of words
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_nperd'] = dd
        dd = data[col].str.count(',') # number of words
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_ncmma'] = dd
        dd = data[col].str.count('!') # number of words
        if len(dd.unique()) > 1: data[prefix_new_vari+col+'_nexcl'] = dd

    return data

target = ['price']

wdata = feature_engineering( pd.concat([train,test], axis=0, sort=False, ignore_index = True) )
            
train = wdata.loc[range(0, train.shape[0])]
test = wdata.loc[range(train.shape[0], train.shape[0] + test.shape[0])]

del wdata
gc.collect()

print('train shape =', train.shape)
print('test shape =', test.shape)

#-----------------------------------------------------------------------
droplist03 = ['host_location', 'host_neighbourhood', 'neighbourhood', 'city'] # location

# droplist1 = list(stat_train.columns[stat_train.loc['Types'] == 'object'])
droplist2 = list(stat_train.columns[stat_train.loc['% of nulls'] > 98])
droplist00 = ['listing_url', 'experiences_offered', 'picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'jurisdiction_names', # garbage variable
              'street', 'smart_location', # location: redundant
              'country_code', 'country', 'market', 'state' # location: almost uniform data values
             ] 
droplist = droplist00 + droplist_processed + droplist2 + droplist03   

# features = [col for col in train.columns if col not in droplist]
features = [col for col in train.columns if col not in droplist + target]
print('# of features =',len(features))
# print(features)

    
### XGBoost ##################################################################
eval_metric = 'rmse'; MIN_MAX = 'min'; mlmodel = XGBRegressor(learning_rate = 0.1, n_jobs = 4, seed = 123) # regression

mlmodel = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.30000000000000004, gamma=0.0,
       importance_type='gain', learning_rate=0.007167896137269041, max_delta_step=0,
       max_depth=5, min_child_weight=1.7782794100389228, missing=None,
       n_estimators=824, n_jobs=4, nthread=None, objective='reg:linear',
       random_state=0, reg_alpha=421.6965034285823,
       reg_lambda=1.1291051154657759, scale_pos_weight=1, seed=123,
       silent=True, subsample=0.9375)

# cross validation 1: XGBoost
def user_defined_eval_function(train, test, features, target, mlmodel, eval_metric, MIN_MAX, predict_test_output = False):
    dtrain = xgb.DMatrix(train[features], label = train[target], missing = np.nan) # missing value handling: https://www.youtube.com/watch?v=cVqDguNWh4M
    cvoutp = xgb.cv(mlmodel.get_xgb_params(), dtrain, num_boost_round = 10000, verbose_eval =  False, 
                      nfold = 5, metrics = eval_metric, early_stopping_rounds = 50) # early_stopping_rounds
    mlmodel.set_params(n_estimators = cvoutp.shape[0]) # update n_estimator 
    train_score = cvoutp.tail(1)[cvoutp.columns[cvoutp.columns.str.contains('train-.+-mean', regex=True)]].squeeze()
    valid_score = cvoutp.tail(1)[cvoutp.columns[cvoutp.columns.str.contains('test-.+-mean', regex=True)]].squeeze()

    if predict_test_output == True:
        mlmodel.fit(train[features], train[target].values.ravel(), eval_metric = eval_metric) #Fit the algorithm on the data
        test_pred = mlmodel.predict(test[features])    
    else: 
        test_pred = []
    return test_pred, valid_score    

test_pred, valid_score = user_defined_eval_function(train, test, features, target, mlmodel, eval_metric, MIN_MAX, predict_test_output = True)
print('final validatoin score = ',valid_score)
print(mlmodel) # final model confirmation


### submission ############################################################
sub = pd.DataFrame({"id": test['id']})
sub["price"] = test_pred
sub.to_csv('submission.csv', index=False)

