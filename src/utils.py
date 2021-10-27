import os
import glob
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from arch import arch_model

import warnings
warnings.filterwarnings('ignore')


# Data processing routines. Taken mainly from https://www.kaggle.com/mayangrui/lgbm-ffnn


# directory
data_dir = 'data/'
models_dir = 'models/'

# Function to calculate first WAP
def calc_wap1(df):
	wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
	return wap

# Function to calculate second WAP
def calc_wap2(df):
	wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
	return wap

def calc_wap3(df):
	wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
	return wap

def calc_wap4(df):
	wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
	return wap

# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
	return np.log(series).diff()

# Calculate the realized volatility
def realized_volatility(series):
	return np.sqrt(np.sum(series**2))

# Function to count unique elements of a series
def count_unique(series):
	return len(np.unique(series))

# Calculate the realized skew
def realized_skew(series):
	return np.sqrt(series.count())*np.sum(series**3)/(realized_volatility(series)**3)

# Calculate the realized kurtosis
def realized_kurtosis(series):
	return series.count()*np.sum(series**4)/(realized_volatility(series)**4)

# Calculate integrated quarticity
def realized_quarticity(series):
	return (series.count()/3)*np.sum(series**4)

# Calculate order book depth
def calc_depth(df):
	depth = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1'] + df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']
	return depth

# Calculate order book slope
def calc_slope(df):
	v0 = (df['bid_size1']+df['ask_size1'])/2
	p0 = (df['bid_price1']+df['ask_price1'])/2
	slope_bid = ((df['bid_size1']/v0)-1)/abs((df['bid_price1']/p0)-1)+(
				(df['bid_size2']/df['bid_size1'])-1)/abs((df['bid_price2']/df['bid_price1'])-1)
	slope_ask = ((df['ask_size1']/v0)-1)/abs((df['ask_price1']/p0)-1)+(
				(df['ask_size2']/df['ask_size1'])-1)/abs((df['ask_price2']/df['ask_price1'])-1)
	return (slope_bid+slope_ask)/2, abs(slope_bid-slope_ask)


# Function to calculate the GARCH volatility estimation
df_sec = pd.DataFrame()
df_sec['seconds'] = np.arange(0,601)

def garch_est(df):
	df_aux = pd.merge(df, df_sec, left_on='seconds_in_bucket', right_on='seconds', how='right')
	df_aux = df_aux.ffill()
	df_aux = df_aux.fillna(0)
	df_aux['cum_vol'] = df_aux.log_return1.pow(2).cumsum().pow(0.5)

	garch = arch_model(df_aux.cum_vol, mean='Zero', vol='GARCH', p=10, q=10)
	garch_fit = garch.fit(show_warning=False, disp='off')
	sigma_hat = garch_fit.forecast(horizon=10)
	garch_estimation = np.sqrt(sigma_hat.variance.values[-1][-1])

	df['garch_estimation'] = garch_estimation
	#ans = pd.Series(garch_estimation, index=df.index)
	return df

# Function to read our base train and test set
def read_train_test():
	train = pd.read_csv(data_dir + 'train.csv')
	test = pd.read_csv(data_dir + 'test.csv')
	# Create a key to merge with book and trade data
	train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
	test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
	print(f'Our training set has {train.shape[0]} rows')
	return train, test

# Function to preprocess book data (for each stock id)
def book_preprocessor(file_path):
	df = pd.read_parquet(file_path)
	# Calculate Wap
	df['wap1'] = calc_wap1(df)
	df['wap2'] = calc_wap2(df)
	df['wap3'] = calc_wap3(df)
	df['wap4'] = calc_wap4(df)
	df['depth'] = calc_depth(df)
	#df = df.groupby(['time_id']).apply(garch_est)
	# Calculate squared difference in time
	df['time_diff'] = df.groupby(['time_id'])['seconds_in_bucket'].diff()
	df.time_diff = df.time_diff.fillna(1)
	df['updates'] = df.groupby(['time_id'])['seconds_in_bucket'].transform('count')
	# Calculate log returns
	df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
	df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
	df['log_return3'] = df.groupby(['time_id'])['wap3'].apply(log_return)
	df['log_return4'] = df.groupby(['time_id'])['wap4'].apply(log_return)
	# Calculate wap balance
	df['wap_balance'] = abs(df['wap1'] - df['wap2'])
	# Calculate spread
	df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
	df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
	df['bid_spread'] = df['bid_price1'] - df['bid_price2']
	df['ask_spread'] = df['ask_price1'] - df['ask_price2']
	df["bid_ask_spread"] = abs(df['bid_spread'] - df['ask_spread'])
	df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
	df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
	df['instant_volatility'] = df['log_return1'] / (df['time_diff'].pow(0.5))
	
	# Dict for aggregations
	create_feature_dict = {
		'wap1': [np.sum, np.mean, np.std],
		'wap2': [np.sum, np.mean, np.std],
		'wap3': [np.sum, np.mean, np.std],
		'wap4': [np.sum, np.mean, np.std],
		'depth': [np.sum, np.mean, np.std],
		'log_return1': [np.sum, realized_volatility, realized_skew, realized_skew, realized_kurtosis, realized_quarticity, np.mean, np.std],
		'log_return2': [np.sum, realized_volatility, realized_skew, realized_skew, realized_kurtosis, realized_quarticity, np.mean, np.std],
		'log_return3': [np.sum, realized_volatility, realized_skew, realized_skew, realized_kurtosis, realized_quarticity, np.mean, np.std],
		'log_return4': [np.sum, realized_volatility, realized_skew, realized_skew, realized_kurtosis, realized_quarticity, np.mean, np.std],
		'wap_balance': [np.sum, np.mean, np.std],
		'price_spread':[np.sum, np.mean, np.std],
		'price_spread2':[np.sum, np.mean, np.std],
		'bid_spread':[np.sum, np.mean, np.std],
		'ask_spread':[np.sum, np.mean, np.std],
		'total_volume':[np.sum, np.mean, np.std],
		'volume_imbalance':[np.sum, np.mean, np.std],
		"bid_ask_spread":[np.sum, np.mean, np.std],
		'instant_volatility': [realized_volatility],
	}
	create_feature_dict_time = {
		'log_return1': [realized_volatility,realized_skew, realized_skew, realized_kurtosis],
		'log_return2': [realized_volatility,realized_skew, realized_skew, realized_kurtosis],
		'log_return3': [realized_volatility,realized_skew, realized_skew, realized_kurtosis],
		'log_return4': [realized_volatility,realized_skew, realized_skew, realized_kurtosis],
	}
	
	# Function to get group stats for different windows (seconds in bucket)
	def get_stats_window(fe_dict, seconds_in_bucket, add_suffix = False):
		# Group by the window
		df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
		# Rename columns joining suffix
		df_feature.columns = ['_'.join(col) for col in df_feature.columns]
		# Add a suffix to differentiate windows
		if add_suffix:
			df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
		return df_feature

	# Get the stats for different windows
	df_feature = get_stats_window(create_feature_dict,seconds_in_bucket = 0, add_suffix = False)	
	df_feature_450 = get_stats_window(create_feature_dict_time, seconds_in_bucket = 450, add_suffix = True)
	df_feature_300 = get_stats_window(create_feature_dict_time, seconds_in_bucket = 300, add_suffix = True)	
	df_feature_150 = get_stats_window(create_feature_dict_time, seconds_in_bucket = 150, add_suffix = True)

	# Merge all
	df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')	
	df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')	
	df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
	# Drop unnecesary time_ids
	df_feature.drop(['time_id__450','time_id__300', 'time_id__150'], axis = 1, inplace = True)

	# Create row_id so we can merge
	stock_id = file_path.split('=')[1]
	df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
	df_feature.drop(['time_id_'], axis = 1, inplace = True)
	return df_feature

# Function to preprocess trade data (for each stock id)
def trade_preprocessor(file_path):
	df = pd.read_parquet(file_path)
	df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
	df['amount']=df['price']*df['size']
	# Dict for aggregations
	create_feature_dict = {
		'log_return':[realized_volatility],
		'seconds_in_bucket':[count_unique],
		'size':[np.sum, np.max, np.min],
		'order_count':[np.sum,np.max],
		'amount':[np.sum,np.max,np.min],
	}
	create_feature_dict_time = {
		'log_return':[realized_volatility],
		'seconds_in_bucket':[count_unique],
		'size':[np.sum],
		'order_count':[np.sum],
	}
	# Function to get group stats for different windows (seconds in bucket)
	def get_stats_window(fe_dict,seconds_in_bucket, add_suffix = False):
		# Group by the window
		df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
		# Rename columns joining suffix
		df_feature.columns = ['_'.join(col) for col in df_feature.columns]
		# Add a suffix to differentiate windows
		if add_suffix:
			df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
		return df_feature
	

	# Get the stats for different windows
	df_feature = get_stats_window(create_feature_dict,seconds_in_bucket = 0, add_suffix = False)
	df_feature_500 = get_stats_window(create_feature_dict_time, seconds_in_bucket = 500, add_suffix = True)
	df_feature_400 = get_stats_window(create_feature_dict_time, seconds_in_bucket = 400, add_suffix = True)
	df_feature_300 = get_stats_window(create_feature_dict_time, seconds_in_bucket = 300, add_suffix = True)
	df_feature_200 = get_stats_window(create_feature_dict_time, seconds_in_bucket = 200, add_suffix = True)
	df_feature_100 = get_stats_window(create_feature_dict_time, seconds_in_bucket = 100, add_suffix = True)
	
	def tendency(price, vol):    
		df_diff = np.diff(price)
		val = (df_diff/price[1:])*100
		power = np.sum(val*vol[1:])
		return(power)
	
	lis = []
	for n_time_id in df['time_id'].unique():
		df_id = df[df['time_id'] == n_time_id]        
		tendencyV = tendency(df_id['price'].values, df_id['size'].values)      
		f_max = np.sum(df_id['price'].values > np.mean(df_id['price'].values))
		f_min = np.sum(df_id['price'].values < np.mean(df_id['price'].values))
		df_max =  np.sum(np.diff(df_id['price'].values) > 0)
		df_min =  np.sum(np.diff(df_id['price'].values) < 0)
		# new
		abs_diff = np.median(np.abs( df_id['price'].values - np.mean(df_id['price'].values)))        
		energy = np.mean(df_id['price'].values**2)
		iqr_p = np.percentile(df_id['price'].values,75) - np.percentile(df_id['price'].values,25)
		
		# vol vars
		
		abs_diff_v = np.median(np.abs( df_id['size'].values - np.mean(df_id['size'].values)))        
		energy_v = np.sum(df_id['size'].values**2)
		iqr_p_v = np.percentile(df_id['size'].values,75) - np.percentile(df_id['size'].values,25)
		
		lis.append({'time_id':n_time_id,'tendency':tendencyV,'f_max':f_max,'f_min':f_min,'df_max':df_max,'df_min':df_min,
				   'abs_diff':abs_diff,'energy':energy,'iqr_p':iqr_p,'abs_diff_v':abs_diff_v,'energy_v':energy_v,'iqr_p_v':iqr_p_v})
	
	df_lr = pd.DataFrame(lis)
		
   
	df_feature = df_feature.merge(df_lr, how = 'left', left_on = 'time_id_', right_on = 'time_id')
	
	# Merge all
	df_feature = df_feature.merge(df_feature_500, how = 'left', left_on = 'time_id_', right_on = 'time_id__500')
	df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')
	df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
	df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')
	df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
	# Drop unnecesary time_ids
	df_feature.drop(['time_id__500','time_id__400', 'time_id__300', 'time_id__200','time_id','time_id__100'], axis = 1, inplace = True)
	
	
	df_feature = df_feature.add_prefix('trade_')
	stock_id = file_path.split('=')[1]
	df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
	df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
	return df_feature

# Function to get group stats for the stock_id and time_id
def get_time_stock(df):
	vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 'log_return1_realized_volatility_450', 'log_return2_realized_volatility_450',
				'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 'log_return1_realized_volatility_150', 'log_return2_realized_volatility_150', 
				'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_400', 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_200']

	# Group by the stock id
	df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
	# Rename columns joining suffix
	df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
	df_stock_id = df_stock_id.add_suffix('_' + 'stock')

	# Group by the stock id
	df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
	# Rename columns joining suffix
	df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
	df_time_id = df_time_id.add_suffix('_' + 'time')
	
	# Merge with original dataframe
	df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
	df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
	df.drop(['stock_id__stock', 'time_id__time'], axis = 1, inplace = True)
	return df

# Function to make preprocessing function in parallel (for each stock id)
def preprocessor(list_stock_ids, is_train=True):
	
	# Parallel for loop
	def for_joblib(stock_id):
		# Train
		if is_train:
			file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
			file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
		# Test
		else:
			file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
			file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
	
		# Preprocess book and trade data and merge them
		df_tmp = pd.merge(book_preprocessor(file_path_book), trade_preprocessor(file_path_trade), on = 'row_id', how = 'left')

		# Return the merge dataframe
		return df_tmp

	# Use parallel api to call paralle for loop
	df = Parallel(n_jobs=15, verbose=1)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)	
	# Concatenate all the dataframes that return from Parallel
	df = pd.concat(df, ignore_index=True)
	return df

def tau_features(train, test):
	# replace by order sum (tau)
	train['size_tau'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique'] )
	test['size_tau'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique'] )
	train['size_tau_400'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique_400'] )
	test['size_tau_400'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique_400'] )
	train['size_tau_300'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique_300'] )
	test['size_tau_300'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique_300'] )
	train['size_tau_200'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique_200'] )
	test['size_tau_200'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique_200'] )

	train['size_tau2'] = np.sqrt( 1/ train['trade_order_count_sum'] )
	test['size_tau2'] = np.sqrt( 1/ test['trade_order_count_sum'] )
	train['size_tau2_400'] = np.sqrt( 0.33/ train['trade_order_count_sum'] )
	test['size_tau2_400'] = np.sqrt( 0.33/ test['trade_order_count_sum'] )
	train['size_tau2_300'] = np.sqrt( 0.5/ train['trade_order_count_sum'] )
	test['size_tau2_300'] = np.sqrt( 0.5/ test['trade_order_count_sum'] )
	train['size_tau2_200'] = np.sqrt( 0.66/ train['trade_order_count_sum'] )
	test['size_tau2_200'] = np.sqrt( 0.66/ test['trade_order_count_sum'] )

	# delta tau
	train['size_tau2_d'] = train['size_tau2_400'] - train['size_tau2']
	test['size_tau2_d'] = test['size_tau2_400'] - test['size_tau2']

	return train, test

def era_projection(train, test):

	reducer_times = umap.UMAP()

	df_times = train.groupby('time_id').mean()
	scaler = StandardScaler()
	X = scaler.fit_transform(df_times.dropna())
	projection = reducer_times.fit_transform(X)
	df_proj = pd.DataFrame()
	df_proj['x'] = projection[:,0]
	df_proj['y'] = projection[:,1]
	df_proj['time_id'] = df_times.index.values

	train = pd.merge(train, df_proj, on=['time_id'])

	df_times_test = test.groupby('time_id').mean()
	X_test = scaler.fit_transform(df_times_test.dropna())
	projection_test = reducer_times.transform(X_test)
	df_proj_test = pd.DataFrame()
	df_proj_test['x'] = projection_test[:,0]
	df_proj_test['y'] = projection_test[:,1]
	df_proj_test['time_id'] = df_times_test.index.values

	test = pd.merge(test, df_proj_test, on=['time_id'])

	return train, test

# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
	return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

# Function to early stop with root mean squared percentage error
def feval_rmspe(y_pred, lgb_train):
	y_true = lgb_train.get_label()
	return 'RMSPE', rmspe(y_true, y_pred), False


########################################## USAGE #############################################

# # Read train and test
# train, test = read_train_test()

# # Get unique stock ids 
# train_stock_ids = train['stock_id'].unique()
# # Preprocess them using Parallel and our single stock id functions
# train_ = preprocessor(train_stock_ids, is_train = True)
# train = train.merge(train_, on = ['row_id'], how = 'left')

# # Get unique stock ids 
# test_stock_ids = test['stock_id'].unique()
# # Preprocess them using Parallel and our single stock id functions
# test_ = preprocessor(test_stock_ids, is_train = False)
# test = test.merge(test_, on = ['row_id'], how = 'left')

# # Get group stats of time_id and stock_id
# train = get_time_stock(train)
# test = get_time_stock(test)

# # Save train dataset
# train.to_csv(data_dir+'processed_train.csv', index=False)

# # Traing and evaluate

# test_predictions = train_and_evaluate(train, test)

# # Save test predictions

# test['target'] = test_predictions
# test[['row_id', 'target']].to_csv('submission.csv',index = False)