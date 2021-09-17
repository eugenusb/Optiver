import os
import glob
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.model_selection import KFold
import lightgbm as lgb
import catboost as ctb
import seaborn as sns
import umap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from arch import arch_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from . import mdn
from scipy.stats import invgamma

import warnings
warnings.filterwarnings('ignore')

# directory
data_dir = 'data/'
models_dir = 'models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to calculate first WAP
def calc_wap1(df):
	wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
	return wap

# Function to calculate second WAP
def calc_wap2(df):
	wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
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
	# Calculate log returns
	df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
	df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)

	#df = df.groupby(['time_id']).apply(garch_est)

	# Calculate squared difference in time
	df['time_diff'] = df.groupby(['time_id'])['seconds_in_bucket'].diff()
	df.time_diff = df.time_diff.fillna(1)
	df['updates'] = df.groupby(['time_id'])['seconds_in_bucket'].transform('count')

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
		'log_return1': [np.sum, realized_volatility, np.mean, np.std],
		'log_return2': [np.sum, realized_volatility, np.mean, np.std],
		'wap_balance': [np.sum, np.mean, np.std],
		'price_spread':[np.sum, np.mean, np.std],
		'price_spread2':[np.sum, np.mean, np.std],
		'bid_spread': [np.sum, np.mean, np.std],
		'ask_spread': [np.sum, np.mean, np.std],
		'total_volume': [np.sum, np.mean, np.std],
		'volume_imbalance': [np.sum, np.mean, np.std],
		'bid_ask_spread': [np.sum, np.mean, np.std],
		'instant_volatility': [realized_volatility],		
	}
	
	create_time_feature_dict = {
		'time_diff': [np.sum, np.std],
		'updates': [np.mean],
	#	'garch_estimation': [np.mean],
	}

	# Function to get group stats for different windows (seconds in bucket)
	def get_stats_window(seconds_in_bucket, feature_dict=create_feature_dict, add_suffix = False):
		# Group by the window
		df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(feature_dict).reset_index()
		# Rename columns joining suffix
		df_feature.columns = ['_'.join(col) for col in df_feature.columns]
		# Add a suffix to differentiate windows
		if add_suffix:
			df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
		return df_feature

	# Get the stats for different windows
	df_feature = get_stats_window(seconds_in_bucket=0, add_suffix=False)
	df_feature_updates = get_stats_window(seconds_in_bucket=0, feature_dict=create_time_feature_dict, add_suffix=False)
	df_feature_updates['time_diff_sum'] = np.sqrt(df_feature_updates['time_diff_sum'].values)
	df_feature_450 = get_stats_window(seconds_in_bucket=450, add_suffix=True)
	df_feature_300 = get_stats_window(seconds_in_bucket=300, add_suffix=True)
	df_feature_150 = get_stats_window(seconds_in_bucket=150, add_suffix=True)

	# Merge all
	df_feature = df_feature.merge(df_feature_updates, how = 'left', left_on = 'time_id_', right_on = 'time_id_')
	df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
	df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
	df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
	# Drop unnecesary time_ids
	df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis = 1, inplace = True)

	df_sec = pd.DataFrame()
	df_sec['seconds'] = np.arange(0,601)

	# Create row_id so we can merge
	stock_id = file_path.split('=')[1]
	df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
	df_feature.drop(['time_id_'], axis = 1, inplace = True)

	return df_feature

# Function to preprocess trade data (for each stock id)
def trade_preprocessor(file_path):

	df = pd.read_parquet(file_path)
	df['log_return'] = df.groupby('time_id')['price'].apply(log_return)

	# Dict for aggregations
	create_feature_dict = {
		'log_return':[realized_volatility],
		'seconds_in_bucket':[count_unique],
		'size':[np.sum],
		'order_count':[np.mean],
	}

	# Function to get group stats for different windows (seconds in bucket)
	def get_stats_window(seconds_in_bucket, add_suffix = False):
		# Group by the window
		df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()

		# Rename columns joining suffix
		df_feature.columns = ['_'.join(col) for col in df_feature.columns]
		# Add a suffix to differentiate windows
		if add_suffix:
			df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
		return df_feature

	# Get the stats for different windows
	df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
	df_feature_450 = get_stats_window(seconds_in_bucket = 450, add_suffix = True)
	df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
	df_feature_150 = get_stats_window(seconds_in_bucket = 150, add_suffix = True)

	# Merge all
	df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
	df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
	df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
	# Drop unnecesary time_ids
	df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis = 1, inplace = True)

	df_feature = df_feature.add_prefix('trade_')
	stock_id = file_path.split('=')[1]
	df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
	df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
	return df_feature

# Function to get group stats for the stock_id and time_id
def get_time_stock(df):

	# Get realized volatility columns
	vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 'log_return1_realized_volatility_450', 'log_return2_realized_volatility_450', 
				'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 'log_return1_realized_volatility_150', 'log_return2_realized_volatility_150', 
				'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_450', 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_150']

	# Group by the stock id
	df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
	# Rename columns joining suffix
	df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
	df_stock_id = df_stock_id.add_suffix('_' + 'stock')

	#df_time_by_stock = df.groupby(['stock_id'])['squared_time_diff'].sum().reset_index()
	#df_stock_id = df_stock_id.merge(df_time_by_stock)
	
	# Group by the time id
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
def preprocessor(list_stock_ids, is_train = True):
	
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
	df = Parallel(n_jobs = -1, verbose = 1)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)
	# Concatenate all the dataframes that return from Parallel
	df = pd.concat(df, ignore_index = True)
	return df

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

def train_and_evaluate(train, test, seed=29, folds=5, save_model=True):
	# Hyperparammeters (optimized)
	
	params = {
		'learning_rate': 0.1,
		'lambda_l1': 2,
		'lambda_l2': 7,
		'num_leaves': 1000,
		'min_sum_hessian_in_leaf': 20,
		'feature_fraction': 0.8,
		'feature_fraction_bynode': 0.8,
		'bagging_fraction': 0.9,
		'bagging_freq': 42,
		'min_data_in_leaf': 700,
		'max_depth': 5,
		'seed': seed,
		'feature_fraction_seed': seed,
		'bagging_seed': seed,
		'drop_seed': seed,
		'data_random_seed': seed,
		'objective': 'rmse',
		'boosting': 'gbdt',
		'verbosity': -1,
		'n_jobs': -1,
	}   
	
	# Split features and target
	correlations = train.corrwith(train.target).abs()
	predictors = list(correlations[correlations > 0.5].index.drop('target')) + ['stock_id', 'time_id']

	x = train[predictors]
	y = train['target']
	x_test = test[predictors]
	# Transform stock id to a numeric value
	x['stock_id'] = x['stock_id'].astype(int)
	x_test['stock_id'] = x_test['stock_id'].astype(int)
	
	# Create out of folds array
	oof_predictions = np.zeros(x.shape[0])
	# Create test array to store predictions
	test_predictions = np.zeros(x_test.shape[0])
	# Create a KFold object
	kfold = KFold(n_splits=folds, random_state=1111, shuffle=True)
	# Iterate through each fold
	for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
		print(f'Training fold {fold + 1}')
		x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
		y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
		#x_train, x_val = era_projection(x_train, x_val)
		#x_train = x_train.sort_values(by=['stock_id', 'time_id'])
		#x_val = x_val.sort_values(by=['stock_id', 'time_id'])
		#x_train.drop(columns=['time_id'], inplace=True)
		#x_val.drop(columns=['time_id'], inplace=True)
		# Root mean squared percentage error weights
		train_weights = 1 / np.square(y_train)
		val_weights = 1 / np.square(y_val)
		train_dataset = lgb.Dataset(x_train, y_train, weight = train_weights, categorical_feature = ['stock_id'])
		val_dataset = lgb.Dataset(x_val, y_val, weight = val_weights, categorical_feature = ['stock_id'])
		model = lgb.train(params = params, 
						  train_set = train_dataset, 
						  valid_sets = [train_dataset, val_dataset], 
						  num_boost_round = 6000, 
						  early_stopping_rounds = 300, 
						  verbose_eval = 100,
						  feval = feval_rmspe)

		#if save_model:
		#    model.booster_.save_model(models_dir + 'lgbm_' + str(fold) + '.txt')
			# To load the model use:
			# model = lgb.Booster(model_file='mode.txt')

		plt.figure(figsize=(12,6))
		lgb.plot_importance(model, max_num_features=10)
		plt.title("Feature importance")
		plt.show()
		# Add predictions to the out of folds array
		oof_predictions[val_ind] = model.predict(x_val)
		# Predict the test set
		#test_predictions += model.predict(x_test) / 20

	rmspe_score = rmspe(y, oof_predictions)
	print(f'Our out of folds RMSPE is {rmspe_score}')
	# Return test predictions
	return test_predictions

# Prepare data for Pytorch MDN Model

def data_load(X_train, y_train, X_test, y_test, batch_size=32):

	train_data = TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device))
	test_data = TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).float().to(device))

	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	return train_dataloader, test_dataloader

class MDN_model(nn.Module):
	def __init__(self, n_features, n_gaussians=3):
		super(MDN_model, self).__init__()
		
		self.n_features = n_features
		self.silu = nn.SiLU()

		self.linear1 = nn.Linear(n_features, 64)
		self.linear2 = nn.Linear(64, 32)
		self.linear3 = nn.Linear(32, 16)
		self.mix_density = nn.MDN(16, 1, n_gaussians)

	def forward(self, x):
		output = self.linear1(x)
		output = self.silu(output)
		output = self.linear2(output)
		output = self.silu(output)
		output = self.linear3(output)
		output = self.silu(output)
		output = self.mix_density(output)

		return output

class InvGamma_model(nn.Module):
	def __init__(self, n_features):
		super(InvGamma_model, self).__init__()
		
		self.n_features = n_features
		self.silu = nn.SiLU()
		self.linear1 = nn.Linear(n_features, 64)
		self.linear2 = nn.Linear(64, 32)
		self.linear3 = nn.Linear(32, 16)
		self.density = mdn.InvGamma(16)

	def forward(self, x):
		output = self.linear1(x)
		output = self.silu(output)
		output = self.linear2(output)
		output = self.silu(output)
		output = self.linear3(output)
		output = self.silu(output)
		output = self.density(output)

		return output


def get_predictions(alpha, sigma):
	n = alpha.shape[0]
	preds = np.zeros(n)

	for i in range(n):
		sample_gamma = invgamma.rvs(alpha[i], loc=0, scale=sigma[i], size=1000)
		inv_expected_value = ( 1 / sample_gamma ).mean()
		inv_variance = ( 1 / (sample_gamma**2) ).mean()

		preds[i] = inv_expected_value / inv_variance

	return preds


def train_and_evaluate_InvGamma(train, test, seed=29, folds=5, epochs=40, save_model=True):

	# Split features and target
	correlations = train.corrwith(train.target).abs()
	predictors = list(correlations[correlations > 0.5].index.drop('target')) + ['stock_id', 'time_id']

	x = train[predictors]
	y = train['target']
	x_test = test[predictors]
	# Transform stock id to a numeric value
	x['stock_id'] = x['stock_id'].astype(int)
	x_test['stock_id'] = x_test['stock_id'].astype(int)

	# Create out of folds array
	oof_predictions = np.zeros(x.shape[0])
	# Create test array to store predictions
	test_predictions = np.zeros(x_test.shape[0])
	# Create a KFold object
	kfold = KFold(n_splits=folds, random_state=1111, shuffle=True)
	# Iterate through each fold	

	for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
		print(f'Training fold {fold + 1}')
		x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
		y_train, y_val = y.iloc[trn_ind].values, y.iloc[val_ind].values

		scaler = MinMaxScaler(feature_range=(0,1))
		x_train = scaler.fit_transform(x_train)
		x_val = scaler.transform(x_val)
		model = InvGamma_model(x_train.shape[1])

		train_dataloader, val_dataloader = data_load(x_train, y_train, x_val, y_val)
		model.cuda()
		optimizer = optim.Adam(model.parameters())

		# train the model
		model.train()
		for epoch in range(epochs):
			train_loss = 0
			counter = 0
			for inputs, labels in train_dataloader:
				inputs, labels = inputs.to(device), labels.to(device)
				model.zero_grad()
				alpha, sigma = model(inputs)
				loss = mdn.inv_gamma_loss(alpha, sigma, labels)
				train_loss += loss.item()
				loss.backward()
				optimizer.step()
				counter += 1

			print('TRAIN | Epoch: {}/{} | Loss: {:.8f}'.format(epoch+1, epochs, train_loss / counter))

		model.eval()
		val_tensor = torch.tensor(x_val, dtype=torch.float).to(device)
		alpha, sigma = model(val_tensor)
		alpha, sigma = alpha.detach().cpu().numpy(), sigma.detach().cpu().numpy()
		# Add predictions to the out of folds array
		oof_predictions[val_ind] = get_predictions(alpha, sigma)
		# Predict the test set
		#test_predictions += model.predict(x_test) / 20

		#if save_model:
		#    model.booster_.save_model(models_dir + 'lgbm_' + str(fold) + '.txt')
			# To load the model use:
			# model = lgb.Booster(model_file='mode.txt')


	rmspe_score = rmspe(y, oof_predictions)
	print(f'Our out of folds RMSPE is {rmspe_score}')
	# Return test predictions
	return test_predictions


def train_and_evaluate_MDN(train, test, seed=29, folds=5, epochs=40, save_model=True):

	# Split features and target
	correlations = train.corrwith(train.target).abs()
	predictors = list(correlations[correlations > 0.5].index.drop('target')) + ['stock_id', 'time_id']

	x = train[predictors]
	y = train['target']
	x_test = test[predictors]
	# Transform stock id to a numeric value
	x['stock_id'] = x['stock_id'].astype(int)
	x_test['stock_id'] = x_test['stock_id'].astype(int)

	# Create out of folds array
	oof_predictions = np.zeros(x.shape[0])
	# Create test array to store predictions
	test_predictions = np.zeros(x_test.shape[0])
	# Create a KFold object
	kfold = KFold(n_splits=folds, random_state=seed, shuffle=True)
	# Iterate through each fold

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
		print(f'Training fold {fold + 1}')
		x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
		y_train, y_val = y.iloc[trn_ind].values, y.iloc[val_ind].values

		scaler = MinMaxScaler(feature_range=(0,1))
		x_train = scaler.fit_transform(x_train)
		x_val = scaler.transform(x_val)
		model = MDN_model(x_train.shape[1])

		train_dataloader, val_dataloader = data_load(x_train, y_train, x_val, y_val)
		model.cuda()
		optimizer = optim.Adam(model.parameters())

		# train the model
		model.train()
		for epoch in range(epochs):
			train_loss = 0
			counter = 0
			for inputs, labels in train_dataloader:
				model.zero_grad()
				pi, sigma, mu = model(inputs)
				loss = mdn.mdn_loss(pi, sigma, mu, labels)
				train_loss += loss.item()
				loss.backward()
				optimizer.step()
				counter += 1

			print('TRAIN | Epoch: {}/{} | Loss: {:.8f}'.format(epoch+1, epochs, train_loss / counter))

		model.eval()
		val_tensor = torch.tensor(x_val, dtype=torch.float).to(device)
		pi, sigma, mu = model(val_tensor)
		pi, sigma, mu = torch.exp(pi).detach().cpu().numpy(), torch.exp(sigma).detach().cpu().numpy(), torch.exp(mu).detach().cpu().numpy()
		# Add predictions to the out of folds array
		oof_predictions[val_ind] = get_predictions(pi, sigma, mu)
		# Predict the test set
		#test_predictions += model.predict(x_test) / 20

		#if save_model:
		#    model.booster_.save_model(models_dir + 'lgbm_' + str(fold) + '.txt')
			# To load the model use:
			# model = lgb.Booster(model_file='mode.txt')


	rmspe_score = rmspe(y, oof_predictions)
	print(f'Our out of folds RMSPE is {rmspe_score}')
	# Return test predictions
	return test_predictions


def train_and_evaluate_ctb(train, test):

	seed = 29    
	# Split features and target
	x = train.drop(['row_id', 'target', 'time_id'], axis = 1)
	y = train['target']
	x_test = test.drop(['row_id', 'time_id'], axis = 1)
	# Transform stock id to a numeric value
	x['stock_id'] = x['stock_id'].astype(int)
	x_test['stock_id'] = x_test['stock_id'].astype(int)
	
	# Create out of folds array
	oof_predictions = np.zeros(x.shape[0])
	# Create test array to store predictions
	test_predictions = np.zeros(x_test.shape[0])
	# Create a KFold object
	kfold = KFold(n_splits = 20, random_state = 1111, shuffle = True)
	# Iterate through each fold
	
	cat_vars = ['stock_id']
	cat_indexes = [train.columns.get_loc(x) for x in cat_vars]
	
	for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
		print(f'Training fold {fold + 1}')
		x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
		y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
		# Root mean squared percentage error weights
		train_weights = 1 / np.square(y_train)
		val_weights = 1 / np.square(y_val)
		
		train_pool = ctb.Pool(x_train, y_train, weight=train_weights, cat_features=cat_indexes)
		val_pool = ctb.Pool(x_val, y_val, weight=val_weights, cat_features=cat_indexes)
		
		model = ctb.CatBoostRegressor(num_boost_round = 5000,                              
							   l2_leaf_reg = 7,
							   depth = 7,
							   max_bin = 40,
							   rsm = 0.7,
							   use_best_model = True,
							   one_hot_max_size = 30,
							   #logging_level = "Silent",
							   random_seed = seed)
		
		model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=300)

		feat_imp = pd.Series(model.feature_importances_, x_train.columns).sort_values(ascending=False)[:10]
		feat_imp.plot(kind = 'bar', title = 'Feature Importance')
		# Add predictions to the out of folds array
		oof_predictions[val_ind] = model.predict(x_val)
		# Predict the test set
		test_predictions += model.predict(x_test) / 20
		
	rmspe_score = rmspe(y, oof_predictions)
	print(f'Our out of folds RMSPE is {rmspe_score}')
	# Return test predictions
	return test_predictions


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