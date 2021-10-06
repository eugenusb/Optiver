from numpy.random import seed
seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
import numpy as np
from keras import backend as K
def swish(x, beta = 1):
	return (x * K.sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from . import mdn
from . import mdn_keras
from scipy.stats import invgamma
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import catboost as ctb
import seaborn as sns
import umap
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from .utils import *

early_stop = tf.keras.callbacks.EarlyStopping(
	monitor='val_loss', patience=20, verbose=0,
	mode='min', restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
	monitor='val_loss', factor=0.2, patience=7, verbose=0,
	mode='min')

hidden_units = (128,64,32)
stock_embedding_size = 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################################################################
# PYTORCH MODELS
#############################################################################################

# Prepare data for Pytorch MDN Model

def data_load(x_train, y_train, stocks_train, x_test, y_test, stocks_test, batch_size=1024, move_to_device=True):

	if move_to_device:
		train_data = TensorDataset(torch.from_numpy(x_train).float().to(device), torch.from_numpy(stocks_train).int().to(device), torch.from_numpy(y_train).float().to(device))
		test_data = TensorDataset(torch.from_numpy(x_test).float().to(device), torch.from_numpy(stocks_test).int().to(device), torch.from_numpy(y_test).float().to(device))
	else:
		train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(stocks_train).int(), torch.from_numpy(y_train).float())
		test_data = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(stocks_test).int(), torch.from_numpy(y_test).float())
	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	return train_dataloader, test_dataloader


def get_predictions(alpha, sigma):
	n = alpha.shape[0]
	preds = np.zeros(n)

	for i in range(n):
		sample_gamma = invgamma.rvs(alpha[i], loc=0, scale=sigma[i], size=1000)
		inv_expected_value = ( 1 / sample_gamma ).mean()
		inv_variance = ( 1 / (sample_gamma**2) ).mean()

		preds[i] = inv_expected_value / inv_variance

	return preds

def get_predictions_MDN(params, n_gaussians=3):
	print('Starting sampling from mixture of gaussians....')

	n = params.shape[0]
	preds = np.zeros(n)

	for i in range(n):
		mus, sigs, pi_logits = mdn_keras.split_mixture_params(params[i], 1, n_gaussians)
		pis = mdn_keras.softmax(pi_logits)		
		simul = np.zeros(1000)
		cat = np.random.choice(a=np.arange(0,n_gaussians), p=pis, size=1000)
		mus_sample = np.array([mus[c] for c in cat])
		sigs_sample = np.array([sigs[c] for c in cat])
		for j in range(1000):
			simul[j] = np.random.normal(mus_sample[j],sigs_sample[j])

		inv_expected_value = ( 1 / simul ).mean()
		inv_variance = ( 1 / (simul**2) ).mean()

		preds[i] = inv_expected_value / inv_variance

	return preds

def mspe_loss(output, target):
	loss = torch.mean(((output - target)/target)**2)
	return loss

class RMSPE(Metric):
	def __init__(self):
		self._name = "rmspe"
		self._maximize = False

	def __call__(self, y_true, y_score):
		
		return np.sqrt(np.mean(np.square((y_true - y_score) / y_true)))
	
def RMSPELoss(y_pred, y_true):
	return torch.sqrt(torch.mean( ((y_true - y_pred) / y_true) ** 2 )).clone()

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
	def __init__(self, n_features, tam, stock_emb_dim=20):
		super(InvGamma_model, self).__init__()
		
		self.n_features = n_features
		self.silu = nn.SiLU()
		self.flatten = nn.Flatten()
		self.linear1 = nn.Linear(n_features, 128)
		self.embedding = nn.Embedding(tam, stock_emb_dim)
		self.linear2 = nn.Linear(128 + stock_emb_dim, 64)
		self.linear3 = nn.Linear(64, 32)
		self.density = mdn.InvGamma(32)

	def forward(self, x, stocks):
		output = self.linear1(x)
		stock_emb = self.embedding(stocks)
		stock_emb = self.flatten(stock_emb)
		merged = torch.cat((output, stock_emb), dim=1)
		output = self.silu(merged)
		output = self.linear2(output)
		output = self.silu(output)
		output = self.linear3(output)
		output = self.silu(output)
		output = self.density(output)

		return output

class FFNN(nn.Module):
	def __init__(self, n_features, tam, stock_emb_dim=20):
		super(FFNN, self).__init__()
		
		self.n_features = n_features
		self.silu = nn.SiLU()
		self.flatten = nn.Flatten()
		self.linear1 = nn.Linear(n_features, 128)
		self.embedding = nn.Embedding(tam, stock_emb_dim)
		self.linear2 = nn.Linear(128 + stock_emb_dim, 64)
		self.linear3 = nn.Linear(64, 32)
		self.linear4 = nn.Linear(32, 1)

	def forward(self, x, stocks):
		output = self.linear1(x)
		stock_emb = self.embedding(stocks)
		stock_emb = self.flatten(stock_emb)
		merged = torch.cat((output, stock_emb), dim=1)
		output = self.silu(merged)
		output = self.linear2(output)
		output = self.silu(output)
		output = self.linear3(output)
		output = self.silu(output)
		output = self.linear4(output)

		return output

#############################################################################################
# KERAS MODELS
#############################################################################################

def root_mean_squared_per_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square( (y_true - y_pred)/ y_true )))

def base_model(n_features, tam):
	num_input = keras.Input(shape=(n_features,), name='num_data')
	stock_id_input = keras.Input(shape=(1,), name='stock_id')	

	#embedding, flatenning and concatenating
	stock_embedded = keras.layers.Embedding(tam, stock_embedding_size, input_length=1, name='stock_embedding')(stock_id_input)
	stock_flattened = keras.layers.Flatten()(stock_embedded)
	out = keras.layers.Concatenate()([stock_flattened, num_input])

	# Add one or more hidden layers
	for n_hidden in hidden_units:
		out = keras.layers.Dense(n_hidden, activation='swish')(out)       

	# A single output: our predicted rating
	out = keras.layers.Dense(1, activation='linear', name='prediction')(out)
	
	model = keras.Model(inputs=[num_input, stock_id_input], outputs=out,)

	return model

def mdn_keras_model(n_features, tam, n_gaussians=3):
	num_input = keras.Input(shape=(n_features,), name='num_data')
	stock_id_input = keras.Input(shape=(1,), name='stock_id')	

	#embedding, flatenning and concatenating
	stock_embedded = keras.layers.Embedding(tam, stock_embedding_size, input_length=1, name='stock_embedding')(stock_id_input)
	stock_flattened = keras.layers.Flatten()(stock_embedded)
	out = keras.layers.Concatenate()([stock_flattened, num_input])

	# Add one or more hidden layers
	for n_hidden in hidden_units:
		out = keras.layers.Dense(n_hidden, activation='swish')(out)
	out = mdn_keras.MDN_Keras(1, n_gaussians)(out)

	model = keras.Model(inputs=[num_input, stock_id_input], outputs=out,)	

	return model


def invgamma_keras_model(n_features, tam):
	num_input = keras.Input(shape=(n_features,), name='num_data')
	stock_id_input = keras.Input(shape=(1,), name='stock_id')	

	#embedding, flatenning and concatenating
	stock_embedded = keras.layers.Embedding(tam, stock_embedding_size, input_length=1, name='stock_embedding')(stock_id_input)
	stock_flattened = keras.layers.Flatten()(stock_embedded)
	out = keras.layers.Concatenate()([stock_flattened, num_input])

	# Add one or more hidden layers
	for n_hidden in hidden_units:
		out = keras.layers.Dense(n_hidden, activation='swish')(out)
	out = mdn_keras.InvGamma_Keras()(out)

	model = keras.Model(inputs=[num_input, stock_id_input], outputs=out)

	return model


def train_and_evaluate_NN(train, test, folds=5, save_model=True):
	predictors = [col for col in list(train.columns) if col not in {"stock_id", "time_id", "target", "row_id"}]

	train.replace([np.inf, -np.inf], np.nan,inplace=True)
	test.replace([np.inf, -np.inf], np.nan,inplace=True)    
	x = train[predictors]
	stocks = train['stock_id']
	y = train['target']
	test = test[predictors]

	tam = stocks.unique().max()+1

	# for col in predictors:
	# qt_train = []
	# qt = QuantileTransformer(random_state=21,n_quantiles=2000, output_distribution='normal')
	# train_nn[col] = qt.fit_transform(train_nn[[col]])
	# test_nn[col] = qt.transform(test_nn[[col]])    
	# qt_train.append(qt)

	# Create out of folds array
	oof_predictions = np.zeros(x.shape[0])
	# Create test array to store predictions
	test_predictions = np.zeros(test.shape[0])
	# Create a KFold object
	kfold = KFold(n_splits=folds, shuffle=True, random_state=2020)

	train[predictors] = train[predictors].fillna(train[predictors].mean())
	test[predictors] = test[predictors].fillna(train[predictors].mean())

	for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
		print(f'Training fold {fold + 1}')
		x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
		stock_train, stock_val = stocks.iloc[trn_ind], stocks.iloc[val_ind]
		y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

		#############################################################################################
		# NN
		#############################################################################################

		model = base_model(x_train.shape[1], tam)
		model.compile(keras.optimizers.Adam(learning_rate=0.006), loss=root_mean_squared_per_error)

		scaler = MinMaxScaler(feature_range=(0, 1))         
		x_train = scaler.fit_transform(x_train.values)
		x_val = scaler.transform(x_val.values)

		model.fit([x_train, stock_train], y_train, batch_size=2048, epochs=1000, validation_data=([x_val, stock_val], y_val),
					callbacks=[early_stop, plateau], shuffle=True, verbose = 1)

		oof_predictions[val_ind] = model.predict([x_val, stock_val]).reshape(-1)

	rmspe_score = rmspe(y, oof_predictions)
	print(f'Our out of folds RMSPE is {rmspe_score}')

	return oof_predictions

def train_and_evaluate_MDN_keras(train, test, folds=5, n_gaussians=3, epochs=1000, save_model=True):
	predictors = [col for col in list(train.columns) if col not in {"stock_id", "time_id", "target", "row_id"}]

	train.replace([np.inf, -np.inf], np.nan,inplace=True)
	test.replace([np.inf, -np.inf], np.nan,inplace=True)    
	x = train[predictors]
	stocks = train['stock_id']
	y = train['target']
	test = test[predictors]

	tam = stocks.unique().max()+1

	# for col in predictors:
	# qt_train = []
	# qt = QuantileTransformer(random_state=21,n_quantiles=2000, output_distribution='normal')
	# train_nn[col] = qt.fit_transform(train_nn[[col]])
	# test_nn[col] = qt.transform(test_nn[[col]])    
	# qt_train.append(qt)

	# Create out of folds array
	oof_predictions = np.zeros(x.shape[0])
	# Create test array to store predictions
	test_predictions = np.zeros(test.shape[0])
	# Create a KFold object
	kfold = KFold(n_splits=folds, shuffle=True, random_state=2020)

	train[predictors] = train[predictors].fillna(train[predictors].mean())
	test[predictors] = test[predictors].fillna(train[predictors].mean())

	for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
		print(f'Training fold {fold + 1}')
		x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
		stock_train, stock_val = stocks.iloc[trn_ind], stocks.iloc[val_ind]
		y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

		#############################################################################################
		# MDN
		#############################################################################################

		model = mdn_keras_model(x_train.shape[1], tam)
		model.compile(loss=mdn_keras.get_mixture_loss_func(1, n_gaussians), optimizer=keras.optimizers.Adam())

		scaler = MinMaxScaler(feature_range=(0, 1))         
		x_train = scaler.fit_transform(x_train.values)
		x_val = scaler.transform(x_val.values)

		model.fit([x_train, stock_train], y_train, batch_size=2048, epochs=epochs, validation_data=([x_val, stock_val], y_val),
					callbacks=[early_stop, plateau], shuffle=True, verbose = 1)

		oof_predictions[val_ind] = get_predictions_MDN(model.predict([x_val, stock_val]), n_gaussians)

	rmspe_score = rmspe(y, oof_predictions)
	print(f'Our out of folds RMSPE is {rmspe_score}')

	return oof_predictions


def train_and_evaluate_InvGamma_keras(train, test, folds=5, n_gaussians=3, epochs=1000, save_model=True):
	predictors = [col for col in list(train.columns) if col not in {"stock_id", "time_id", "target", "row_id"}]

	train.replace([np.inf, -np.inf], np.nan,inplace=True)
	test.replace([np.inf, -np.inf], np.nan,inplace=True)    
	x = train[predictors]
	stocks = train['stock_id']
	y = 1000*train['target']
	test = test[predictors]

	tam = stocks.unique().max()+1

	# Create out of folds array
	oof_predictions = np.zeros(x.shape[0])
	# Create test array to store predictions
	test_predictions = np.zeros(test.shape[0])
	# Create a KFold object
	kfold = KFold(n_splits=folds, shuffle=True, random_state=2020)

	train[predictors] = train[predictors].fillna(train[predictors].mean())
	test[predictors] = test[predictors].fillna(train[predictors].mean())

	for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
		print(f'Training fold {fold + 1}')
		x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
		stock_train, stock_val = stocks.iloc[trn_ind], stocks.iloc[val_ind]
		y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

		#############################################################################################
		# Inv Gamma
		#############################################################################################

		model = invgamma_keras_model(x_train.shape[1], tam)
		model.compile(loss=mdn_keras.inv_gamma_loss, optimizer=keras.optimizers.Adam())

		scaler = MinMaxScaler(feature_range=(0, 1))         
		x_train = scaler.fit_transform(x_train.values)
		x_val = scaler.transform(x_val.values)

		model.fit([x_train, stock_train], y_train, batch_size=2048, epochs=epochs, validation_data=([x_val, stock_val], y_val),
					callbacks=[early_stop, plateau], shuffle=True, verbose = 1)

		output = model.predict([x_val, stock_val])
		alpha, sigma = output[:,0], output[:,1]
		oof_predictions[val_ind] = get_predictions(alpha, sigma)
		fold_score = rmspe(y_val, oof_predictions[val_ind])
		print(f'Fold RMSPE score {fold_score}')
	rmspe_score = rmspe(y, oof_predictions)
	print(f'Our out of folds RMSPE is {rmspe_score}')

	return oof_predictions


def train_and_evaluate(train, test, seed=29, folds=5, log=False, save_model=True):
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
	x = train.drop(['row_id', 'target', 'time_id'], axis = 1)
	y = train['target']
	if log:
		y = np.log(y)

	x_test = test.drop(['row_id', 'time_id'], axis = 1)
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
		
		if log:
			train_dataset = lgb.Dataset(x_train, y_train, categorical_feature = ['stock_id'])
			val_dataset = lgb.Dataset(x_val, y_val, categorical_feature = ['stock_id'])
			model = lgb.train(params = params, 
							  train_set = train_dataset, 
							  valid_sets = [train_dataset, val_dataset], 
							  num_boost_round = 6000, 
							  early_stopping_rounds = 300, 
							  verbose_eval = 100)			
		else:			
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

	if log:
		score = mean_squared_error(y, oof_predictions, squared=False)
	else:
		score = rmspe(y, oof_predictions)

	print(f'Our out of folds score is {score}')
	# Return test predictions
	#return test_predictions
	return oof_predictions

def train_and_evaluate_FFNN(train, test, seed=29, folds=5, epochs=40, save_model=True):
	# Split features and target
	correlations = train.corrwith(train.target).abs()
	predictors = list(correlations[correlations > 0.5].index.drop('target'))

	x = train[predictors]
	y = train['target']
	stock_ids = train.stock_id
	x_test = test[predictors]
	# Transform stock id to a numeric value
	#x['stock_id'] = x['stock_id'].astype(int)
	#x_test['stock_id'] = x_test['stock_id'].astype(int)
	tam = train.stock_id.unique().max()+1

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
		stocks_train, stocks_val = stock_ids.iloc[trn_ind].values, stock_ids.iloc[val_ind].values
		y_train, y_val = y.iloc[trn_ind].values, y.iloc[val_ind].values

		scaler = MinMaxScaler(feature_range=(0,1))
		x_train = scaler.fit_transform(x_train)
		x_val = scaler.transform(x_val)
		model = FFNN(x_train.shape[1], tam)

		train_dataloader, val_dataloader = data_load(x_train, y_train, stocks_train, x_val, y_val, stocks_val)
		model.cuda()
		optimizer = optim.Adam(model.parameters())

		x_val_tensor, stocks_val_tensor = torch.tensor(x_val, dtype=torch.float).to(device), torch.tensor(stocks_val, dtype=torch.int).to(device)

		# train the model	

		for epoch in range(epochs):
			train_loss = 0
			counter = 0
			model.train()
			for inputs, stocks, labels in train_dataloader:
				model.zero_grad()
				output = model(inputs, stocks)
				loss = mspe_loss(output, labels)
				train_loss += loss.item()
				loss.backward()
				optimizer.step()
				counter += 1

			model.eval()
			loss_validation = rmspe(y_val, model(x_val_tensor, stocks_val_tensor).view(-1).detach().cpu().numpy()).item()
			print('TRAIN | Epoch: {}/{} | Training loss: {:.8f} | Validation loss: {:.8f}'.format(epoch+1, epochs, train_loss / counter, loss_validation))

		model.eval()
		# Add predictions to the out of folds array
		oof_predictions[val_ind] = model(x_val_tensor, stocks_val_tensor).view(-1).detach().cpu().numpy()
		# Predict the test set
		#test_predictions += model.predict(x_test) / 20

	rmspe_score = rmspe(y, oof_predictions)
	print(f'Our out of folds RMSPE is {rmspe_score}')
	# Return test predictions
	return test_predictions


def train_and_evaluate_InvGamma(train, test, seed=29, folds=5, epochs=40, save_model=True):

	# Split features and target
	correlations = train.corrwith(train.target).abs()
	predictors = list(correlations[correlations > 0.5].index.drop('target'))

	x = train[predictors]
	y = train['target']
	stock_ids = train.stock_id
	x_test = test[predictors]
	# Transform stock id to a numeric value
	#x['stock_id'] = x['stock_id'].astype(int)
	#x_test['stock_id'] = x_test['stock_id'].astype(int)
	tam = train.stock_id.unique().max()+1

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
		stocks_train, stocks_val = stock_ids.iloc[trn_ind].values, stock_ids.iloc[val_ind].values
		y_train, y_val = y.iloc[trn_ind].values, y.iloc[val_ind].values

		scaler = MinMaxScaler(feature_range=(0,1))
		x_train = scaler.fit_transform(x_train)
		x_val = scaler.transform(x_val)
		model = InvGamma_model(x_train.shape[1], tam)

		train_dataloader, val_dataloader = data_load(x_train, y_train, stocks_train, x_val, y_val, stocks_val)
		model.cuda()
		optimizer = optim.Adam(model.parameters())

		# train the model
		model.train()
		for epoch in range(epochs):
			train_loss = 0
			counter = 0
			for inputs, stocks, labels in train_dataloader:
				model.zero_grad()
				alpha, sigma = model(inputs, stocks)
				loss = mdn.inv_gamma_loss(alpha, sigma, labels)
				train_loss += loss.item()
				loss.backward()
				optimizer.step()
				counter += 1

			print('TRAIN | Epoch: {}/{} | Loss: {:.8f}'.format(epoch+1, epochs, train_loss / counter))

		model.eval()
		x_val_tensor, stocks_val_tensor = torch.tensor(x_val, dtype=torch.float).to(device), torch.tensor(stocks_val, dtype=torch.int).to(device)
		alpha, sigma = model(x_val_tensor, stocks_val_tensor)
		alpha, sigma = alpha.detach().cpu().numpy(), sigma.detach().cpu().numpy()
		# Add predictions to the out of folds array
		oof_predictions[val_ind] = get_predictions(alpha, sigma)
		# Predict the test set
		#test_predictions += model.predict(x_test) / 20

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
	kfold = KFold(n_splits=folds, random_state=1111, shuffle=True)
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


def train_and_evaluate_ctb(train, test, folds=5):

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
	kfold = KFold(n_splits=folds, random_state=1111, shuffle=True)
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


def tabnet_model(train, test, folds=5, save_model=True):

	# Split features and target
	x = train.drop(['row_id', 'target', 'time_id'], axis = 1)
	y = train['target']

	x_test = test.drop(['row_id', 'time_id'], axis = 1)
	# Transform stock id to a numeric value
	x['stock_id'] = x['stock_id'].astype(int)
	x_test['stock_id'] = x_test['stock_id'].astype(int)

	# Prepare data: encode categorical columns and scale the other ones

	categorical_columns = []
	categorical_dims =  {}

	for col in x.columns:
		if  col == 'stock_id':
			l_enc = LabelEncoder()
			x[col] = l_enc.fit_transform(x[col].values)
			x_test[col] = l_enc.transform(x_test[col].values)
			categorical_columns.append(col)
			categorical_dims[col] = len(l_enc.classes_)
		else:
			scaler = StandardScaler()
			x[col] = scaler.fit_transform(x[col].values.reshape(-1, 1))
			x_test[col] = scaler.transform(x_test[col].values.reshape(-1, 1))

	cat_idxs = [ i for i, f in enumerate(x.columns.tolist()) if f in categorical_columns]
	cat_dims = [ categorical_dims[f] for i, f in enumerate(x.columns.tolist()) if f in categorical_columns]

	tabnet_params = dict(
		cat_idxs=cat_idxs,
		cat_dims=cat_dims,
		cat_emb_dim=1,
		n_d = 16,
		n_a = 16,
		n_steps = 2,
		gamma = 2,
		n_independent = 2,
		n_shared = 2,
		lambda_sparse = 0,
		optimizer_fn = optim.Adam,
		optimizer_params = dict(lr = (2e-2)),
		mask_type = "entmax",
		scheduler_params = dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
		scheduler_fn = CosineAnnealingWarmRestarts,
		seed = 42,
		verbose = 10
	)

	kfold = KFold(n_splits=folds, random_state=42, shuffle=True)
	
	# Create out of folds array
	oof_predictions = np.zeros(x.shape[0])
	feature_importances = pd.DataFrame()
	feature_importances["feature"] = x.columns.tolist()
	stats = pd.DataFrame()
	explain_matrices = []
	masks_ = []

	for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
		print(f'Training fold {fold + 1}')
		x_train, x_val = x.iloc[trn_ind].values, x.iloc[val_ind].values
		y_train, y_val = y.iloc[trn_ind].values.reshape(-1,1), y.iloc[val_ind].values.reshape(-1,1)

		tabnet =  TabNetRegressor(**tabnet_params)
		tabnet.fit(
		  x_train, y_train,
		  eval_set=[(x_val, y_val)],
		  max_epochs = 200,
		  patience = 50,
		  batch_size = 1024*20, 
		  virtual_batch_size = 128*20,
		  num_workers = 4,
		  drop_last = False,
		  eval_metric=[RMSPE],
		  loss_fn=RMSPELoss
		  )

		if save_model:
			saving_path_name = f"./fold{fold}"
			saved_filepath = tabnet.save_model(saving_path_name)

		#explain_matrix, masks = tabnet.explain(x_val)
		#explain_matrices.append(explain_matrix)
		#masks_.append(masks[0])
		#masks_.append(masks[1])

		oof_predictions[val_ind] = tabnet.predict(x_val).reshape(-1)
		#feature_importances[f"importance_fold{fold}+1"] = tabnet.feature_importances_

		#stats[f'fold{fold+1}_train_rmspe'] = tabnet.history['loss']
		#stats[f'fold{fold+1}_val_rmspe'] = tabnet.history['val_0_rmspe']

	print(f'Our out of folds RMSPE is: {rmspe(y, oof_predictions.flatten())}')

	return oof_predictions
	#return oof_predictions, stats, explain_matrices, masks_, feature_importances