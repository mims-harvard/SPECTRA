from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score
import torch, torch.nn as nn, torch.optim as optim, numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy import stats

def return_relevant_metrics(total_y, y_pred, val_state = False):
	fpr, tpr, thresholds = roc_curve(total_y, y_pred)
	roc_auc = auc(fpr, tpr)
	apc = average_precision_score(total_y, y_pred)
	

	gmeans = np.sqrt(tpr * (1-fpr))
	ix = np.argmax(gmeans)

	best_threshold = thresholds[ix]

	if torch.is_tensor(y_pred[0]):
		thresholded_result = np.where(torch.Tensor(y_pred) >= best_threshold, 1, 0)			
	else:
		thresholded_result = np.where(y_pred >= best_threshold, 1, 0)

	f1_score_result = f1_score(total_y, thresholded_result)
	accuracy = accuracy_score(total_y, thresholded_result)

	if sum(total_y):
		tn, fp, fn, tp = confusion_matrix(total_y, thresholded_result).ravel()		
		sensitivity = tp/(tp+fn)
		specificity = tn/(tn+fp)
	else:
		sensitivity = 0
		specificity = 0

	return {f'{val_state}_auc':roc_auc, f'{val_state}_apc': apc , f'{val_state}_sensitivity': sensitivity ,  f'{val_state}_specificity': specificity, f'{val_state}_f1_score': f1_score_result }

def return_relevant_metrics_KL(total_y, y_pred):
	KL = nn.KLDivLoss(reduction='batchmean')
	loss = KL(torch.stack(total_y).log(), torch.stack(y_pred))
	return {'KL_Divergence': loss}

def return_categorical_metrics(y_actual, y_pred, strains, val_state = False):
	"""
		https://arxiv.org/pdf/2203.16427.pdf
		All Based on Mean Absolute Error
		For Categorical Prediction.
		(1) Bin into percentile categories, calculate loss in each, average, percentile-MSE
		(2) Tail 5% Performance
		(3) Tail 10% Performance
		(4) Unbinned Overall Loss
		(5) Each percentile performance

		ALL METRICS ASSUME NORMALIZED SCORES
	"""	

	overall_MAE = mean_absolute_error(y_pred, y_actual)
	sort_index = np.argsort(y_actual)
	
	yas = y_actual[sort_index] 
	yps = y_pred[sort_index]

	MAE_top_5 = mean_absolute_error(yps[len(yps)-round(len(yps)*0.05):len(yps)],yas[len(yas)-round(len(yas)*0.05):len(yas)]) 
	MAE_top_10 = mean_absolute_error(yps[len(yps)-round(len(yps)*0.1):len(yps)],yas[len(yas)-round(len(yas)*0.1):len(yas)])


	percentile_MAES = []
	less_25_cond = y_actual < 0.25
	between_25_50_cond = (y_actual >= 0.25) & (y_actual < 0.5)
	between_50_75_cond = (y_actual >= 0.5) & (y_actual < 0.75)
	more_75_cond = y_actual >= 0.75	

	to_return = {f'{val_state}_overall_MAE': overall_MAE,
                f'{val_state}_top_5_MAE': MAE_top_5,
                f'{val_state}_top_10_MAE': MAE_top_10}

	percentile_MAES = []

	if sum(less_25_cond):	
		less_25 = mean_absolute_error(y_pred[less_25_cond], y_actual[less_25_cond])
		percentile_MAES.append(less_25)
		to_return[f'{val_state}_less_25_MAE'] = less_25
	if sum(between_25_50_cond):
		between_25_50 = mean_absolute_error(y_pred[between_25_50_cond], y_actual[between_25_50_cond])
		percentile_MAES.append(between_25_50)
		to_return[f'{val_state}_between_25_50'] = between_25_50
	if sum(between_50_75_cond):
		between_50_75 =  mean_absolute_error(y_pred[between_50_75_cond], y_actual[between_50_75_cond])
		percentile_MAES.append(between_50_75)
		to_return[f'{val_state}_between_50_75'] = between_50_75
	if sum(more_75_cond):
		more_75 = mean_absolute_error(y_pred[more_75_cond], y_actual[more_75_cond])
		percentile_MAES.append(more_75)
		to_return[f'{val_state}_greater_75'] = more_75

	percentile_MAES = np.mean(percentile_MAES)
	
	to_return[f'{val_state}_percentile_MAE'] = percentile_MAES

	r_squared_value = r2_score(yas, yps)
	mse = mean_squared_error(yas, yps)

	to_return[f'{val_state}_rsquared'] = r_squared_value
	to_return[f'{val_state}_MSE'] = mse

	return to_return


from scipy import stats

def return_categorical_metrics_normal(y_actual, y_pred, samples, val_state = False):
	y_actual = y_actual[~np.isnan(y_pred)]
	y_pred = y_pred[~np.isnan(y_pred)]

	to_return = {}
	overall_MAE = mean_absolute_error(y_pred, y_actual)
	sort_index = np.argsort(y_actual)
	
	yas = y_actual[sort_index] 
	yps = y_pred[sort_index]
	mse = mean_squared_error(yas, yps)
	spearman_rank = stats.spearmanr(yas, yps).correlation

	if len(yas) > 15:
		r_squared_value = r2_score(yas, yps)
		to_return[f'{val_state}_rsquared'] = r_squared_value

	to_return[f'{val_state}_MSE'] = mse
	to_return[f'{val_state}_MAE'] = overall_MAE
	to_return[f'{val_state}_spearman_rank'] = spearman_rank
    
	yas = y_actual[sort_index] 
	yps = y_pred[sort_index]

	if len(yas) > 15:
		MAE_top_5 = mean_absolute_error(yps[len(yps)-round(len(yps)*0.05):len(yps)],yas[len(yas)-round(len(yas)*0.05):len(yas)]) 
		MAE_top_10 = mean_absolute_error(yps[len(yps)-round(len(yps)*0.1):len(yps)],yas[len(yas)-round(len(yas)*0.1):len(yas)])
    
		to_return[f'{val_state}_top_5%_MAE'] = MAE_top_5
		to_return[f'{val_state}_top_10%_MAE'] = MAE_top_10

		correlation_coefficient = np.corrcoef(yas, yps)[0][1]

		to_return[f'{val_state}_PC'] = correlation_coefficient    

	def top_k_accuracy(k,y_actual,y_pred):
		y_actual = np.array(y_actual)
		y_pred = np.array(y_pred)
		top_k_pred = samples[np.argsort(y_pred)][-k:]
		top_k_actual = samples[np.argsort(y_actual)][-k:]
        
		number_in_both = 0
        
        
		return len(np.intersect1d(top_k_pred, top_k_actual))/k

	if len(yas) > 15:
		samples = np.array(samples)
		top_5_accuracy = top_k_accuracy(5, y_actual, y_pred)
		top_10_accuracy = top_k_accuracy(10, y_actual, y_pred)
		top_20_accuracy = top_k_accuracy(20, y_actual, y_pred)
    
		to_return[f'{val_state}_top_5_prioritization'] = top_5_accuracy
		to_return[f'{val_state}_top_10_prioritization'] = top_10_accuracy
		to_return[f'{val_state}_top_20_prioritization'] = top_20_accuracy
    
	range_predictions = np.max(yps) - np.min(yps)
	to_return[f'{val_state}_range_predictions'] = range_predictions
    
	return to_return

def return_mlm_accuracy(labels, final_predictions, val_state = False):

	logits = final_predictions

	logits = logits.detach().cpu()
	labels = labels.detach().cpu()

	match = torch.argmax(logits[labels != -100], dim=-1) == labels[labels != -100]
	accuracy = match.sum()/match.size()[0]
	return {f'{val_state}_accuracy': accuracy} 