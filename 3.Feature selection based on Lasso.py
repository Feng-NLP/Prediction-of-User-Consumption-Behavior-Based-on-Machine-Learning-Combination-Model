import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

def pretty_print_linear(coefs, names = None, sort = False):
	if names is None:
		names = ["X%s" % x for x in range(len(coefs))]
	lst = zip(coefs, names)
	print(lst)
	if sort:
		lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
	return " + ".join("%s * %s" % (np.round(coef, 3), name) for coef, name in lst)

data = pd.read_csv('待lasso特征选择样本.csv')
X = data.iloc[:,0:41]

#X = scaler.fit_transform(boston["data"])
Y = data.iloc[:,41]
names = data.columns.values

lasso = Lasso(alpha=.3)
alpha_can = np.logspace(-3,2,10)#0.001-100的等比数列


lasso_model = GridSearchCV(lasso,param_grid = {'alpha':alpha_can},cv=5)
lasso_model.fit(X, Y)
print('超参数：',lasso_model.best_params_)

optism_alpha = lasso_model.best_params_

optism_lasso = Lasso(alpha=optism_alpha['alpha'])
optism_lasso.fit(X, Y)
print(optism_lasso.coef_,names)
print("Lasso model: ", pretty_print_linear(optism_lasso.coef_, names, sort = True))
