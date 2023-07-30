from sklearn.linear_model import Lasso,ElasticNet
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
import statsmodels.api as sm
from knockpy.knockoff_filter import KnockoffFilter
import matplotlib.pyplot as plt
from lassonet import LassoNetRegressor,LassoNetClassifier
from group_lasso import GroupLasso
from sklearn.metrics import mean_squared_error

class SCAD(object):
    """
    COORDINATE DESCENT ALGORITHMS FOR NONCONVEX PENALIZED REGRESSION,
    WITH APPLICATIONS TO BIOLOGICAL FEATURE SELECTION
    """

    def __init__(self, x, y, lambd, gamma=3.7, iteration=100):
        self.iteration = iteration
        assert ~any(np.isnan(x).ravel())
        assert ~any(np.isnan(y))
        assert lambd > 0
        self.x = x
        self.y = y
        assert self.y.shape[0] == self.x.shape[0]
        self.n = self.y.shape[0]
        self.p = self.x.shape[1]
        self.lambd = lambd
        self.gamma = float(gamma)
        self.__standard()

    def __standard(self):
        self.ymean = self.y.mean()
        self.reg_y = self.y - self.ymean
        self.xmean = self.x.mean(axis=0)
        self.xvar = self.x.var(axis=0)
        self.reg_x = (self.x - self.xmean) / (np.sqrt(self.xvar))

    def __S(self, z, lambd):
        if z > lambd:
            return z - lambd
        elif z < ((-1) * lambd):
            return z + lambd
        else:
            return 0

    def __fscad(self, z):
        if np.abs(z) > (self.gamma * self.lambd):
            return z
        elif np.abs(z) <= (2 * self.lambd):
            return self.__S(z, self.lambd)
        else:
            return self.__S(z, (self.gamma * self.lambd / (self.gamma - 1)) / (1 - 1 / (self.gamma - 1)))

    def fit(self):
        self.initial_model = sm.WLS(self.reg_y, self.reg_x).fit()
        self.params = self.initial_model.params
        self.resid = self.initial_model.resid
        for m in range(self.iteration):
            self.Z = np.dot(self.reg_x.T, self.resid) / self.n + self.params
            parms=[self.__fscad(z)    for z in self.Z]
            self.params = np.array(parms)
        self._coef = self.params / np.sqrt(self.xvar)
        self._intercept = self.ymean - np.sum(self.params * self.xmean / np.sqrt(self.xvar))


    def predict(self, x_pre):
        return np.dot(x_pre, self._coef) + self._intercept

class FeatureImportance(object):
    """
    this class use to evaluate the feature importance with some common methods: LASSO, ElasticNet, SCAD, Knockoff,
    RandomForest,AdaBoost,GradientBoosting ,LassoNet, NeuralNetwork with estimated gradients, numerial gradients
    """
    def __init__(self,x,y,test_ratio,threshold,task='regression',scarler=None,times=20,wanted_num):
        """
        :param x: input  variables
        :param y: output  variables with only 1 dimension
        :param test_ratio: test set ratio
        :param task:  regression or classification
        :param threshold: when accuracy or r2_score > threshold ,the results are accepted
        :param  scarler :  the method of preprocess :  MinMaxScaler,StandardScaler  or None
        :param  times : independent ,replication  experiment times
               wanted_num: most vital features we want
        """
        self.x=x
        self.y=y
        self.test_ratio=test_ratio
        self.task=task
        self.threshold=threshold
        self.scarler=scarler
        self.times=times
        self.wanted_num=wanted_num

    @staticmethod
    def CalculateImportance(score,imp_nums):
        """
        :param score: estimating the feature importance
        :param imp_nums: wanted most important k features
        :return: the most important k features
        """
        top_k = imp_nums
        top_k_idx = score.argsort()[::-1][0:top_k]

        return top_k_idx

    def DataPreprocess(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_ratio)
        if self.scarler == 'MinMaxScaler':
            scalrx = MinMaxScaler()
            scalry = MinMaxScaler()
            self.x_train = scalrx.fit_transform(self.x_train)
            self.x_test = scalrx.transform(self.x_test)
            self.y_train = scalry.fit_transform(self.y_train.reshape((-1, 1)))
            self.y_test = scalry.transform(self.y_test.reshape((-1, 1)))

        elif self.scarler == 'StandardScaler':
            scalrx=StandardScaler()
            scalry=StandardScaler()
            self.x_train = scalrx.fit_transform(self.x_train)
            self.x_test = scalrx.transform(self.x_test)
            self.y_train = scalry.fit_transform(self.y_train.reshape((-1, 1)))
            self.y_test = scalry.transform(self.y_test.reshape((-1, 1)))

    def LASSO(self,lamda,group=None):
        """
        LASSO model
        :param lamda: the regularization coefficient,only for regression
               group:numpy array, same group with same num, example: [0,0,1,1,1,2,2,2,2,3,3]
        :return: fitted lasso model
        """
        if self.task =='classification':
            raise ValueError('task can not be classification')
        self.model_name='LASSO'
        if not group:
            lasso = Lasso(alpha=lamda,max_iter=200)
        else:
            lasso=GroupLasso(groups=group,group_reg=5,
    l1_reg=lamda,frobenius_lipschitz=True,scale_reg="inverse_group_size",
    subsampling_scheme=1,supress_warning=True,n_iter=1000,tol=1e-3)

        lasso.fit(self.x_train,self.y_train)
        return lasso

    def ElasticNET(self,l1,l2):
        """
        ElasticNET model mixing ridge and LASSO,only for regression
        :param l1: the l1 regularization coefficient
        :param l2: the l2 regularization coefficient
        :return: fitted  model
        """
        if self.task =='classification':
            raise ValueError('task can not be classification')
        ELNet= ElasticNet(l1_ratio=l1, alpha=l2)
        ELNet.fit(self.x_train,self.y_train)
        return ELNet

    def SCAD_model(self,gamma,lambd):
        """
        Smoothly Clipped Absolute Deviation, only for regression
        :param gamma:
        :param lambd:  default lambd==sqrt(2 log p) in experience
        :return: fitted  model
        """
        if self.task =='classification':
            raise ValueError('task can not be classification')
        model = SCAD(self.x_train, self.y_train, gamma=gamma, lambd=lambd, iteration=100)
        model.fit()
        return model

    def RandomForestModel(self,max_depth,estimator_num):

        if self.task =='classification':
            rf=RandomForestClassifier(max_depth=max_depth,n_estimators=estimator_num)
        else:
            rf=RandomForestRegressor(max_depth=max_depth,n_estimators=estimator_num)
        rf.fit(self.x_train,self.y_train)
        return rf

    def AdaBoostModel(self,estimator_num):
        if self.task =='classification':
            Ada=AdaBoostClassifier(n_estimators=estimator_num)
        else:
            Ada=AdaBoostRegressor(n_estimators=estimator_num)
        Ada.fit(self.x_train,self.y_train)
        return Ada

    def GradientBoostingModel(self,max_depth,estimator_num):
        if self.task == 'classification':
            GB=GradientBoostingClassifier(max_depth=max_depth,n_estimators=estimator_num)
        else:
            GB=GradientBoostingRegressor(max_depth=max_depth,n_estimators=estimator_num)
        GB.fit(self.x_train,self.y_train)
        return GB

    def ExtraTreesModel(self,max_depth,estimator_num):
        if self.task == 'classification':
            ET=ExtraTreesClassifier(max_depth=max_depth,n_estimators=estimator_num)
        else:
            ET=ExtraTreesRegressor(max_depth=max_depth,n_estimators=estimator_num)
        ET.fit(self.x_train,self.y_train)
        return ET


    def KnockoffFilter(self,mode=1,fdr=0.2,plot=False):
        """
        :param mode: method of knockoff :  1 for fixed X model with  estimated cov matrix ,
        2 for  Random forest statistics with swap importances
        :param fdr: the false discovery rate we can tolerate
        :return: selection results of filter (rejections) , importance scores
        """
        if mode==1:
            kfilter = KnockoffFilter(ksampler='fx', fstat='lasso')
            rejections = kfilter.forward(X=self.x_train, y=self.y_train, fdr=fdr, shrinkage="ledoitwolf")
            score=kfilter.W
        else:
            kfilter = KnockoffFilter(ksampler='gaussian', fstat='randomforest')
            rejections= kfilter.forward(X=self.x_train, y=self.y_train, fdr=fdr, shrinkage="ledoitwolf")
            score=kfilter.W
        if plot:
            kfilter.seqstep_plot()
            plt.savefig('knokcoff_result.png')

        return rejections,score

    def LassoNetModel(self,hidden_dims,M,group=None,plot=False):
        """
        Lasso Net filter for variable selections
        :param hidden_dims: tuple :(64,32,16)
        :param M: a hyperparameters for controling the liner effective
        :param group: for detecting group importance [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12], list(range(13, 26))]
              plot : plot the mse trend with lambda and number of features
        :return: eature importance score and the selection number of different features
        """
        def LassoNetFilter(hidden_dims,M,group=None,plot=False):
            """
            :return: fitted model
            """
            score = []
            n_selected = []
            mse = []
            lambda_ = []
            if self.task=='regression':
                model=LassoNetRegressor(hidden_dims=hidden_dims,M=M,patience=(100,5),groups=group)
                score_metrics=r2_score

            else:
                model = LassoNetClassifier(hidden_dims=hidden_dims, M=M, patience=(100, 5), groups=group)
                score_metrics =accuracy_score

            #split the validation test to choose the hyperparameters
            xtrain,xvalid,ytrain,yvalid=train_test_split(self.x_train,self.y_train,test_size=0.2)

            path=model.path(xtrain,ytrain)

            # choose the best hyperparameter with train data ,because regularization coefficient lamda need to choose in a range
            for save in path:
                model.load(save.state_dict)
                ypred=model.predict(xvalid)
                score.append(score_metrics(yvalid,ypred))
                n_selected.append(save.selected.sum())
                mse.append(mean_squared_error(yvalid, ypred))
                lambda_.append(save.lambda_)

            #get the best lambda
            max_index=np.argmax(score)
            model_save=path[max_index]
            model.load(model_save.state_dict)
            if plot:
                fig = plt.figure(figsize=(12, 12))
                plt.subplot(311)
                plt.grid(True)
                plt.plot(n_selected, mse, ".-")
                plt.xlabel("number of selected features")
                plt.ylabel("MSE")

                plt.subplot(312)
                plt.grid(True)
                plt.plot(lambda_, mse, ".-")
                plt.xlabel("lambda")
                plt.xscale("log")
                plt.ylabel("MSE")

                plt.subplot(313)
                plt.grid(True)
                plt.plot(lambda_, n_selected, ".-")
                plt.xlabel("lambda")
                plt.xscale("log")
                plt.ylabel("number of selected features")

                plt.savefig("lassonet_result.png")

            return model

        self.DataPreprocess()
        total_choose = np.zeros((self.times, self.x_train.shape[1]))
        coef = 0
        for time in range(self.times):
            print('the round {} for fitting LassoNet'.format(time))
            model=LassoNetFilter(hidden_dims,M,group,plot)
            if self.task=='regression':
                test_score=r2_score(y_true=self.y_test,y_pred=model.predict(self.x_test))
            else:
                test_score=accuracy_score(self.y_test,model.predict(self.x_test))
            if test_score < self.threshold:
                raise ValueError(' prediction accuracy is too low')
            coef_=model.feature_importances_.numpy()
            id = self.CalculateImportance(coef_, self.wanted_num)
            total_choose[time, id] = 1
            coef += coef_
        total = np.sum(total_choose, axis=0).reshape((1, -1))
        coef = np.mean(coef)
        return coef, total

    def NetworkGradient(self,net,backend):
        """
        estimate the gradient dy/dx with trained neural network
        :param net: trained neural network, default with GPU: cuda 0
        :param backend:  tensorflow or torch ; backend must be matched with trained network
        :return:   feature importance score and the total selection times of different features
        """
        if backend=='torch':
            import torch
            #define input tensor
            x=torch.tensor(self.x,requires_grad=True,dtype=torch.float64)
            # forward inference
            y=net(x)
            y.backward()
            #get the gradient
            grad_x=x.grad.cpu.numpy()
        else:
            import tensorflow as tf
            x=tf.convert_to_tensor(x,dtype=tf.float64)

            with tf.GradientTape() as tape:
                y=net(x)
            grad_x=tape.gradient(y,x).numpy()
        return grad_x



    def GetCoefficient(self,model_fun,**kwargs):
        """
        :param model_fun : variable selection model function, including
        LASSO  ,ElasticNET ,SCAD,RandomForest,ExtraTrees,GradientBoosting,AdaBoost
        :param lamda: the regularization coefficient
        :return:  feature importance score and the selection times of different features
        """
        self.DataPreprocess()
        total_choose = np.zeros((self.times, self.x_train.shape[1]))
        coef=0
        for time in range(self.times):
            print('the round {} for fitting model {} '.format(time,self.model_name))
            model=model_fun(**kwargs)
            try:
                test_score=r2_score(self.y_test, model.predict(self.x_test))
                print('the predicting test accuracy of {} is {}'.format(self.model_name,test_score))
                if test_score < self.threshold :
                    raise ValueError(' prediction accuracy is too low')
            except:
                pass

            try:
                coef_ = abs(model.coef_)
            #for ensemble models
            except:
                coef_ = model.feature_importances_
            id = self.CalculateImportance(coef_, self.wanted_num)
            total_choose[time, id] = 1
            coef+=coef_
        total = np.sum(total_choose, axis=0).reshape((1, -1))
        coef=np.mean(coef)
        return coef,total
