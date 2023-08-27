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
from sklearn.neural_network import MLPRegressor,MLPClassifier
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from captum.attr import DeepLift

##create some class for Layer-WiseRelevancePropagation and DeepLIFT algorithm
# 1,create a simple network and train
class FeedForwardNetNetwork(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNetNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=30)
        self.fc2 =nn.Linear(in_features=30, out_features=5)
        self.fc3 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))) )

# 2,create data loader
class Loader(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __getitem__(self,index):
        data=self.data[index]
        labels=self.label[index]
        return data,labels
    def __len__(self):
        return len(self.data)

#3 , create main function for LRP

# a train network function for class LayerWiseRelevancePropagation and  DeepLIFT
def TrainModel(xtrain,ytrain,xtest,ytest,epochs,device='cpu'):


    """
    train the Neural Network
    :param epochs: iteration times
    :param device: cpu or cuda
    """

    #make a dataloader
    trainData = Loader(xtrain, ytrain)
    trainData = DataLoader(trainData)
    testData = Loader(xtest, ytest)
    testData = DataLoader(testData)

    #create and build neural network
    input_size = xtrain.shape[1]
    num_epochs = epochs
    model = FeedForwardNetNetwork(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    # train the model and evaluate it
    for epoch in range(num_epochs):
        for data, labels in trainData:
            # print(data.shape,labels.shape)
            data = data.float().to(device=device)
            labels = labels.float().to(device=device)
            pred = model(data)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred, true = [], []
        with torch.no_grad():
            for data, labels in testData:
                data = data.float().to(device=device)
                labels = labels.float().to(device=device)
                pred_ = model(data)
                pred.append(pred_.detach().numpy()[0])
                true.append(labels.detach().numpy()[0])

        true, pred = np.array(true).reshape((-1, 1)), np.array(pred).reshape((-1, 1))
        print('the test score in epoch {} is {}'.format(epoch,r2_score(y_true=true, y_pred=pred)))
    return model




class LayerWiseRelevancePropagation(object):
    def __init__(self,xtrain,ytrain,xtest,ytest):
        """
        :param xtrain, ytrain, xtest, ytest:  all data with numpy array
        """
        self.xtrain,self.ytrain=xtrain,ytrain
        self.xtest,self.ytest=xtest,ytest
        self.x,self.y=np.concatenate([self.xtrain,self.xtest],axis=0),np.concatenate([self.ytrain,self.ytest],axis=0)

    def predict(self,model,xnew):
        """
        predict using trained model
        :param xnew: new data
        :return: predict results
        """
        if len(xnew.shape)==1:
            xnew=xnew.reshape((1,-1))
        xnew=torch.tensor(xnew,dtype=torch.float)
        predict=model(xnew).data.numpy()

        return predict


    def main(self,model,x,y):
        """
        this function execute the layer wise relevance propagation algorithm when given neural network 'model' and dataset (x,y)
        :param model: neural network has been trainned
        :param x: input data,2-d array
        :param y: output data,1-d array
        :return: feature importance score
        """
        #get the weights of hidden layers
        params=model.state_dict()
        W=[params['fc1.weight'].numpy().T,params['fc2.weight'].numpy().T,params['fc3.weight'].numpy().T]
        B=[params['fc1.bias'].numpy(),params['fc2.bias'].numpy(),params['fc3.bias'].numpy()]

        L=3 # L is the layer number
        # the forward pass can be computed as a sequence of matrix multiplications and nonlinearities
        # return A as a list with length of layer number
        A = [x]+[None]*L
        for l in range(L):
            A[l+1] = np.maximum(0,A[l].dot(W[l])+B[l]  )

        #create a list to store relevance scores at each layer
        # return R as a list with length of layer number
        T=y
        if len(T.shape)==1:
            T=T[:,None]
        R = [None]*L + [A[L]*T  ]

        def rho(w,l):  return w + [None,0.1,0.0,0.0][l] * np.maximum(0,w)
        def incr(z,l): return z + [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9

        L=3
        for l in range(1, L)[::-1]:
            w = rho(W[l], l)
            b = rho(B[l], l)

            z = incr(A[l].dot(w) + b, l)  # step 1
            s = R[l + 1] / z  # step 2
            c = s.dot(w.T)  # step 3
            R[l] = A[l] * c  # step 4

        # propagate relevance scores until the pixels, apply an alternate propagation rule that properly handles pixel values received as input
        w  = W[0]
        wp = np.maximum(0,w)
        wm = np.minimum(0,w)
        lb = A[0]*0-1
        hb = A[0]*0+1

        z = A[0].dot(w)-lb.dot(wp)-hb.dot(wm)+1e-9        # step 1
        s = R[1]/z                                        # step 2
        c,cp,cm  = s.dot(w.T),s.dot(wp.T),s.dot(wm.T)     # step 3
        R[0] = A[0]*c-lb*cp-hb*cm
        important_score=np.mean(R[0],axis=0)

        return important_score

#create for DeepLIFT model
#DeepLIFT model is designed to attribute the change between the input and baseline to
# a predictive class or a value that the neural network outputs.
class DeepLIFT(object):
    def __init__(self,xtrain,ytrain,xtest,ytest):
        self.xtrain, self.ytrain = xtrain, ytrain
        self.xtest, self.ytest = xtest, ytest
        self.x, self.y = np.concatenate([self.xtrain, self.xtest], axis=0), np.concatenate([self.ytrain, self.ytest],axis=0)

    def main(self,model,x,y):
        model.eval()
        x=torch.tensor(x,dtype=torch.float32)
        deeplift=DeepLift(model)
        baseline=torch.zeros_like(x,dtype=torch.float32)
        attributions,delta=deeplift.attribute(x,baseline,return_convergence_delta=True)
        attributions=np.mean(attributions.detach().numpy(),axis=0)
        return attributions



#create class for implementing gradient learning

class GradientLearning(object):
    def __init__(self,x,y,eps,lambd):
        """
        Class for the Gradient Learning Variable Selection method
        :param x: a matrix x that is m by dim (p) where m is the number of samples
        :param y: a vector y that is m by 1
        :param eps: a constraint on the ratio of the top s eigenvalues to the sum over all eigenvalues
        :param lambd: the regularization constant
        """
        self.x=x
        self.y=y
        self.eps=eps
        self.lambd=lambd

    def GaussianKernel(self,sigma=8):
        """
        sigma : hyperparameter of gaussian kernel
        calculate the gaussian kernel  using the sample of x
        :return:  gaussian kernel mxm
        """
        temp=np.sum(self.x*self.x,axis=1,keepdims=True)
        dist_norm=temp+temp.T-2* self.x  @ (self.x.T)
        return np.exp(-dist_norm/(2*sigma**2))

    def ComputeWeightVariance(self):
        """
        :return: the variance of the weight matrix computed automatically from the data
        """
        temp = np.sum(self.x * self.x, axis=1, keepdims=True)
        self.dist_norm = temp + temp.T - 2 * self.x @ (self.x.T)
        self.sigma = np.median(self.dist_norm)

    def main(self,kernel_type):
        """
        the main function for gradient learning variable selection algorithm
        :param kernel_type: "Gaussian" Kernel or   "linear" kernel
        :return: nrm: the RKHS norm for each dimension ; F : the gradient evaluated at each sample again a p by m matrix
        """
        m,p=self.x.shape[0:2]
        self.ComputeWeightVariance()
        #computes the weight matrix
        w=(1/(self.sigma*np.sqrt(2*np.pi)))*np.exp(-self.dist_norm/(2*self.sigma**2))
        #give the kernel matrix
        if kernel_type=="Gaussian":
            K=self.GaussianKernel()
        else:
            K=self.x@ (self.x.T)

        #constructs the matrix of differences between all m samples
        xm=self.x[m-1,:].reshape((-1,1))
        Mx=self.x.T-xm

        #SVD decompose
        V,Sigma,UT=np.linalg.svd(Mx)

        #inverse accumulate (begin from smallest)
        cum_Sigma=np.cumsum(Sigma[::-1])

        #find the drop out index according to the ratio of  accumulate singular value
        judge=( (cum_Sigma/cum_Sigma[-1])  < self.eps)
        cut_index=np.max(np.where(judge==1))

        #get the remain number
        s=p-cut_index-1

        #projects of the paired differences into the subspace of the s eigenfunctions
        t=np.zeros((s,m))
        for j in range(m):
            t[:,j]=Sigma[0:s]* (UT[0:s,j])

        #initialize the transient matrix
        Ktilde=np.zeros((m*s,m*s))
        ytilde=np.zeros((m*s,1))

        # computes the Ktilde matrix and the vector script Y
        for i in range(m):
            Bmat = np.zeros((s, s))
            yi = np.zeros((s, 1))
            for j in range(m):
                Bmat = Bmat + w[i, j] * (t[:, j] - t[:, i]) * (t[:, j] - t[:, i]).T
                yi = yi + w[i, j] * (y[j, 0] - y[i, 0]) * (t[:, j].reshape((-1,1)) - t[:, i].reshape((-1,1)))
            ytilde[i * s :(i+1) *s] = yi
            for j in range(m):
                Ktilde[i*s: (i+1)*s, j*s: (j+1)*s]=K[i,j]*Bmat

        #solves the linear system for coefficients c
        I=np.eye(m*s)
        c=np.linalg.inv(m**2*self.lambd*I+Ktilde) @ ytilde

        #uwraps the coefficients into a vector for each sample
        Cmat=np.zeros((p,m))
        for i in range(m):
            vec=np.zeros((p,1))
            for l in range(s):
                vec=vec+ c[i*s+l,0]*V[:,l].reshape((-1,1))
            Cmat[:,i]=vec.ravel()

        #computes the gradient for each sample
        F= Cmat @ K

        nrm=np.zeros((p,1))

        for i in range(p):
            nrm[i,0]=Cmat[i,:]@ K @ (Cmat[i,:].T)

        return F, nrm

# create class for SCAD

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
        self.coef_ = self.params / np.sqrt(self.xvar)
        self._intercept = self.ymean - np.sum(self.params * self.xmean / np.sqrt(self.xvar))


    def predict(self, x_pre):
        return np.dot(x_pre, self.coef_) + self._intercept

#A class combing all important features selection methods

class FeatureImportance(object):
    """
    this class use to evaluate the feature importance with some common methods: LASSO, ElasticNet, SCAD, Knockoff,
    RandomForest,AdaBoost,GradientBoosting ,LassoNet,  Gradients Learning ,LRP
    """
    def __init__(self,x,y,test_ratio,threshold,wanted_num,task='regression',scarler=None,times=20):
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
        if len(score.shape)>1:
            score=score.ravel()
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
        self.model_name='ElasticNet'
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
        self.model_name='SCAD'
        model = SCAD(self.x_train, self.y_train, gamma=gamma, lambd=lambd, iteration=100)
        model.fit()
        return model

    def RandomForestModel(self,max_depth,estimator_num):
        self.model_name='RandomForest'
        if self.task =='classification':
            rf=RandomForestClassifier(max_depth=max_depth,n_estimators=estimator_num)
        else:
            rf=RandomForestRegressor(max_depth=max_depth,n_estimators=estimator_num)
        rf.fit(self.x_train,self.y_train.ravel())
        return rf

    def AdaBoostModel(self,estimator_num):
        self.model_name='AdaBoost'
        if self.task =='classification':
            Ada=AdaBoostClassifier(n_estimators=estimator_num)
        else:
            Ada=AdaBoostRegressor(n_estimators=estimator_num)
        Ada.fit(self.x_train,self.y_train.ravel())
        return Ada

    def GradientBoostingModel(self,max_depth,estimator_num):
        self.model_name='GradientBoosting'
        if self.task == 'classification':
            GB=GradientBoostingClassifier(max_depth=max_depth,n_estimators=estimator_num)
        else:
            GB=GradientBoostingRegressor(max_depth=max_depth,n_estimators=estimator_num)
        GB.fit(self.x_train,self.y_train.ravel())
        return GB

    def ExtraTreesModel(self,max_depth,estimator_num):
        self.model_name='ExtraTrees'
        if self.task == 'classification':
            ET=ExtraTreesClassifier(max_depth=max_depth,n_estimators=estimator_num)
        else:
            ET=ExtraTreesRegressor(max_depth=max_depth,n_estimators=estimator_num)
        ET.fit(self.x_train,self.y_train.ravel())
        return ET


    def KnockoffFilter(self,mode=1,fdr=0.2,plot=False):
        """
        :param mode: method of knockoff :  1 for fixed X model with  estimated cov matrix ,
        2 for  Random forest statistics with swap importances
        :param fdr: the false discovery rate we can tolerate
        :return: selection results of filter (rejections) , importance scores
        """
        self.model_name='KnockoffFilter'
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

        return score

    def GradientLearningFilter(self,eps,l1_lamda,kernel_type="Gaussian"):
        """
        execute the gradient learning one time
        eps: a constraint on the ratio of the top s eigenvalues to the sum over all eigenvalues
        l1_lamda:  the regularization constant
        kernel_type:  Gaussian or linear
        return the RKHS norm for each dimension
        """
        self.model_name='GradientLearning'
        gl = GradientLearning(self.x, self.y, eps, l1_lamda)
        f, nrm = gl.main(kernel_type=kernel_type)

        return nrm

    def LRP(self,epochs,device='cpu'):
        self.model_name='LayerWiseRelevancePropagation'
        LRP=LayerWiseRelevancePropagation(self.x_train,self.y_train,self.x_test,self.y_test)
        model=TrainModel(xtrain=self.x_train,ytrain=self.y_train,xtest=self.x_test,ytest=self.y_test,epochs=epochs,device=device)
        score=LRP.main(model,self.x,self.y)

        return score

    def DeepLIFT(self,epochs,device='cpu'):
        self.model_name='DeepLIFT'
        deeplift=DeepLIFT(self.x_train,self.y_train,self.x_test,self.y_test)
        model=TrainModel(xtrain=self.x_train,ytrain=self.y_train,xtest=self.x_test,ytest=self.y_test,epochs=epochs,device=device)
        score=deeplift.main(model,self.x,self.y)

        return score

    def LassoNetModel(self,hidden_dims,M,group=None,plot=False):
        """
        Lasso Net filter for variable selections
        :param hidden_dims: tuple :(64,32,16) or 964,
        :param M: a hyperparameters for controling the liner effective
        :param group: for detecting group importance [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12], list(range(13, 26))]
              plot : plot the mse trend with lambda and number of features
        :return: eature importance score and the selection number of different features

        example:
        filter=FeatureImportance(x,y,test_ratio=0.2,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
        coef, total=filter.LassoNetModel(hidden_dims=(64,),M=10,plot=True)
        """
        self.model_name='LassoNet'
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


    def SHAP(self,hidden_num,plot=False):
        """
        SHAP method for feature importance using MLP as basic
        hidden_num: tuple, like (12,) ,(64,32)
         """
        if self.task=='regression':
            model=MLPRegressor(hidden_layer_sizes=hidden_num,max_iter=5000)
        elif self.task=='classification':
            model=MLPClassifier(hidden_layer_sizes=hidden_num,max_iter=5000)
        else:
            raise ValueError('task must be regression or classification')
        self.model_name='SHAP'
        model.fit(self.x_train,self.y_train)
        explainer = shap.KernelExplainer(model.predict, self.x_train)
        shap_values = explainer.shap_values(self.x_test, nsamples=int(self.x_test.shape[0]*0.65))

        if plot:
            shap.summary_plot(shap_values,self.x_test,show=False)
            plt.savefig('vital.png')
        shap_values = np.mean(abs(shap_values), axis=0)
        return shap_values


    def GetCoefficient1(self,model_fun,**kwargs):
        """
        this function use the methods are able to do prediction and feature selection at the same time
        :param model_fun : variable selection model function, including
        LASSO  ,ElasticNET ,SCAD,RandomForest,ExtraTrees,GradientBoosting,AdaBoost
        :return:  feature importance score and the selection times of different features

        example:
        filter=FeatureImportance(x,y,test_ratio=0.2,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
        coef, total=filter.GetCoefficient1(filter.ExtraTreesModel,max_depth=5,estimator_num=100)
        """
        self.DataPreprocess()
        total_choose = np.zeros((self.times, self.x_train.shape[1]))
        coef=0
        for time in range(self.times):
            model=model_fun(**kwargs)
            print('the round {} for fitting model {} '.format(time, self.model_name))
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


    def GetCoefficient2(self,filter_fun,**kwargs):
        """
        this function use the methods are able to do feature selection only
        filter_fun : the exeuting function of  GradientLearning , Knockoff ,SHAP,LRP
        return:  feature importance score and the selection times of different features

        example:
        filter=FeatureImportance(x,y,test_ratio=0.001,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
        coef, total=filter.GetCoefficient2(filter_fun=filter.GradientLearningFilter,eps=0.25,l1_lamda=0.5,kernel_type="Gaussian")
        or :
        coef, total=filter.GetCoefficient2(filter_fun=filter.KnockoffFilter,mode=1,fdr=0.2,plot=False)

        """
        self.DataPreprocess()
        total_choose = np.zeros((self.times, self.x_train.shape[1]))
        coef = 0
        for time in range(self.times):
            score = filter_fun(**kwargs)

            print('the round {} for fitting model {} '.format(time, self.model_name))
            id = self.CalculateImportance(abs(score), self.wanted_num)
            total_choose[time, id] = 1
            coef+=score
        total = np.sum(total_choose, axis=0).reshape((1, -1))
        coef = np.mean(coef)
        return coef, total




if __name__ == '__main__':
    n=200
    p=50
    xita=0.25
    w=np.random.normal(loc=1,scale=1,size=(n,p))
    u=np.random.normal(loc=1,scale=1,size=(n,p))
    x=(w+xita*u)/(1+xita)
    y=((2*x[:,1]-1)*(2*x[:,2]-1)).reshape((-1,1))


    filter=FeatureImportance(x,y,test_ratio=0.2,threshold=0,wanted_num=2,task='regression',scarler='MinMaxScaler',times=10)
    coef, total=filter.GetCoefficient2(filter.LRP,epochs=25)
    print(total)




