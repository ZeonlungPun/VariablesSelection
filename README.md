# VariablesSelection

Variables Selection detecting  important features from huge input variables.

This file VariableSelection.py provides a class "FeatureImportance" combining most vital feature selection models, which is convenient for users to call.

All the 14 methods we use are : LASSO, ElasticNet, SCAD, Knockoff, RandomForest, AdaBoost, GradientBoosting , ExtraTrees, LassoNet,GradientLearning,  LassoNet , GroupLasso ,Layer-WiseRelevancePropagation and SHAP.

Among these algorithms, LASSO, ElasticNet, SCAD , GroupLasso are based on linear model ;  RandomForest, AdaBoost, GradientBoosting , ExtraTrees are Tree ensemble models ; LassoNet and Layer-WiseRelevancePropagation combine the neural network and features seletion

# Paper Links  

* [LASSO](https://www.jstor.org/stable/2346178)

* [SCAD](https://andrewcharlesjones.github.io/journal/scad.html)

* [Fix-xKnockoff](https://arxiv.org/pdf/1404.5609.pdf)

* [Model-XKnockoff](https://arxiv.org/pdf/1610.02351.pdf)

* [LassoNet](https://arxiv.org/pdf/1907.12207.pdf)

* [GroupLasso](http://www.columbia.edu/~my2550/papers/glasso.final.pdf)

* [SHAP](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

* [GradientLearning](https://jmlr.csail.mit.edu/papers/volume7/mukherjee06a/mukherjee06a.pdf)

* [Layer-WiseRelevancePropagation](https://iphome.hhi.de/samek/pdf/MonXAI19.pdf)

* [DeepLIFT](https://arxiv.org/abs/1704.02685)


# Packages Version Need

```
knockpy==1.3.0
lassonet==0.0.14
numpy==1.24.4
group-lasso==1.5.0
matplotlib==3.7.2
torch==2.0.1
shap==0.42.1
statsmodels==0.13.5
captum==0.6.0
```


# Method To Use

Regardless of the method used, first instantiate the 'FeatureImportance' class.

__To use LASSO, ElasticNet, SCAD, RandomForest, AdaBoost, ExtraTrees, GroupLasso__:
```
filter=FeatureImportance(x,y,test_ratio=0.2,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
coef, total=filter.GetCoefficient1(filter.ExtraTreesModel,max_depth=5,estimator_num=100)   
```

__To use GradientLearning,  SHAP , Layer-WiseRelevancePropagation ,DeepLIFT, Knockoff__:
```
filter=FeatureImportance(x,y,test_ratio=0.001,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
coef, total=filter.GetCoefficient2(filter_fun=filter.GradientLearningFilter,eps=0.25,l1_lamda=0.5,kernel_type="Gaussian")
```

__To use LassoNet__ :
```
filter=FeatureImportance(x,y,test_ratio=0.2,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
coef, total=filter.LassoNetModel(hidden_dims=(64,),M=10,plot=True)
```

'coef' is the important score of each feature, and 'total' is the summaration time of the feature be choosen during all the experiments.

# Example

```
#create data
n=200
p=50
xita=0.25
w=np.random.normal(loc=1,scale=1,size=(n,p))
u=np.random.normal(loc=1,scale=1,size=(n,p))
x=(w+xita*u)/(1+xita)
y=((2*x[:,0]-1)*(2*x[:,1]-1)).reshape((-1,1))

#execute feature selection 
filter=FeatureImportance(x,y,test_ratio=0.2,threshold=0,wanted_num=2,task='regression',scarler='MinMaxScaler',times=20)
coef, total=filter.GetCoefficient2(filter.SHAP,hidden_num=(12,),plot=True)
```
In the function filter.GetCoefficient1 or filter.GetCoefficient2 , you need to pass a feature selection function in 'FeatureImportance' class as first parameter, 
other parameters passed depends on the  feature selection method.


# Visualization

if the parameter 'plot' in filter.GetCoefficient1 and filter.GetCoefficient2
```
plot=True
```


The results of knockoff can be visualized as :<br>
![image](https://github.com/ZeonlungPun/VariablesSelection/blob/main/knokcoff_result.png)



The results of SHAP can be visualized as : 
![image](https://github.com/ZeonlungPun/VariablesSelection/blob/main/vital.png)



Visualization of LassoNet's tuning hyperparameters process:
![image](https://github.com/ZeonlungPun/VariablesSelection/blob/main/lassonet_result.png)

