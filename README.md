# VariablesSelection
==

Variables Selection detecting  important features from huge input variables.

This file provides a class "FeatureImportance" combining most vital feature selection models, which is convenient for users to call.

All the 13 methods we use are : LASSO, ElasticNet, SCAD, Knockoff, RandomForest, AdaBoost, GradientBoosting , ExtraTrees, LassoNet,GradientLearning,  LassoNet , GroupLasso and SHAP.

# Paper Links  

* [LASSO](https://www.jstor.org/stable/2346178)

* [SCAD](https://andrewcharlesjones.github.io/journal/scad.html)

* [Fix-xKnockoff](https://arxiv.org/pdf/1404.5609.pdf)

* [Model-XKnockoff](https://arxiv.org/pdf/1610.02351.pdf)

* [LassoNet](https://arxiv.org/pdf/1907.12207.pdf)

* [GroupLasso](http://www.columbia.edu/~my2550/papers/glasso.final.pdf)

* [SHAP](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

* [GradientLearning](https://jmlr.csail.mit.edu/papers/volume7/mukherjee06a/mukherjee06a.pdf)

# Packages Need


# Method To Use
Regardless of the method used, first instantiate the 'FeatureImportance' class.

To use LASSO, ElasticNet, SCAD, RandomForest, AdaBoost, ExtraTrees, GroupLasso:
```
filter=FeatureImportance(x,y,test_ratio=0.2,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
coef, total=filter.GetCoefficient1(filter.ExtraTreesModel,max_depth=5,estimator_num=100)   
```

To use GradientLearning,  SHAP , Knockoff:
```
filter=FeatureImportance(x,y,test_ratio=0.001,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
coef, total=filter.GetCoefficient2(filter_fun=filter.GradientLearningFilter,eps=0.25,l1_lamda=0.5,kernel_type="Gaussian")
```

To use LassoNet :
```
filter=FeatureImportance(x,y,test_ratio=0.2,threshold=0,wanted_num=2,task='regression',scarler=None,times=10)
coef, total=filter.LassoNetModel(hidden_dims=(64,),M=10,plot=True)
```

coef is the important score of each feature, and total is the summaration time of the feature be choosen during all the experiments.





