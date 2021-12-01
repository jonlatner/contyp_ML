# contyp_ML

A hidden secret in most research on temporary employment is that the explanatory power is quite low (r = 0.10 +/-).  Furthermore, to the degree that models are accurate, they are almost entirely based on predicting true negative (TN), not true positive (TN).  This project seeks to compare the traditional method (logistic) with modern methods of machine learning (ML), such as Naive Bayes (NB), Decision Tree (DT), and Linear Discrimination Analysis (LDA).  

To test this idea, we use data from the European Labour Force Survey (EU-LFS), between 1995 and 2019.  Our sample are prime-aged (25-54) individuals who are employed, with a contract type that is observable.  We use a limited set of independent variables, including education, age, and gender.  Our dichotomous dependent variable is temporary employment (reference: permanent employment).  One model is estimated separately for each country, year combination.  Note: While data are not publicly available, the code used to clean, analyze, and graph the data are available here.

Results suggest that ML techniques are far better than logistic regression in predicting both TN and TP.


