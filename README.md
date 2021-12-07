# contyp_ML

A hidden secret in most research on temporary employment is that the explanatory power is quite low (r = 0.10 +/-).  Furthermore, to the degree that models are accurate, they are almost entirely based on predicting true negative (TN), not true positive (TN).  This project seeks to compare the traditional method (logistic) with modern methods of machine learning (ML), such as Naive Bayes (NB), Decision Tree (DT), and Linear Discrimination Analysis (LDA).  

To test this idea, we use data from the European Labour Force Survey (EU-LFS), between 1998 and 2019.  Our sample are prime-aged (25-54) individuals who are employed, with a contract type that is observable.  We use a limited set of independent variables, including education, age, and gender, all of which are categorical.  Our dichotomous dependent variable is temporary employment (reference: permanent employment).  One model is estimated separately for each country, year combination.  Note: While data are not publicly available, the code used to clean, analyze, and graph the data are available here.

Results suggest that ML techniques are not so much better than logistic regression in predicting both TN and TP.

Below is a tweet threat I wrote:

I would say a hidden secret within research on temporary employment is that predictive power of basic models are quite low.  Say Psuedo R2 < 0.10.  Further, most, if not all the predictive power comes in the ability to predict who does NOT have a temporary contract, not who does.  Can machine learning help?

I am sharing a public repository for a new project to see if we can better predict temporary employment.  Unfortunately, data are private, but code is public.  I use data from the European Labour Force Survey (EU-LFS), between 1998 and 2019.  

The sample is prime-aged (25-54) individuals who are employed, with a contract type.  Dependent variable is temporary employment (reference: permanent employment).  Independent variables are education, age, and gender, all of which are categorical.  One model is estimated separately for each country, year combination.  

To be honest, the goal was to learn a little python and a little about machine learning.  Plus, I wanted to do it within an open science framework (i.e. GitHub).  That said, I do think there is some contributions to the broader research community as well.

Obviously, there is a well established discussion about the relevance/importance of R2.  Some say we place too much emphasis on it.  Others say too little.  Probably, the right answer is somewhere in the middle.

To me, I think there are two reasons why R2 matters.  First, in simple logits, if we are talking about odds ratios, then if younger people are much more likely to have a temporary contract, this must be understood within the context that our ability to predict who has a temporary contract is very low.

Second, in causal research using propensity score matching, when we predict the likelihood of temporary contract in order to estimate the causal effect of temporary work on some outcome, it matters that our ability to estimate who does and does not get a temporary contract is quite low, even if we can "match" with some relatively small caliper.

In contrast to most social science research, data science research is almost entirely and exclusively concerned with the ability to predict accurately.  Arguably, in the same way that social science should be more concerned with prediction, data science should be more concerned with variation and stratification.  

That said, can machine learning models help create predict better than standard models?  In this case, the answer is yes and no.  

On the one hand, standard machine learning techniques do not perform better than logit.  On the other hand, the Naive Bayes estimator, which is increasingly recommended in social science is superior.  That said, Youden's J, a simple correction to logit can greatly improve estimates.

Whats the lesson?  The main lesson is that social science and data science can learn a lot from each other.  

