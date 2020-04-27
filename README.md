# DC8
Data Challenge 8 - Wine Classifier

This file needs nltk, pandas, and certain features of sklearn

In the part where you read in the data set, make sure to change the path for where it is in your computer.

Our objective was to copy what the other user did to predict Wine Rating Analysis but with a different classifier whether it be Naive Bayes or Logistic Regression (the other example used Random Forest Classifier). We chose to look at both types of classifiers and compare the accuracy between all models, including the normal logisticRegression model and a multinomial(MN) Logistic Regression. All together we tested four different classifiers: Naive Bayes, Logistic Regression, MN Logistic Regression, and Random Forest Classifier. In the end we found that Random Forest Classifier still performed the best, with an accuracy of ≈96% much like the example given. Following the RFC, we found that Logistic Regression and MN Logistic both performed at about the same accuracy of ≈90%, with Logistic Regression performing slightly better than MN Logistic Regression. And lastly, Naive Bayes performed the worst at an accuracy of ≈74%.
