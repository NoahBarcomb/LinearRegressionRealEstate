# Linear Regression Analysis

<p>This is a test model that is used to predict the sell price of a house given some predictor variables. The data has been pulled from https://catalog.data.gov/dataset/real-estate-sales-2001-2018. <br>
Upon running, the R-squared value should be: 0.9056681587237484. <br>
Generally, this means that the model describes 90% of the variance in the data. <br>
The Mean Squared Error is: 2295230040.094677 <br>
Generally, this indicates that there are some outliers still within the data that is causing the skew. In this model, outliers are removed using an IQR then an Anova Test to determine the variables with strong correlation, p-values that are greater than .05 are removed.<br>
<p>To visualize the MSE, some predictions are printed: </p>
<p>Actual: 312371.0, Predicted: 382215.85065459786 </p>
<p>Actual: 285000.0, Predicted: 295048.1037694014 </p>
<p>Actual: 112500.0, Predicted: 124460.36475173547 </p>
<p>Actual: 26000.0, Predicted: -56740.67663472489 </p>
<p>Actual: 200000.0, Predicted: 191188.34564437787 </p>
<p>You can see that for the most part, the predictions are close, but there is a strange value present that is in the negatives. This is proof the MSE is correct.</p>
