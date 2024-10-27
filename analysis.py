from sklearn.feature_selection import f_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pandas.read_csv("Real_Estate_Sales_2001-2022_GL.csv")

#Question 1:
#This dataset contains over 800,000 rows about real estate sales from 2001-2022. It has column headings like List Year Town Address Assessed Value Sale Amount Property Type Residential Type, that
#are similar to datasets we have covered when learning about regression analysis. It has predictor variables, a good target variable, and plenty of data to accurately predict.
#My prediction goal is to estimate the future cost of houses by using the predictor variables mentioned above

#Question 2

#predictor variables: Town, Date Recorded, List Year, Sales Ratio
#target variable: Sale Amount
#drop unwanted columns that have many n/a/null and duplicate values before removing n/a/null and duplicate values so dataset does not shrink so much

unwanted_features = ['Non Use Code', 'Assessor Remarks', 'OPM remarks', 'Location', 'Serial Number', 'Address', 'Date Recorded']
df = df.drop(columns = unwanted_features)

#drop na
print(f"[!] Size of dataset before dropping na and duplicate values: {df.size}")
df = df.dropna()

#drop duplicates
df = df.drop_duplicates()

print(f"[!] Size of dataset after dropping na and duplicate values: {df.size}")

#boxplot to identify outliers
numeric_cols = ['Assessed Value', 'List Year', 'Sales Ratio']

fig = plt.figure(figsize = (10, 8))

for i in range(len(numeric_cols)):
    column = numeric_cols[i]
    sub = fig.add_subplot(2,3, i + 1)

    sns.boxplot(x = column, data = df)
    
plt.show()

#q1 = .25
#q2 = .50
#q3 = .75
#q4 = 1.00
q1 = df[numeric_cols].quantile(0.25)
q3 = df[numeric_cols].quantile(0.75)
iqr = q3 - q1

#define outlier thresholds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

#create a boolean mask for rows without outliers
mask = (df[numeric_cols] >= lower_bound) & (df[numeric_cols] <= upper_bound)

#keep only rows without outliers
df = df[mask.all(axis = 1)]

#convert categorical into numerical
property_unique = df['Property Type'].unique()
residential_unique = df['Residential Type'].unique()

mapping = {value: i for i, value in enumerate(property_unique)}
df['Property Type'] = df['Property Type'].map(mapping)
mapping = {value: i for i, value in enumerate(residential_unique)}
df['Residential Type'] = df['Residential Type'].map(mapping)

#inconsistent values
df['Town'] = df['Town'].str.lower().str.strip()

encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the 'Town' column
encoded_town = encoder.fit_transform(df[['Town']])
encoded_town_df = pandas.DataFrame(encoded_town, columns=encoder.get_feature_names_out(), index = df.index)

# Join the encoded columns to the original DataFrame and drop the 'Town' column
df = df.join(encoded_town_df).drop(columns = ['Town'])

print("DataFrame After One-Hot Encoding:")
print(df.head())

#Question 3

#feature scaling

scaler = StandardScaler()
df[['Assessed Value', 'Sales Ratio']] = scaler.fit_transform(df[['Assessed Value', 'Sales Ratio']])

#ANOVA Test setup & execution

x = df.drop(columns = ['Sale Amount'])
y = df['Sale Amount']

f_scores, p_values = f_classif(x, y)

#display p-values
for feature, p_val in zip(x.columns, p_values):
    print(f'feature: {feature}, p-value: {p_val}')

significant_features = [feature for feature, p_val in zip(x.columns, p_values) if p_val < 0.05]
x_significant = x[significant_features]

#Question 4

x_train, x_test, y_train, y_test = train_test_split(x_significant, y, test_size = 0.2, random_state = 42)

model = LinearRegression()

#cross-validation
cv_scores = cross_val_score(model, x_train, y_train, cv = 10)

#train
model.fit(x_train, y_train)

#predict & evaluate
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

for actual, predicted in zip(y_test[:5], y_pred[:5]):
    print(f"Actual: {actual}, Predicted: {predicted}")

print(f'R-squared: {r2}')
