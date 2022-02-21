# **Expected increase in Nitrous oxide emission**

Nitrous oxide (N2O) is a potent greehouse gas. It is considered a third most important greenhous gas after methane and carbon dioxide. It is considered as one of the biggest human related threat to the ozone layer. Agricultural fertilizer applications and dairy farming are the major source of N2O emission. Nearly a 30% increase in the emission of N2O was noted in the past four decades. Nitrous oxide posses 300 times higher atmosphere warming potential than CO2.

Based on the avaialable data we can predict the increase or decrease in the emission of N2O. There is an ample amount of data stored in the [FAO Database](https://www.fao.org/faostat/en/#data)

We are going to perform the prediction using ***Machine Learning*** algorithms. For that purpose we need important libraries to import, clean and analysis of data.\
Below we imported ***NumPy*** and ***Pandas*** libraries.


```python
## Importing the libraries
import numpy as np
import pandas as pd
```

The nitrous oxide emission data from FAO database was imported. The data include global N2O emission values from 1961-2019 from agricultural soils./
The data is imported from the CSV file using the codes below.


```python
df=pd.read_csv("emm_n2o.csv")
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Domain</th>
      <th>Area</th>
      <th>Element</th>
      <th>Item</th>
      <th>Year</th>
      <th>Source</th>
      <th>Unit</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emissions Totals</td>
      <td>World</td>
      <td>Direct emissions (N2O)</td>
      <td>Agricultural Soils</td>
      <td>1961</td>
      <td>FAO TIER 1</td>
      <td>kilotonnes</td>
      <td>1841.0876</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emissions Totals</td>
      <td>World</td>
      <td>Direct emissions (N2O)</td>
      <td>Agricultural Soils</td>
      <td>1962</td>
      <td>FAO TIER 1</td>
      <td>kilotonnes</td>
      <td>1896.9660</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emissions Totals</td>
      <td>World</td>
      <td>Direct emissions (N2O)</td>
      <td>Agricultural Soils</td>
      <td>1963</td>
      <td>FAO TIER 1</td>
      <td>kilotonnes</td>
      <td>1942.8741</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emissions Totals</td>
      <td>World</td>
      <td>Direct emissions (N2O)</td>
      <td>Agricultural Soils</td>
      <td>1964</td>
      <td>FAO TIER 1</td>
      <td>kilotonnes</td>
      <td>2004.8762</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emissions Totals</td>
      <td>World</td>
      <td>Direct emissions (N2O)</td>
      <td>Agricultural Soils</td>
      <td>1965</td>
      <td>FAO TIER 1</td>
      <td>kilotonnes</td>
      <td>2077.2466</td>
    </tr>
  </tbody>
</table>
</div>



To see the column names only I used the function given below. It makes it easy to select the coulmn names espacially when there are more number of columns.


```python
df.columns
```




    Index(['Domain', 'Area', 'Element', 'Item', 'Year', 'Source', 'Unit', 'Value'], dtype='object')



The required variables are stored into the new variables X and y. The X variable contains the indepenedant/feature variable, while the y variable contains the data of dependant/target variable.


```python
X=df[["Year"]]
y=df["Value"]
```

The model from the Scikit-learn library is fitted on the variables


```python
from sklearn.linear_model import LinearRegression
model1=LinearRegression().fit(X,y)
```

To evaluate the model efficiency/accuracy/fitness, the data is divided into training and testing data. The training data will be used to train the model and the prediction will be carried out on test data.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)
```

The model is trained using training data using the code given below


```python
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(X_train, y_train)
```

Prediction is carried out on test data to see the efficiency of the model and the R2 metrics is used to evaluate the model fitness.


```python
y_pred=model.predict(X_test)
y_pred
```




    array([2132.88430011, 2703.48269538, 4778.38595091, 4155.91497425,
           5452.72950896, 3585.31657898, 4363.40529981, 4259.66013703,
           6075.20048562, 6645.79888089, 2392.24720705, 3377.82625343,
           3740.93432315, 5297.1117648 , 2444.11978844, 2859.10043954,
           3274.08109065, 4934.00369508])




```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```




    0.9942678585164679



As we can see the accuracy of the model is 97% which is quite good. SO now it is confirmed that the model is good to predict the future variation in the data.\
We prepared an array of input variables that we want to predict.


```python
x=[[2020], [2021],[2022],[2023],[2024],[2025],[2026],[2027],[2028], [2029],[2030],[2031],[2032],[2033],[2034],[2035],[2036],[2037],
[2038],[2039],[2040],[2041],[2042],[2043],[2044],[2045],[2046],[2047],[2048],[2049],[2050]]
```

The array of generated input variables is feed to the model prediction function below and we can see the expected values as a output.


```python
exp1=model1.predict(x)
exp1
```

    C:\Python\lib\site-packages\sklearn\base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    




    array([5089.18632269, 5141.14791019, 5193.10949769, 5245.07108518,
           5297.03267268, 5348.99426018, 5400.95584767, 5452.91743517,
           5504.87902266, 5556.84061016, 5608.80219766, 5660.76378515,
           5712.72537265, 5764.68696014, 5816.64854764, 5868.61013514,
           5920.57172263, 5972.53331013, 6024.49489763, 6076.45648512,
           6128.41807262, 6180.37966011, 6232.34124761, 6284.30283511,
           6336.2644226 , 6388.2260101 , 6440.18759759, 6492.14918509,
           6544.11077259, 6596.07236008, 6648.03394758])



The predicted values are in the form of array, so to convert an array to a dataFrame, we used the function given below. The converted data frame is then exported to a CSV file for furthur use.


```python
df2=pd.DataFrame(exp1, columns=["Value"])
```


```python
df2.to_csv("C:/Users/BQD45O/Desktop/Git/meta/exp.csv")
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df2=pd.read_csv("emm_n2o.csv")
```


```python
plt.figure(figsize=(10,10))
Year=df2["Year"]
Value=df2["Value"]
sns.lineplot(x=Year, y=Value, data=df2, color="red", linewidth=5)
plt.axvline(x=2022, ymin=0, ymax=1, color="purple", linestyle="dotted", linewidth=4)
plt.figtext(0.7,0.08, "Prepared by: @Wajidumar007")
plt.figtext(0.65,0.63, "Expected N2O emission")
plt.xlabel("Year", size=12, weight="bold")
plt.ylabel("Nitrous oxide emission (Kilotonnes)", size=12, weight="bold")
plt.title("Nitrous Oxide Emission from  Agricultural soils (1961-2050)", size=14, weight="bold", family="Times New Roman")
```




    Text(0.5, 1.0, 'Nitrous Oxide Emission from  Agricultural soils (1961-2050)')




    
![png](output_29_1.png)
    

