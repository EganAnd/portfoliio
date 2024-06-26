# TLC Statistical Analysis with A/B Testing

#### The objective is to address the question, "Does payment tyoe affect the payment amount?".  That is to say do customers paying with credit cards pay higher fares than those who pay with cash? We will begin by importing the data which has been cleaned and explored earlier in a seperate notebook "EDA_TLC_Notebook".


## Import data and run descriptive statistics on data set.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
df = pd.read_csv('tlc_data_clean.csv')
```


```python
df.shape
```




    (22484, 22)




```python
df.head(4)
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
      <th>Unnamed: 0</th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>...</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>ride_duration</th>
      <th>month</th>
      <th>day</th>
      <th>total_fare_minus_tip</th>
      <th>dollars_per_mile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24870114</td>
      <td>2</td>
      <td>2017-03-25 08:55:43</td>
      <td>2017-03-25 09:09:47</td>
      <td>6</td>
      <td>3.34</td>
      <td>1</td>
      <td>100</td>
      <td>231</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>2.76</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>16.56</td>
      <td>14.066667</td>
      <td>March</td>
      <td>Saturday</td>
      <td>13.8</td>
      <td>4.131737</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35634249</td>
      <td>1</td>
      <td>2017-04-11 14:53:28</td>
      <td>2017-04-11 15:19:58</td>
      <td>1</td>
      <td>1.80</td>
      <td>1</td>
      <td>186</td>
      <td>43</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>4.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>20.80</td>
      <td>26.500000</td>
      <td>April</td>
      <td>Tuesday</td>
      <td>16.8</td>
      <td>9.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>106203690</td>
      <td>1</td>
      <td>2017-12-15 07:26:56</td>
      <td>2017-12-15 07:34:08</td>
      <td>1</td>
      <td>1.00</td>
      <td>1</td>
      <td>262</td>
      <td>236</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.45</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>8.75</td>
      <td>7.200000</td>
      <td>December</td>
      <td>Friday</td>
      <td>7.3</td>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38942136</td>
      <td>2</td>
      <td>2017-05-07 13:17:59</td>
      <td>2017-05-07 13:48:14</td>
      <td>1</td>
      <td>3.70</td>
      <td>1</td>
      <td>188</td>
      <td>97</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>6.39</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>27.69</td>
      <td>30.250000</td>
      <td>May</td>
      <td>Sunday</td>
      <td>21.3</td>
      <td>5.756757</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 22 columns</p>
</div>




```python
df.describe()
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
      <th>Unnamed: 0</th>
      <th>VendorID</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>ride_duration</th>
      <th>total_fare_minus_tip</th>
      <th>dollars_per_mile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.248400e+04</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>2.248400e+04</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
      <td>22484.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.675085e+07</td>
      <td>1.555862</td>
      <td>1.644636</td>
      <td>2.931917</td>
      <td>1.031667</td>
      <td>162.345223</td>
      <td>161.458548</td>
      <td>1.332147</td>
      <td>12.915278</td>
      <td>0.333571</td>
      <td>0.498666</td>
      <td>1.822558</td>
      <td>0.308747</td>
      <td>3.000000e-01</td>
      <td>16.184150</td>
      <td>14.428376</td>
      <td>14.361592</td>
      <td>7.338371</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.273495e+07</td>
      <td>0.496881</td>
      <td>1.284345</td>
      <td>3.655136</td>
      <td>0.229429</td>
      <td>66.596115</td>
      <td>70.096531</td>
      <td>0.489698</td>
      <td>10.805336</td>
      <td>0.461078</td>
      <td>0.025795</td>
      <td>2.427790</td>
      <td>1.384477</td>
      <td>1.169646e-13</td>
      <td>13.387148</td>
      <td>11.648379</td>
      <td>11.765239</td>
      <td>11.921433</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.212700e+04</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.010000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000e-01</td>
      <td>3.300000</td>
      <td>0.016667</td>
      <td>3.300000</td>
      <td>0.380000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.852071e+07</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>114.000000</td>
      <td>112.000000</td>
      <td>1.000000</td>
      <td>6.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000e-01</td>
      <td>8.750000</td>
      <td>6.716667</td>
      <td>7.800000</td>
      <td>4.777328</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.673992e+07</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.630000</td>
      <td>1.000000</td>
      <td>162.000000</td>
      <td>162.000000</td>
      <td>1.000000</td>
      <td>9.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>1.360000</td>
      <td>0.000000</td>
      <td>3.000000e-01</td>
      <td>11.800000</td>
      <td>11.233333</td>
      <td>10.300000</td>
      <td>6.266667</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.536705e+07</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.090000</td>
      <td>1.000000</td>
      <td>233.000000</td>
      <td>233.000000</td>
      <td>2.000000</td>
      <td>14.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>2.450000</td>
      <td>0.000000</td>
      <td>3.000000e-01</td>
      <td>17.800000</td>
      <td>18.383333</td>
      <td>15.800000</td>
      <td>8.192771</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.134863e+08</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>33.960000</td>
      <td>5.000000</td>
      <td>265.000000</td>
      <td>265.000000</td>
      <td>4.000000</td>
      <td>200.010000</td>
      <td>4.500000</td>
      <td>0.500000</td>
      <td>55.500000</td>
      <td>19.100000</td>
      <td>3.000000e-01</td>
      <td>258.210000</td>
      <td>209.166667</td>
      <td>206.570000</td>
      <td>693.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['payment_type'].value_counts()
```




    1    15181
    2     7175
    3       91
    4       37
    Name: payment_type, dtype: int64



### From the data dictionary, the values of the payment type column are
1 credit card
2 cash
3 no charge
4 dispute
5 unkown
6 voided trip

### There appears to be enough data in cash and credit card types to conduct a statistical analysis.

### Splitting the data into two dataframes, one for credit card one for cash.


```python
df_credit = df[df['payment_type']==1]
df_cash = df[df['payment_type']==2]
print(df_credit['total_amount'].mean())
print(df_cash['total_amount'].mean())
print(df_credit['total_amount'].std())
print(df_cash['total_amount'].std())


```

    17.45675844806008
    13.488652264808364
    14.106000040403663
    11.20747035993797


### There appears to be a difference in the mean total amount between the payment types.

## Statistical Test

### The null hypothesis is that the payment types have the same fare amounts.
### The alternative hypothesis is that the credit card payment is greater than the cash.
### The significance level will be 5%


```python
stats.ttest_ind(a= df_credit['total_amount'], b = df_cash['total_amount'], alternative = 'greater',equal_var=False)
```




    Ttest_indResult(statistic=22.679195489830533, pvalue=1.5350085299369956e-112)



### The t-test pvalue is much smaller than the significance level of 5%.  We reject the null hypothesis, and believe that there is a difference in the total amounts based upon the payment type.

### Validating this reult by performing the t-test on both the total fare minus tip value and fare amount values.


```python
stats.ttest_ind(a= df_credit['total_fare_minus_tip'], b = df_cash['total_fare_minus_tip'], alternative = 'greater',equal_var=False)
```




    Ttest_indResult(statistic=7.731607209040485, pvalue=5.647528399886761e-15)




```python
stats.ttest_ind(a= df_credit['fare_amount'], b = df_cash['fare_amount'], alternative = 'greater',equal_var=False)
```




    Ttest_indResult(statistic=7.287937937692885, pvalue=1.6537660414263434e-13)



### All of the tests suggest that credit card fares are greater than cash fares.


```python

```
