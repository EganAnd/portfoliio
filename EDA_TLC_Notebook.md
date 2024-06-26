```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')
```

## General exploration and Analysis


```python
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
      <th>Unnamed: 0</th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24870114</td>
      <td>2</td>
      <td>03/25/2017 8:55:43 AM</td>
      <td>03/25/2017 9:09:47 AM</td>
      <td>6</td>
      <td>3.34</td>
      <td>1</td>
      <td>N</td>
      <td>100</td>
      <td>231</td>
      <td>1</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.76</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>16.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35634249</td>
      <td>1</td>
      <td>04/11/2017 2:53:28 PM</td>
      <td>04/11/2017 3:19:58 PM</td>
      <td>1</td>
      <td>1.80</td>
      <td>1</td>
      <td>N</td>
      <td>186</td>
      <td>43</td>
      <td>1</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>4.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>20.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>106203690</td>
      <td>1</td>
      <td>12/15/2017 7:26:56 AM</td>
      <td>12/15/2017 7:34:08 AM</td>
      <td>1</td>
      <td>1.00</td>
      <td>1</td>
      <td>N</td>
      <td>262</td>
      <td>236</td>
      <td>1</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.45</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>8.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38942136</td>
      <td>2</td>
      <td>05/07/2017 1:17:59 PM</td>
      <td>05/07/2017 1:48:14 PM</td>
      <td>1</td>
      <td>3.70</td>
      <td>1</td>
      <td>N</td>
      <td>188</td>
      <td>97</td>
      <td>1</td>
      <td>20.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>6.39</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>27.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30841670</td>
      <td>2</td>
      <td>04/15/2017 11:32:20 PM</td>
      <td>04/15/2017 11:49:03 PM</td>
      <td>1</td>
      <td>4.37</td>
      <td>1</td>
      <td>N</td>
      <td>4</td>
      <td>112</td>
      <td>2</td>
      <td>16.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>17.80</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (22699, 18)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22699 entries, 0 to 22698
    Data columns (total 18 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   Unnamed: 0             22699 non-null  int64  
     1   VendorID               22699 non-null  int64  
     2   tpep_pickup_datetime   22699 non-null  object 
     3   tpep_dropoff_datetime  22699 non-null  object 
     4   passenger_count        22699 non-null  int64  
     5   trip_distance          22699 non-null  float64
     6   RatecodeID             22699 non-null  int64  
     7   store_and_fwd_flag     22699 non-null  object 
     8   PULocationID           22699 non-null  int64  
     9   DOLocationID           22699 non-null  int64  
     10  payment_type           22699 non-null  int64  
     11  fare_amount            22699 non-null  float64
     12  extra                  22699 non-null  float64
     13  mta_tax                22699 non-null  float64
     14  tip_amount             22699 non-null  float64
     15  tolls_amount           22699 non-null  float64
     16  improvement_surcharge  22699 non-null  float64
     17  total_amount           22699 non-null  float64
    dtypes: float64(8), int64(7), object(3)
    memory usage: 3.1+ MB


### There are no NaN n this dataset, datetimes are objects not datetime dtypes


```python
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
```


```python
df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22699 entries, 0 to 22698
    Data columns (total 18 columns):
     #   Column                 Non-Null Count  Dtype         
    ---  ------                 --------------  -----         
     0   Unnamed: 0             22699 non-null  int64         
     1   VendorID               22699 non-null  int64         
     2   tpep_pickup_datetime   22699 non-null  datetime64[ns]
     3   tpep_dropoff_datetime  22699 non-null  datetime64[ns]
     4   passenger_count        22699 non-null  int64         
     5   trip_distance          22699 non-null  float64       
     6   RatecodeID             22699 non-null  int64         
     7   store_and_fwd_flag     22699 non-null  object        
     8   PULocationID           22699 non-null  int64         
     9   DOLocationID           22699 non-null  int64         
     10  payment_type           22699 non-null  int64         
     11  fare_amount            22699 non-null  float64       
     12  extra                  22699 non-null  float64       
     13  mta_tax                22699 non-null  float64       
     14  tip_amount             22699 non-null  float64       
     15  tolls_amount           22699 non-null  float64       
     16  improvement_surcharge  22699 non-null  float64       
     17  total_amount           22699 non-null  float64       
    dtypes: datetime64[ns](2), float64(8), int64(7), object(1)
    memory usage: 3.1+ MB


## Creation of trip duration in minutes


```python
#create duration of ride field
df['ride_duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df['ride_duration'] = df['ride_duration'].dt.total_seconds()/60
```

## Initial analysis of most important variables

### Trip distance (trip_distance']


```python
plt.figure(figsize=(6,1))
plt.title('trip_distance')
sns.boxplot(data=None, x = df['trip_distance'], fliersize=1)
```




    <Axes: title={'center': 'trip_distance'}, xlabel='trip_distance'>




    
![png](output_12_1.png)
    



```python
df_distance_zero = df[df['trip_distance'] <= 0].sort_values(by='ride_duration')
df_distance_zero
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
      <th>store_and_fwd_flag</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17567</th>
      <td>34210304</td>
      <td>1</td>
      <td>2017-04-25 13:16:31</td>
      <td>2017-04-25 13:16:31</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>132</td>
      <td>264</td>
      <td>2</td>
      <td>62.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>62.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17311</th>
      <td>19238418</td>
      <td>1</td>
      <td>2017-03-07 18:16:47</td>
      <td>2017-03-07 18:16:47</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>162</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>18394854</td>
      <td>1</td>
      <td>2017-03-05 06:41:16</td>
      <td>2017-03-05 06:41:16</td>
      <td>1</td>
      <td>0.0</td>
      <td>5</td>
      <td>N</td>
      <td>233</td>
      <td>264</td>
      <td>2</td>
      <td>80.84</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>81.14</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17270</th>
      <td>20458610</td>
      <td>1</td>
      <td>2017-03-26 22:48:51</td>
      <td>2017-03-26 22:48:51</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>170</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16586</th>
      <td>87350785</td>
      <td>1</td>
      <td>2017-10-17 04:39:44</td>
      <td>2017-10-17 04:39:44</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>145</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3865</th>
      <td>53536625</td>
      <td>2</td>
      <td>2017-06-23 14:43:42</td>
      <td>2017-06-23 14:51:15</td>
      <td>6</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>13</td>
      <td>13</td>
      <td>2</td>
      <td>6.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>6.80</td>
      <td>7.550000</td>
    </tr>
    <tr>
      <th>22043</th>
      <td>57581462</td>
      <td>1</td>
      <td>2017-07-06 10:27:11</td>
      <td>2017-07-06 10:40:07</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>N</td>
      <td>162</td>
      <td>162</td>
      <td>3</td>
      <td>52.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>52.80</td>
      <td>12.933333</td>
    </tr>
    <tr>
      <th>15916</th>
      <td>47368116</td>
      <td>1</td>
      <td>2017-06-29 19:30:30</td>
      <td>2017-06-29 19:43:29</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>79</td>
      <td>148</td>
      <td>3</td>
      <td>8.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>10.30</td>
      <td>12.983333</td>
    </tr>
    <tr>
      <th>20080</th>
      <td>55620713</td>
      <td>2</td>
      <td>2017-06-07 10:27:54</td>
      <td>2017-06-07 10:54:23</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>107</td>
      <td>237</td>
      <td>1</td>
      <td>15.50</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>16.30</td>
      <td>26.483333</td>
    </tr>
    <tr>
      <th>18116</th>
      <td>2677141</td>
      <td>1</td>
      <td>2017-01-10 22:03:42</td>
      <td>2017-01-10 22:35:49</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>234</td>
      <td>61</td>
      <td>1</td>
      <td>29.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>6.15</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>36.95</td>
      <td>32.116667</td>
    </tr>
  </tbody>
</table>
<p>148 rows × 19 columns</p>
</div>




```python
df_distance_long= df[df['trip_distance']>20].sort_values(by='trip_distance', ascending = False)
df_distance_long
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
      <th>store_and_fwd_flag</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9280</th>
      <td>51810714</td>
      <td>2</td>
      <td>2017-06-18 23:33:25</td>
      <td>2017-06-19 00:12:38</td>
      <td>2</td>
      <td>33.96</td>
      <td>5</td>
      <td>N</td>
      <td>132</td>
      <td>265</td>
      <td>2</td>
      <td>150.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>150.30</td>
      <td>39.216667</td>
    </tr>
    <tr>
      <th>13861</th>
      <td>40523668</td>
      <td>2</td>
      <td>2017-05-19 08:20:21</td>
      <td>2017-05-19 09:20:30</td>
      <td>1</td>
      <td>33.92</td>
      <td>5</td>
      <td>N</td>
      <td>229</td>
      <td>265</td>
      <td>1</td>
      <td>200.01</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>51.64</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>258.21</td>
      <td>60.150000</td>
    </tr>
    <tr>
      <th>6064</th>
      <td>49894023</td>
      <td>2</td>
      <td>2017-06-13 12:30:22</td>
      <td>2017-06-13 13:37:51</td>
      <td>1</td>
      <td>32.72</td>
      <td>3</td>
      <td>N</td>
      <td>138</td>
      <td>1</td>
      <td>1</td>
      <td>107.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55.50</td>
      <td>16.26</td>
      <td>0.3</td>
      <td>179.06</td>
      <td>67.483333</td>
    </tr>
    <tr>
      <th>10291</th>
      <td>76319330</td>
      <td>2</td>
      <td>2017-09-11 11:41:04</td>
      <td>2017-09-11 12:18:58</td>
      <td>1</td>
      <td>31.95</td>
      <td>4</td>
      <td>N</td>
      <td>138</td>
      <td>265</td>
      <td>2</td>
      <td>131.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>131.80</td>
      <td>37.900000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>94052446</td>
      <td>2</td>
      <td>2017-11-06 20:30:50</td>
      <td>2017-11-07 00:00:00</td>
      <td>1</td>
      <td>30.83</td>
      <td>1</td>
      <td>N</td>
      <td>132</td>
      <td>23</td>
      <td>1</td>
      <td>80.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>18.56</td>
      <td>11.52</td>
      <td>0.3</td>
      <td>111.38</td>
      <td>209.166667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8778</th>
      <td>74763608</td>
      <td>1</td>
      <td>2017-09-06 13:44:43</td>
      <td>2017-09-06 14:36:38</td>
      <td>1</td>
      <td>20.20</td>
      <td>1</td>
      <td>N</td>
      <td>132</td>
      <td>37</td>
      <td>1</td>
      <td>58.50</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>11.85</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>71.15</td>
      <td>51.916667</td>
    </tr>
    <tr>
      <th>19872</th>
      <td>24876527</td>
      <td>1</td>
      <td>2017-03-25 09:33:54</td>
      <td>2017-03-25 10:08:49</td>
      <td>1</td>
      <td>20.20</td>
      <td>2</td>
      <td>N</td>
      <td>209</td>
      <td>132</td>
      <td>1</td>
      <td>52.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>11.70</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>70.26</td>
      <td>34.916667</td>
    </tr>
    <tr>
      <th>19664</th>
      <td>80299197</td>
      <td>2</td>
      <td>2017-09-24 11:22:07</td>
      <td>2017-09-24 12:02:21</td>
      <td>2</td>
      <td>20.11</td>
      <td>2</td>
      <td>N</td>
      <td>132</td>
      <td>263</td>
      <td>1</td>
      <td>52.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>11.71</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>70.27</td>
      <td>40.233333</td>
    </tr>
    <tr>
      <th>8470</th>
      <td>80749449</td>
      <td>2</td>
      <td>2017-09-25 21:52:24</td>
      <td>2017-09-25 22:32:00</td>
      <td>1</td>
      <td>20.08</td>
      <td>2</td>
      <td>N</td>
      <td>132</td>
      <td>239</td>
      <td>2</td>
      <td>52.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>58.56</td>
      <td>39.600000</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>45088183</td>
      <td>2</td>
      <td>2017-05-25 22:18:05</td>
      <td>2017-05-25 22:52:09</td>
      <td>6</td>
      <td>20.03</td>
      <td>2</td>
      <td>N</td>
      <td>132</td>
      <td>140</td>
      <td>2</td>
      <td>52.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>58.56</td>
      <td>34.066667</td>
    </tr>
  </tbody>
</table>
<p>136 rows × 19 columns</p>
</div>




```python
plt.figure(figsize=(10,5))
sns.histplot(df['trip_distance'], bins= range(0, 26, 1))
plt.title('trip_distance')

```




    Text(0.5, 1.0, 'trip_distance')




    
![png](output_15_1.png)
    


### Analysis of trip_distance variable:  There are a number of outliers on he high end as well as 19 rows with a trip distance = 0.  These rows will be addressed in cleaning.

## Ride Duration('ride_duration')


```python
df_duration_less_than_zero = df[df['ride_duration']<=0].sort_values(by='ride_duration', ascending=False)
df_duration_less_than_zero
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
      <th>store_and_fwd_flag</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>472</th>
      <td>52474677</td>
      <td>1</td>
      <td>2017-06-20 18:57:39</td>
      <td>2017-06-20 18:57:39</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>162</td>
      <td>264</td>
      <td>2</td>
      <td>9.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>11.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13062</th>
      <td>85154968</td>
      <td>1</td>
      <td>2017-10-10 09:53:00</td>
      <td>2017-10-10 09:53:00</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>186</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21977</th>
      <td>67022415</td>
      <td>1</td>
      <td>2017-08-08 07:28:47</td>
      <td>2017-08-08 07:28:47</td>
      <td>2</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>75</td>
      <td>264</td>
      <td>2</td>
      <td>10.50</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>11.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21836</th>
      <td>65082578</td>
      <td>2</td>
      <td>2017-08-01 09:52:15</td>
      <td>2017-08-01 09:52:15</td>
      <td>3</td>
      <td>0.0</td>
      <td>5</td>
      <td>N</td>
      <td>264</td>
      <td>143</td>
      <td>1</td>
      <td>59.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>68.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21620</th>
      <td>19232159</td>
      <td>1</td>
      <td>2017-03-07 17:58:40</td>
      <td>2017-03-07 17:58:40</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>237</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>106367018</td>
      <td>1</td>
      <td>2017-12-15 16:09:43</td>
      <td>2017-12-15 16:09:43</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>43</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19838</th>
      <td>80341574</td>
      <td>1</td>
      <td>2017-09-24 13:37:55</td>
      <td>2017-09-24 13:37:55</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>234</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17567</th>
      <td>34210304</td>
      <td>1</td>
      <td>2017-04-25 13:16:31</td>
      <td>2017-04-25 13:16:31</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>132</td>
      <td>264</td>
      <td>2</td>
      <td>62.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>62.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17311</th>
      <td>19238418</td>
      <td>1</td>
      <td>2017-03-07 18:16:47</td>
      <td>2017-03-07 18:16:47</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>162</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17270</th>
      <td>20458610</td>
      <td>1</td>
      <td>2017-03-26 22:48:51</td>
      <td>2017-03-26 22:48:51</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>170</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16586</th>
      <td>87350785</td>
      <td>1</td>
      <td>2017-10-17 04:39:44</td>
      <td>2017-10-17 04:39:44</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>145</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15295</th>
      <td>83531047</td>
      <td>1</td>
      <td>2017-10-04 22:07:50</td>
      <td>2017-10-04 22:07:50</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>143</td>
      <td>264</td>
      <td>2</td>
      <td>6.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>7.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14062</th>
      <td>25713867</td>
      <td>1</td>
      <td>2017-03-30 19:56:31</td>
      <td>2017-03-30 19:56:31</td>
      <td>3</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>113</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12501</th>
      <td>31724098</td>
      <td>1</td>
      <td>2017-04-18 17:47:58</td>
      <td>2017-04-18 17:47:58</td>
      <td>0</td>
      <td>0.0</td>
      <td>99</td>
      <td>N</td>
      <td>264</td>
      <td>264</td>
      <td>1</td>
      <td>77.20</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>78.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>795</th>
      <td>101135030</td>
      <td>1</td>
      <td>2017-11-30 07:11:34</td>
      <td>2017-11-30 07:11:34</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>246</td>
      <td>264</td>
      <td>2</td>
      <td>8.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>8.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9155</th>
      <td>101666430</td>
      <td>1</td>
      <td>2017-12-01 18:41:19</td>
      <td>2017-12-01 18:41:19</td>
      <td>4</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>163</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8889</th>
      <td>25813</td>
      <td>1</td>
      <td>2017-01-07 22:48:08</td>
      <td>2017-01-07 22:48:08</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>229</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8839</th>
      <td>9033526</td>
      <td>1</td>
      <td>2017-02-08 17:54:50</td>
      <td>2017-02-08 17:54:50</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>164</td>
      <td>264</td>
      <td>2</td>
      <td>18.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>20.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7327</th>
      <td>19562548</td>
      <td>1</td>
      <td>2017-03-08 16:11:57</td>
      <td>2017-03-08 16:11:57</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>233</td>
      <td>264</td>
      <td>3</td>
      <td>24.00</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>25.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5630</th>
      <td>6520188</td>
      <td>1</td>
      <td>2017-01-29 20:16:21</td>
      <td>2017-01-29 20:16:21</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>132</td>
      <td>264</td>
      <td>2</td>
      <td>39.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>40.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>18394854</td>
      <td>1</td>
      <td>2017-03-05 06:41:16</td>
      <td>2017-03-05 06:41:16</td>
      <td>1</td>
      <td>0.0</td>
      <td>5</td>
      <td>N</td>
      <td>233</td>
      <td>264</td>
      <td>2</td>
      <td>80.84</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>81.14</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4171</th>
      <td>12399699</td>
      <td>1</td>
      <td>2017-02-16 20:37:04</td>
      <td>2017-02-16 20:37:04</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>238</td>
      <td>264</td>
      <td>2</td>
      <td>9.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>10.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2740</th>
      <td>42450170</td>
      <td>1</td>
      <td>2017-05-12 12:49:56</td>
      <td>2017-05-12 12:49:56</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>Y</td>
      <td>186</td>
      <td>264</td>
      <td>2</td>
      <td>11.50</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>12.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2440</th>
      <td>63574825</td>
      <td>1</td>
      <td>2017-07-26 22:26:58</td>
      <td>2017-07-26 22:26:58</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>162</td>
      <td>264</td>
      <td>2</td>
      <td>5.50</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>6.80</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1205</th>
      <td>112363821</td>
      <td>1</td>
      <td>2017-01-18 17:53:45</td>
      <td>2017-01-18 17:53:45</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>264</td>
      <td>264</td>
      <td>2</td>
      <td>2.50</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22026</th>
      <td>63642923</td>
      <td>1</td>
      <td>2017-07-27 07:44:24</td>
      <td>2017-07-27 07:44:24</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>N</td>
      <td>41</td>
      <td>264</td>
      <td>2</td>
      <td>10.50</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>11.30</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9356</th>
      <td>93542707</td>
      <td>1</td>
      <td>2017-11-05 01:23:08</td>
      <td>2017-11-05 01:06:09</td>
      <td>1</td>
      <td>5.7</td>
      <td>1</td>
      <td>N</td>
      <td>161</td>
      <td>157</td>
      <td>3</td>
      <td>28.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>29.30</td>
      <td>-16.983333</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(6,1))
plt.title('ride_duration')
sns.boxplot(data=None, x = df['ride_duration'], fliersize=1)
```




    <Axes: title={'center': 'ride_duration'}, xlabel='ride_duration'>




    
![png](output_19_1.png)
    



```python
plt.scatter(x=df['ride_duration'], y = df['total_amount'])
plt.show
```




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_20_1.png)
    


### The data for ride duration shows a number of outlier both great and small and their inaccuracy can be exemplified by correlating to fare amounts.

## Fare Amount ('fare_amount') and Tip Amount('tip_amount')


```python
plt.figure(figsize=(6,1))
plt.title('fare_amount')
sns.boxplot(data=None, x = df['fare_amount'], fliersize=1)
```




    <Axes: title={'center': 'fare_amount'}, xlabel='fare_amount'>




    
![png](output_23_1.png)
    



```python
plt.figure(figsize=(10,5))
ax = sns.histplot(df['fare_amount'], bins= range(-10, 101, 5))
ax.set_xticks(range(-10,101,5))
ax.set_xticklabels(range(-10,101,5))
plt.title('fare_amount_histogram')
```




    Text(0.5, 1.0, 'fare_amount_histogram')




    
![png](output_24_1.png)
    



```python
plt.figure(figsize=(6,1))
plt.title('tip_amount')
sns.boxplot(data=None, x = df['tip_amount'], fliersize=1)
```




    <Axes: title={'center': 'tip_amount'}, xlabel='tip_amount'>




    
![png](output_25_1.png)
    



```python
plt.figure(figsize=(14,8))
ax = sns.histplot(data=df, x='tip_amount', bins=range(0,21,1), 
                  hue='VendorID', 
                  multiple='stack',
                  palette='Oranges')
ax.set_xticks(range(0,21,1))
ax.set_xticklabels(range(0,21,1))
plt.title('Tip amount by Vendor Histogram');
```


    
![png](output_26_0.png)
    



```python
df_fare_zero = df[df['fare_amount']==0].sort_values(by= 'trip_distance', ascending=False)
df_fare_zero
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
      <th>store_and_fwd_flag</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21842</th>
      <td>31708083</td>
      <td>1</td>
      <td>2017-04-18 16:55:29</td>
      <td>2017-04-18 18:29:44</td>
      <td>2</td>
      <td>20.40</td>
      <td>5</td>
      <td>N</td>
      <td>264</td>
      <td>264</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.5</td>
      <td>0.3</td>
      <td>12.8</td>
      <td>94.250000</td>
    </tr>
    <tr>
      <th>4402</th>
      <td>108016954</td>
      <td>2</td>
      <td>2017-12-20 16:06:53</td>
      <td>2017-12-20 16:47:50</td>
      <td>1</td>
      <td>7.06</td>
      <td>1</td>
      <td>N</td>
      <td>263</td>
      <td>169</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.950000</td>
    </tr>
    <tr>
      <th>19067</th>
      <td>58713019</td>
      <td>1</td>
      <td>2017-07-10 14:40:09</td>
      <td>2017-07-10 14:40:59</td>
      <td>1</td>
      <td>0.10</td>
      <td>5</td>
      <td>N</td>
      <td>261</td>
      <td>13</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>5722</th>
      <td>49670364</td>
      <td>2</td>
      <td>2017-06-12 12:08:55</td>
      <td>2017-06-12 12:08:57</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>N</td>
      <td>264</td>
      <td>193</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.033333</td>
    </tr>
    <tr>
      <th>10506</th>
      <td>26005024</td>
      <td>2</td>
      <td>2017-03-30 03:14:26</td>
      <td>2017-03-30 03:14:28</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>N</td>
      <td>264</td>
      <td>193</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.033333</td>
    </tr>
    <tr>
      <th>22566</th>
      <td>19022898</td>
      <td>2</td>
      <td>2017-03-07 02:24:47</td>
      <td>2017-03-07 02:24:50</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>N</td>
      <td>264</td>
      <td>193</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.050000</td>
    </tr>
  </tbody>
</table>
</div>



### Analysis:  There are again a number of outliers in this field as well as a number of zero values.  These will be addressed in cleaning.

## Initial Analysis Passenger Counts('passenger_count')


```python
df['passenger_count'].value_counts()
```




    1    16117
    2     3305
    5     1143
    3      953
    6      693
    4      455
    0       33
    Name: passenger_count, dtype: int64




```python
df_passenger_zero = df[df['passenger_count'] ==0].sort_values(by='ride_duration', ascending=False)
df_passenger_zero
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
      <th>store_and_fwd_flag</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1516</th>
      <td>96563556</td>
      <td>1</td>
      <td>2017-11-14 15:45:23</td>
      <td>2017-11-14 16:26:38</td>
      <td>0</td>
      <td>8.8</td>
      <td>1</td>
      <td>N</td>
      <td>138</td>
      <td>164</td>
      <td>1</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>5.55</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>45.11</td>
      <td>41.250000</td>
    </tr>
    <tr>
      <th>5767</th>
      <td>83401081</td>
      <td>1</td>
      <td>2017-10-04 15:17:52</td>
      <td>2017-10-04 15:56:26</td>
      <td>0</td>
      <td>10.4</td>
      <td>1</td>
      <td>N</td>
      <td>113</td>
      <td>138</td>
      <td>1</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>8.10</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>48.66</td>
      <td>38.566667</td>
    </tr>
    <tr>
      <th>14527</th>
      <td>96810412</td>
      <td>1</td>
      <td>2017-11-15 10:00:37</td>
      <td>2017-11-15 10:24:43</td>
      <td>0</td>
      <td>5.5</td>
      <td>1</td>
      <td>N</td>
      <td>161</td>
      <td>87</td>
      <td>1</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.50</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>23.30</td>
      <td>24.100000</td>
    </tr>
    <tr>
      <th>19217</th>
      <td>92734995</td>
      <td>1</td>
      <td>2017-11-02 19:01:54</td>
      <td>2017-11-02 19:25:31</td>
      <td>0</td>
      <td>2.8</td>
      <td>1</td>
      <td>N</td>
      <td>262</td>
      <td>142</td>
      <td>2</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>17.80</td>
      <td>23.616667</td>
    </tr>
    <tr>
      <th>4919</th>
      <td>106693550</td>
      <td>1</td>
      <td>2017-12-16 14:59:17</td>
      <td>2017-12-16 15:21:59</td>
      <td>0</td>
      <td>3.1</td>
      <td>1</td>
      <td>N</td>
      <td>163</td>
      <td>114</td>
      <td>1</td>
      <td>15.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>4.05</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>20.35</td>
      <td>22.700000</td>
    </tr>
    <tr>
      <th>5603</th>
      <td>107019016</td>
      <td>1</td>
      <td>2017-12-17 12:18:49</td>
      <td>2017-12-17 12:40:45</td>
      <td>0</td>
      <td>4.2</td>
      <td>1</td>
      <td>N</td>
      <td>230</td>
      <td>211</td>
      <td>1</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>3.75</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>22.55</td>
      <td>21.933333</td>
    </tr>
    <tr>
      <th>8595</th>
      <td>96507020</td>
      <td>1</td>
      <td>2017-11-14 12:06:01</td>
      <td>2017-11-14 12:27:38</td>
      <td>0</td>
      <td>7.0</td>
      <td>1</td>
      <td>N</td>
      <td>162</td>
      <td>13</td>
      <td>1</td>
      <td>23.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>6.05</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>30.35</td>
      <td>21.616667</td>
    </tr>
    <tr>
      <th>19456</th>
      <td>90729144</td>
      <td>1</td>
      <td>2017-10-27 14:11:07</td>
      <td>2017-10-27 14:32:17</td>
      <td>0</td>
      <td>2.7</td>
      <td>1</td>
      <td>N</td>
      <td>231</td>
      <td>186</td>
      <td>1</td>
      <td>14.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.50</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>17.80</td>
      <td>21.166667</td>
    </tr>
    <tr>
      <th>21752</th>
      <td>91222179</td>
      <td>1</td>
      <td>2017-10-28 22:25:02</td>
      <td>2017-10-28 22:43:44</td>
      <td>0</td>
      <td>1.6</td>
      <td>1</td>
      <td>N</td>
      <td>113</td>
      <td>246</td>
      <td>1</td>
      <td>12.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>3.45</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>17.25</td>
      <td>18.700000</td>
    </tr>
    <tr>
      <th>10145</th>
      <td>102526701</td>
      <td>1</td>
      <td>2017-12-04 10:41:30</td>
      <td>2017-12-04 10:58:23</td>
      <td>0</td>
      <td>2.0</td>
      <td>1</td>
      <td>N</td>
      <td>142</td>
      <td>263</td>
      <td>1</td>
      <td>12.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>3.30</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>16.60</td>
      <td>16.883333</td>
    </tr>
    <tr>
      <th>13919</th>
      <td>93579534</td>
      <td>1</td>
      <td>2017-11-05 03:10:59</td>
      <td>2017-11-05 03:26:38</td>
      <td>0</td>
      <td>2.3</td>
      <td>1</td>
      <td>N</td>
      <td>79</td>
      <td>261</td>
      <td>1</td>
      <td>11.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>14.80</td>
      <td>15.650000</td>
    </tr>
    <tr>
      <th>3352</th>
      <td>107464306</td>
      <td>1</td>
      <td>2017-12-18 22:44:57</td>
      <td>2017-12-18 23:00:24</td>
      <td>0</td>
      <td>2.7</td>
      <td>1</td>
      <td>N</td>
      <td>87</td>
      <td>79</td>
      <td>1</td>
      <td>12.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.75</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>16.55</td>
      <td>15.450000</td>
    </tr>
    <tr>
      <th>17914</th>
      <td>107896724</td>
      <td>1</td>
      <td>2017-12-20 09:42:39</td>
      <td>2017-12-20 09:57:47</td>
      <td>0</td>
      <td>0.8</td>
      <td>1</td>
      <td>N</td>
      <td>229</td>
      <td>162</td>
      <td>1</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.15</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>12.95</td>
      <td>15.133333</td>
    </tr>
    <tr>
      <th>7102</th>
      <td>110296406</td>
      <td>1</td>
      <td>2017-12-29 13:41:17</td>
      <td>2017-12-29 13:55:57</td>
      <td>0</td>
      <td>2.1</td>
      <td>1</td>
      <td>N</td>
      <td>264</td>
      <td>264</td>
      <td>1</td>
      <td>11.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.45</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>14.75</td>
      <td>14.666667</td>
    </tr>
    <tr>
      <th>13921</th>
      <td>97593329</td>
      <td>1</td>
      <td>2017-11-17 16:39:02</td>
      <td>2017-11-17 16:52:38</td>
      <td>0</td>
      <td>0.9</td>
      <td>1</td>
      <td>N</td>
      <td>236</td>
      <td>141</td>
      <td>1</td>
      <td>9.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>2.25</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>13.55</td>
      <td>13.600000</td>
    </tr>
    <tr>
      <th>12802</th>
      <td>79604871</td>
      <td>1</td>
      <td>2017-09-22 06:49:25</td>
      <td>2017-09-22 07:01:57</td>
      <td>0</td>
      <td>2.9</td>
      <td>1</td>
      <td>N</td>
      <td>87</td>
      <td>234</td>
      <td>1</td>
      <td>11.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.45</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>14.75</td>
      <td>12.533333</td>
    </tr>
    <tr>
      <th>5563</th>
      <td>74279671</td>
      <td>1</td>
      <td>2017-09-04 17:40:00</td>
      <td>2017-09-04 17:51:52</td>
      <td>0</td>
      <td>1.3</td>
      <td>1</td>
      <td>N</td>
      <td>164</td>
      <td>233</td>
      <td>1</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.95</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>11.75</td>
      <td>11.866667</td>
    </tr>
    <tr>
      <th>21638</th>
      <td>98705879</td>
      <td>1</td>
      <td>2017-11-21 05:47:11</td>
      <td>2017-11-21 05:58:23</td>
      <td>0</td>
      <td>2.2</td>
      <td>1</td>
      <td>N</td>
      <td>238</td>
      <td>230</td>
      <td>1</td>
      <td>10.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.36</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>14.16</td>
      <td>11.200000</td>
    </tr>
    <tr>
      <th>9828</th>
      <td>84313191</td>
      <td>1</td>
      <td>2017-10-07 10:02:35</td>
      <td>2017-10-07 10:13:34</td>
      <td>0</td>
      <td>6.0</td>
      <td>1</td>
      <td>N</td>
      <td>140</td>
      <td>88</td>
      <td>1</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>3.75</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>22.55</td>
      <td>10.983333</td>
    </tr>
    <tr>
      <th>5668</th>
      <td>106785624</td>
      <td>1</td>
      <td>2017-12-16 19:50:12</td>
      <td>2017-12-16 20:00:30</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>N</td>
      <td>211</td>
      <td>249</td>
      <td>1</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.50</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>10.30</td>
      <td>10.300000</td>
    </tr>
    <tr>
      <th>4060</th>
      <td>100326273</td>
      <td>1</td>
      <td>2017-11-27 13:08:01</td>
      <td>2017-11-27 13:17:55</td>
      <td>0</td>
      <td>1.7</td>
      <td>1</td>
      <td>N</td>
      <td>161</td>
      <td>239</td>
      <td>1</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.95</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>11.75</td>
      <td>9.900000</td>
    </tr>
    <tr>
      <th>10199</th>
      <td>78367526</td>
      <td>1</td>
      <td>2017-09-17 17:05:16</td>
      <td>2017-09-17 17:13:51</td>
      <td>0</td>
      <td>0.9</td>
      <td>1</td>
      <td>N</td>
      <td>236</td>
      <td>237</td>
      <td>1</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>8.80</td>
      <td>8.583333</td>
    </tr>
    <tr>
      <th>20310</th>
      <td>108220152</td>
      <td>1</td>
      <td>2017-12-21 08:55:39</td>
      <td>2017-12-21 09:04:03</td>
      <td>0</td>
      <td>1.1</td>
      <td>1</td>
      <td>N</td>
      <td>113</td>
      <td>90</td>
      <td>1</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.55</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>9.35</td>
      <td>8.400000</td>
    </tr>
    <tr>
      <th>1874</th>
      <td>102273749</td>
      <td>1</td>
      <td>2017-12-03 12:05:52</td>
      <td>2017-12-03 12:13:51</td>
      <td>0</td>
      <td>1.1</td>
      <td>1</td>
      <td>N</td>
      <td>48</td>
      <td>237</td>
      <td>2</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>7.80</td>
      <td>7.983333</td>
    </tr>
    <tr>
      <th>12203</th>
      <td>97131247</td>
      <td>1</td>
      <td>2017-11-16 09:42:47</td>
      <td>2017-11-16 09:50:04</td>
      <td>0</td>
      <td>1.1</td>
      <td>1</td>
      <td>N</td>
      <td>107</td>
      <td>79</td>
      <td>2</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>7.30</td>
      <td>7.283333</td>
    </tr>
    <tr>
      <th>18309</th>
      <td>101059790</td>
      <td>1</td>
      <td>2017-11-29 21:57:19</td>
      <td>2017-11-29 22:04:13</td>
      <td>0</td>
      <td>0.7</td>
      <td>1</td>
      <td>N</td>
      <td>50</td>
      <td>100</td>
      <td>2</td>
      <td>6.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>7.80</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>13419</th>
      <td>107412151</td>
      <td>1</td>
      <td>2017-12-18 19:32:14</td>
      <td>2017-12-18 19:39:07</td>
      <td>0</td>
      <td>2.4</td>
      <td>1</td>
      <td>N</td>
      <td>68</td>
      <td>50</td>
      <td>1</td>
      <td>8.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>11.30</td>
      <td>6.883333</td>
    </tr>
    <tr>
      <th>14519</th>
      <td>109119545</td>
      <td>1</td>
      <td>2017-12-24 09:02:00</td>
      <td>2017-12-24 09:07:56</td>
      <td>0</td>
      <td>0.9</td>
      <td>1</td>
      <td>N</td>
      <td>75</td>
      <td>74</td>
      <td>1</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.35</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>8.15</td>
      <td>5.933333</td>
    </tr>
    <tr>
      <th>21216</th>
      <td>109673439</td>
      <td>1</td>
      <td>2017-12-27 06:21:58</td>
      <td>2017-12-27 06:27:32</td>
      <td>0</td>
      <td>1.3</td>
      <td>1</td>
      <td>N</td>
      <td>237</td>
      <td>262</td>
      <td>2</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>7.30</td>
      <td>5.566667</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>75654951</td>
      <td>1</td>
      <td>2017-09-09 03:44:45</td>
      <td>2017-09-09 03:49:19</td>
      <td>0</td>
      <td>0.8</td>
      <td>1</td>
      <td>N</td>
      <td>48</td>
      <td>48</td>
      <td>2</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>6.30</td>
      <td>4.566667</td>
    </tr>
    <tr>
      <th>13477</th>
      <td>105179313</td>
      <td>1</td>
      <td>2017-12-12 09:45:42</td>
      <td>2017-12-12 09:49:37</td>
      <td>0</td>
      <td>0.6</td>
      <td>1</td>
      <td>N</td>
      <td>164</td>
      <td>233</td>
      <td>1</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.05</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>6.35</td>
      <td>3.916667</td>
    </tr>
    <tr>
      <th>13716</th>
      <td>89183211</td>
      <td>1</td>
      <td>2017-10-22 17:55:51</td>
      <td>2017-10-22 17:58:34</td>
      <td>0</td>
      <td>0.6</td>
      <td>1</td>
      <td>N</td>
      <td>162</td>
      <td>170</td>
      <td>1</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.72</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>5.52</td>
      <td>2.716667</td>
    </tr>
    <tr>
      <th>12501</th>
      <td>31724098</td>
      <td>1</td>
      <td>2017-04-18 17:47:58</td>
      <td>2017-04-18 17:47:58</td>
      <td>0</td>
      <td>0.0</td>
      <td>99</td>
      <td>N</td>
      <td>264</td>
      <td>264</td>
      <td>1</td>
      <td>77.2</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>78.00</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Most rides were single occupancy but a significant number were higher, the zero rider data will be addressed.

## Tips


```python
mean_tips_by_passenger_count = df.groupby(['passenger_count']).mean(numeric_only=True)[['tip_amount']]
mean_tips_by_passenger_count
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
      <th>tip_amount</th>
    </tr>
    <tr>
      <th>passenger_count</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.135758</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.848920</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.856378</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.716768</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.530264</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.873185</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.720260</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = mean_tips_by_passenger_count.tail(-1)
plt.figure(figsize=(13,8))
ax = sns.barplot(x= data.index,
                y = data['tip_amount'])
plt.title('Mean Tips by Passenger Count')

```




    Text(0.5, 1.0, 'Mean Tips by Passenger Count')




    
![png](output_35_1.png)
    



```python
df['month'] = df['tpep_pickup_datetime'].dt.month_name()
df['day'] = df['tpep_pickup_datetime'].dt.day_name()
```

## Ride Counts by Month and Day


```python
monthly_rides = df['month'].value_counts()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']
monthly_rides= monthly_rides.reindex(index=month_order)
monthly_rides
```




    January      1997
    February     1769
    March        2049
    April        2019
    May          2013
    June         1964
    July         1697
    August       1724
    September    1734
    October      2027
    November     1843
    December     1863
    Name: month, dtype: int64




```python
plt.figure(figsize=(12,7))
ax = sns.barplot(x=monthly_rides.index, y= monthly_rides)
ax.set_xticklabels(month_order)
plt.title('Rides by Month')
```




    Text(0.5, 1.0, 'Rides by Month')




    
![png](output_39_1.png)
    



```python
daily_rides = df['day'].value_counts()
day_order=  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_rides = daily_rides.reindex(index= day_order)
daily_rides
```




    Monday       2931
    Tuesday      3198
    Wednesday    3390
    Thursday     3402
    Friday       3413
    Saturday     3367
    Sunday       2998
    Name: day, dtype: int64




```python
plt.figure(figsize=(12,7))
ax = sns.barplot(x=daily_rides.index, y= daily_rides)
ax.set_xticklabels(day_order)
plt.title('Rides by Day of the Week')
```




    Text(0.5, 1.0, 'Rides by Day of the Week')




    
![png](output_41_1.png)
    


### Total Revenue per day of the week


```python
day_order=  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
total_amount_day= df.groupby('day').sum(numeric_only=True)[['total_amount']]
total_amount_day = total_amount_day.reindex(index=day_order)
total_amount_day
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
      <th>total_amount</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Monday</th>
      <td>49574.37</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>52527.14</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>55310.47</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>57181.91</td>
    </tr>
    <tr>
      <th>Friday</th>
      <td>55818.74</td>
    </tr>
    <tr>
      <th>Saturday</th>
      <td>51195.40</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>48624.06</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,7))
ax= sns.barplot(x= total_amount_day.index, y = total_amount_day['total_amount'])
ax.set_xticklabels(day_order)
plt.title('Total Revenue(USD) by Day')
```




    Text(0.5, 1.0, 'Total Revenue(USD) by Day')




    
![png](output_44_1.png)
    


### Plot mean trip distance by drop off location


```python
# Number of unique drop off locations
df['DOLocationID'].nunique()
```




    216




```python
ride_distance_by_dropoff = df.groupby('DOLocationID').mean(numeric_only=True)[['trip_distance']]
ride_distance_by_dropoff = ride_distance_by_dropoff.sort_values(by='trip_distance')
ride_distance_by_dropoff
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
      <th>trip_distance</th>
    </tr>
    <tr>
      <th>DOLocationID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>207</th>
      <td>1.200000</td>
    </tr>
    <tr>
      <th>193</th>
      <td>1.390556</td>
    </tr>
    <tr>
      <th>237</th>
      <td>1.555494</td>
    </tr>
    <tr>
      <th>234</th>
      <td>1.727806</td>
    </tr>
    <tr>
      <th>137</th>
      <td>1.818852</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>17.310000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>17.945000</td>
    </tr>
    <tr>
      <th>210</th>
      <td>20.500000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>21.650000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24.275000</td>
    </tr>
  </tbody>
</table>
<p>216 rows × 1 columns</p>
</div>




```python
plt.figure(figsize=(14,6))
ax = sns.barplot(x= ride_distance_by_dropoff.index,
                y= ride_distance_by_dropoff['trip_distance'],
                order = ride_distance_by_dropoff.index)
ax.set_xticklabels([])
ax.set_xticks([])
plt.title('Mean Trip Distance by Dropoff Location', fontsize=16)
```




    Text(0.5, 1.0, 'Mean Trip Distance by Dropoff Location')




    
![png](output_48_1.png)
    


### The distribution of data by dropoff point demonstrates a roughly normal distribution.

### Rides and revenue show some variation over time periods but not dramatically different.


```python
# eliminate unnecessary column
df.drop('store_and_fwd_flag',axis = 1, inplace = True)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22699 entries, 0 to 22698
    Data columns (total 20 columns):
     #   Column                 Non-Null Count  Dtype         
    ---  ------                 --------------  -----         
     0   Unnamed: 0             22699 non-null  int64         
     1   VendorID               22699 non-null  int64         
     2   tpep_pickup_datetime   22699 non-null  datetime64[ns]
     3   tpep_dropoff_datetime  22699 non-null  datetime64[ns]
     4   passenger_count        22699 non-null  int64         
     5   trip_distance          22699 non-null  float64       
     6   RatecodeID             22699 non-null  int64         
     7   PULocationID           22699 non-null  int64         
     8   DOLocationID           22699 non-null  int64         
     9   payment_type           22699 non-null  int64         
     10  fare_amount            22699 non-null  float64       
     11  extra                  22699 non-null  float64       
     12  mta_tax                22699 non-null  float64       
     13  tip_amount             22699 non-null  float64       
     14  tolls_amount           22699 non-null  float64       
     15  improvement_surcharge  22699 non-null  float64       
     16  total_amount           22699 non-null  float64       
     17  ride_duration          22699 non-null  float64       
     18  month                  22699 non-null  object        
     19  day                    22699 non-null  object        
    dtypes: datetime64[ns](2), float64(9), int64(7), object(2)
    memory usage: 3.5+ MB


### All data types are now date time or numerical 


## General Analysis of data validity and cleanliness


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.269900e+04</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.675849e+07</td>
      <td>1.556236</td>
      <td>1.642319</td>
      <td>2.913313</td>
      <td>1.043394</td>
      <td>162.412353</td>
      <td>161.527997</td>
      <td>1.336887</td>
      <td>13.026629</td>
      <td>0.333275</td>
      <td>0.497445</td>
      <td>1.835781</td>
      <td>0.312542</td>
      <td>0.299551</td>
      <td>16.310502</td>
      <td>17.013777</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.274493e+07</td>
      <td>0.496838</td>
      <td>1.285231</td>
      <td>3.653171</td>
      <td>0.708391</td>
      <td>66.633373</td>
      <td>70.139691</td>
      <td>0.496211</td>
      <td>13.243791</td>
      <td>0.463097</td>
      <td>0.039465</td>
      <td>2.800626</td>
      <td>1.399212</td>
      <td>0.015673</td>
      <td>16.097295</td>
      <td>61.996482</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.212700e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-120.000000</td>
      <td>-1.000000</td>
      <td>-0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.300000</td>
      <td>-120.300000</td>
      <td>-16.983333</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.852056e+07</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>1.000000</td>
      <td>114.000000</td>
      <td>112.000000</td>
      <td>1.000000</td>
      <td>6.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>8.750000</td>
      <td>6.650000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.673150e+07</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.610000</td>
      <td>1.000000</td>
      <td>162.000000</td>
      <td>162.000000</td>
      <td>1.000000</td>
      <td>9.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>1.350000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>11.800000</td>
      <td>11.183333</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.537452e+07</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.060000</td>
      <td>1.000000</td>
      <td>233.000000</td>
      <td>233.000000</td>
      <td>2.000000</td>
      <td>14.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>2.450000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>17.800000</td>
      <td>18.383333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.134863e+08</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>33.960000</td>
      <td>99.000000</td>
      <td>265.000000</td>
      <td>265.000000</td>
      <td>4.000000</td>
      <td>999.990000</td>
      <td>4.500000</td>
      <td>0.500000</td>
      <td>200.000000</td>
      <td>19.100000</td>
      <td>0.300000</td>
      <td>1200.290000</td>
      <td>1439.550000</td>
    </tr>
  </tbody>
</table>
</div>



## Problematic Data Initial
### Negative or zero duration mins
### Extremely long max durations
### Trip Distance of zero
### Fare amount is zero
### Fare amount is excessively high
### Passenger count is zero



```python
df_clean = df
```


```python
df_clean[df_clean['ride_duration']<=0].shape
```




    (27, 20)




```python
#27 rows exist with ride durations less than or equal to zero.  Deleting these rows as they are a small part of dataset.
df_clean.drop(df_clean[df_clean.ride_duration<=0].index, inplace= True)
```


```python
df_clean[df_clean['ride_duration']<=0].shape
```




    (0, 20)




```python
df_clean[df_clean.ride_duration > 300].shape
```




    (45, 20)




```python
#There are 45 rows with ride durations in excess of 5 hours(300 min). These rows will also be deleted.
df_clean.drop(df_clean[df_clean.ride_duration>300].index, inplace = True)
```


```python
df_clean[df_clean.ride_duration > 300].shape
```




    (0, 20)




```python
df_clean[df_clean['trip_distance']<=0].shape
```




    (122, 20)




```python
# there are 122 rows with trip distance = 0, These rows will be deleted.
df_clean.drop(df_clean[df_clean.trip_distance <=0].index, inplace = True)
```


```python
df_clean[df_clean['trip_distance']<=0].shape
```




    (0, 20)




```python
df_clean[df_clean['fare_amount']<=0].shape
```




    (15, 20)




```python
#Dropping 15 rows with zero fare amount
df_clean.drop(df_clean[df_clean.fare_amount<=0].index, inplace = True)
df_clean[df_clean['fare_amount']<=0].shape
```




    (0, 20)




```python
df_clean_high_fares = df_clean[df_clean['total_amount']> 10
                               *np.mean(df_clean['total_amount'])]
df_clean_high_fares.sort_values(by='total_amount', ascending= False)
    
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
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>ride_duration</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8476</th>
      <td>11157412</td>
      <td>1</td>
      <td>2017-02-06 05:50:10</td>
      <td>2017-02-06 05:51:08</td>
      <td>1</td>
      <td>2.60</td>
      <td>5</td>
      <td>226</td>
      <td>226</td>
      <td>1</td>
      <td>999.99</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>200.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>1200.29</td>
      <td>0.966667</td>
      <td>February</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>13861</th>
      <td>40523668</td>
      <td>2</td>
      <td>2017-05-19 08:20:21</td>
      <td>2017-05-19 09:20:30</td>
      <td>1</td>
      <td>33.92</td>
      <td>5</td>
      <td>229</td>
      <td>265</td>
      <td>1</td>
      <td>200.01</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>51.64</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>258.21</td>
      <td>60.150000</td>
      <td>May</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>6064</th>
      <td>49894023</td>
      <td>2</td>
      <td>2017-06-13 12:30:22</td>
      <td>2017-06-13 13:37:51</td>
      <td>1</td>
      <td>32.72</td>
      <td>3</td>
      <td>138</td>
      <td>1</td>
      <td>1</td>
      <td>107.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55.50</td>
      <td>16.26</td>
      <td>0.3</td>
      <td>179.06</td>
      <td>67.483333</td>
      <td>June</td>
      <td>Tuesday</td>
    </tr>
  </tbody>
</table>
</div>




```python
#There is 1 total fare amount that is excessivley high that will be droppped
df_clean.drop(df_clean[df_clean['total_amount']>1000].index, inplace = True)
df_clean_high_fares = df_clean[df_clean['total_amount']> 10
                               *np.mean(df_clean['total_amount'])]
df_clean_high_fares.sort_values(by='total_amount', ascending= False)
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
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>ride_duration</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13861</th>
      <td>40523668</td>
      <td>2</td>
      <td>2017-05-19 08:20:21</td>
      <td>2017-05-19 09:20:30</td>
      <td>1</td>
      <td>33.92</td>
      <td>5</td>
      <td>229</td>
      <td>265</td>
      <td>1</td>
      <td>200.01</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>51.64</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>258.21</td>
      <td>60.150000</td>
      <td>May</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>6064</th>
      <td>49894023</td>
      <td>2</td>
      <td>2017-06-13 12:30:22</td>
      <td>2017-06-13 13:37:51</td>
      <td>1</td>
      <td>32.72</td>
      <td>3</td>
      <td>138</td>
      <td>1</td>
      <td>1</td>
      <td>107.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55.50</td>
      <td>16.26</td>
      <td>0.3</td>
      <td>179.06</td>
      <td>67.483333</td>
      <td>June</td>
      <td>Tuesday</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Passenger count 0
df_pass_zero = df_clean[df_clean['passenger_count']==0]
df_pass_zero.shape
```




    (32, 20)




```python
df_pass_zero
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
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>ride_duration</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1516</th>
      <td>96563556</td>
      <td>1</td>
      <td>2017-11-14 15:45:23</td>
      <td>2017-11-14 16:26:38</td>
      <td>0</td>
      <td>8.8</td>
      <td>1</td>
      <td>138</td>
      <td>164</td>
      <td>1</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>5.55</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>45.11</td>
      <td>41.250000</td>
      <td>November</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>1874</th>
      <td>102273749</td>
      <td>1</td>
      <td>2017-12-03 12:05:52</td>
      <td>2017-12-03 12:13:51</td>
      <td>0</td>
      <td>1.1</td>
      <td>1</td>
      <td>48</td>
      <td>237</td>
      <td>2</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>7.80</td>
      <td>7.983333</td>
      <td>December</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>75654951</td>
      <td>1</td>
      <td>2017-09-09 03:44:45</td>
      <td>2017-09-09 03:49:19</td>
      <td>0</td>
      <td>0.8</td>
      <td>1</td>
      <td>48</td>
      <td>48</td>
      <td>2</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>6.30</td>
      <td>4.566667</td>
      <td>September</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>3352</th>
      <td>107464306</td>
      <td>1</td>
      <td>2017-12-18 22:44:57</td>
      <td>2017-12-18 23:00:24</td>
      <td>0</td>
      <td>2.7</td>
      <td>1</td>
      <td>87</td>
      <td>79</td>
      <td>1</td>
      <td>12.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.75</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>16.55</td>
      <td>15.450000</td>
      <td>December</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>4060</th>
      <td>100326273</td>
      <td>1</td>
      <td>2017-11-27 13:08:01</td>
      <td>2017-11-27 13:17:55</td>
      <td>0</td>
      <td>1.7</td>
      <td>1</td>
      <td>161</td>
      <td>239</td>
      <td>1</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.95</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>11.75</td>
      <td>9.900000</td>
      <td>November</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>4919</th>
      <td>106693550</td>
      <td>1</td>
      <td>2017-12-16 14:59:17</td>
      <td>2017-12-16 15:21:59</td>
      <td>0</td>
      <td>3.1</td>
      <td>1</td>
      <td>163</td>
      <td>114</td>
      <td>1</td>
      <td>15.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>4.05</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>20.35</td>
      <td>22.700000</td>
      <td>December</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>5563</th>
      <td>74279671</td>
      <td>1</td>
      <td>2017-09-04 17:40:00</td>
      <td>2017-09-04 17:51:52</td>
      <td>0</td>
      <td>1.3</td>
      <td>1</td>
      <td>164</td>
      <td>233</td>
      <td>1</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.95</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>11.75</td>
      <td>11.866667</td>
      <td>September</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>5603</th>
      <td>107019016</td>
      <td>1</td>
      <td>2017-12-17 12:18:49</td>
      <td>2017-12-17 12:40:45</td>
      <td>0</td>
      <td>4.2</td>
      <td>1</td>
      <td>230</td>
      <td>211</td>
      <td>1</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>3.75</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>22.55</td>
      <td>21.933333</td>
      <td>December</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>5668</th>
      <td>106785624</td>
      <td>1</td>
      <td>2017-12-16 19:50:12</td>
      <td>2017-12-16 20:00:30</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>211</td>
      <td>249</td>
      <td>1</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.50</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>10.30</td>
      <td>10.300000</td>
      <td>December</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>5767</th>
      <td>83401081</td>
      <td>1</td>
      <td>2017-10-04 15:17:52</td>
      <td>2017-10-04 15:56:26</td>
      <td>0</td>
      <td>10.4</td>
      <td>1</td>
      <td>113</td>
      <td>138</td>
      <td>1</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>8.10</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>48.66</td>
      <td>38.566667</td>
      <td>October</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>7102</th>
      <td>110296406</td>
      <td>1</td>
      <td>2017-12-29 13:41:17</td>
      <td>2017-12-29 13:55:57</td>
      <td>0</td>
      <td>2.1</td>
      <td>1</td>
      <td>264</td>
      <td>264</td>
      <td>1</td>
      <td>11.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.45</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>14.75</td>
      <td>14.666667</td>
      <td>December</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>8595</th>
      <td>96507020</td>
      <td>1</td>
      <td>2017-11-14 12:06:01</td>
      <td>2017-11-14 12:27:38</td>
      <td>0</td>
      <td>7.0</td>
      <td>1</td>
      <td>162</td>
      <td>13</td>
      <td>1</td>
      <td>23.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>6.05</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>30.35</td>
      <td>21.616667</td>
      <td>November</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>9828</th>
      <td>84313191</td>
      <td>1</td>
      <td>2017-10-07 10:02:35</td>
      <td>2017-10-07 10:13:34</td>
      <td>0</td>
      <td>6.0</td>
      <td>1</td>
      <td>140</td>
      <td>88</td>
      <td>1</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>3.75</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>22.55</td>
      <td>10.983333</td>
      <td>October</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>10145</th>
      <td>102526701</td>
      <td>1</td>
      <td>2017-12-04 10:41:30</td>
      <td>2017-12-04 10:58:23</td>
      <td>0</td>
      <td>2.0</td>
      <td>1</td>
      <td>142</td>
      <td>263</td>
      <td>1</td>
      <td>12.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>3.30</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>16.60</td>
      <td>16.883333</td>
      <td>December</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>10199</th>
      <td>78367526</td>
      <td>1</td>
      <td>2017-09-17 17:05:16</td>
      <td>2017-09-17 17:13:51</td>
      <td>0</td>
      <td>0.9</td>
      <td>1</td>
      <td>236</td>
      <td>237</td>
      <td>1</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>8.80</td>
      <td>8.583333</td>
      <td>September</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>12203</th>
      <td>97131247</td>
      <td>1</td>
      <td>2017-11-16 09:42:47</td>
      <td>2017-11-16 09:50:04</td>
      <td>0</td>
      <td>1.1</td>
      <td>1</td>
      <td>107</td>
      <td>79</td>
      <td>2</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>7.30</td>
      <td>7.283333</td>
      <td>November</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>12802</th>
      <td>79604871</td>
      <td>1</td>
      <td>2017-09-22 06:49:25</td>
      <td>2017-09-22 07:01:57</td>
      <td>0</td>
      <td>2.9</td>
      <td>1</td>
      <td>87</td>
      <td>234</td>
      <td>1</td>
      <td>11.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.45</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>14.75</td>
      <td>12.533333</td>
      <td>September</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>13419</th>
      <td>107412151</td>
      <td>1</td>
      <td>2017-12-18 19:32:14</td>
      <td>2017-12-18 19:39:07</td>
      <td>0</td>
      <td>2.4</td>
      <td>1</td>
      <td>68</td>
      <td>50</td>
      <td>1</td>
      <td>8.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>11.30</td>
      <td>6.883333</td>
      <td>December</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>13477</th>
      <td>105179313</td>
      <td>1</td>
      <td>2017-12-12 09:45:42</td>
      <td>2017-12-12 09:49:37</td>
      <td>0</td>
      <td>0.6</td>
      <td>1</td>
      <td>164</td>
      <td>233</td>
      <td>1</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.05</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>6.35</td>
      <td>3.916667</td>
      <td>December</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>13716</th>
      <td>89183211</td>
      <td>1</td>
      <td>2017-10-22 17:55:51</td>
      <td>2017-10-22 17:58:34</td>
      <td>0</td>
      <td>0.6</td>
      <td>1</td>
      <td>162</td>
      <td>170</td>
      <td>1</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.72</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>5.52</td>
      <td>2.716667</td>
      <td>October</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>13919</th>
      <td>93579534</td>
      <td>1</td>
      <td>2017-11-05 03:10:59</td>
      <td>2017-11-05 03:26:38</td>
      <td>0</td>
      <td>2.3</td>
      <td>1</td>
      <td>79</td>
      <td>261</td>
      <td>1</td>
      <td>11.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>14.80</td>
      <td>15.650000</td>
      <td>November</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>13921</th>
      <td>97593329</td>
      <td>1</td>
      <td>2017-11-17 16:39:02</td>
      <td>2017-11-17 16:52:38</td>
      <td>0</td>
      <td>0.9</td>
      <td>1</td>
      <td>236</td>
      <td>141</td>
      <td>1</td>
      <td>9.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>2.25</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>13.55</td>
      <td>13.600000</td>
      <td>November</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>14519</th>
      <td>109119545</td>
      <td>1</td>
      <td>2017-12-24 09:02:00</td>
      <td>2017-12-24 09:07:56</td>
      <td>0</td>
      <td>0.9</td>
      <td>1</td>
      <td>75</td>
      <td>74</td>
      <td>1</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.35</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>8.15</td>
      <td>5.933333</td>
      <td>December</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>14527</th>
      <td>96810412</td>
      <td>1</td>
      <td>2017-11-15 10:00:37</td>
      <td>2017-11-15 10:24:43</td>
      <td>0</td>
      <td>5.5</td>
      <td>1</td>
      <td>161</td>
      <td>87</td>
      <td>1</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.50</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>23.30</td>
      <td>24.100000</td>
      <td>November</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>17914</th>
      <td>107896724</td>
      <td>1</td>
      <td>2017-12-20 09:42:39</td>
      <td>2017-12-20 09:57:47</td>
      <td>0</td>
      <td>0.8</td>
      <td>1</td>
      <td>229</td>
      <td>162</td>
      <td>1</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.15</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>12.95</td>
      <td>15.133333</td>
      <td>December</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>18309</th>
      <td>101059790</td>
      <td>1</td>
      <td>2017-11-29 21:57:19</td>
      <td>2017-11-29 22:04:13</td>
      <td>0</td>
      <td>0.7</td>
      <td>1</td>
      <td>50</td>
      <td>100</td>
      <td>2</td>
      <td>6.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>7.80</td>
      <td>6.900000</td>
      <td>November</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>19217</th>
      <td>92734995</td>
      <td>1</td>
      <td>2017-11-02 19:01:54</td>
      <td>2017-11-02 19:25:31</td>
      <td>0</td>
      <td>2.8</td>
      <td>1</td>
      <td>262</td>
      <td>142</td>
      <td>2</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>17.80</td>
      <td>23.616667</td>
      <td>November</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>19456</th>
      <td>90729144</td>
      <td>1</td>
      <td>2017-10-27 14:11:07</td>
      <td>2017-10-27 14:32:17</td>
      <td>0</td>
      <td>2.7</td>
      <td>1</td>
      <td>231</td>
      <td>186</td>
      <td>1</td>
      <td>14.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.50</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>17.80</td>
      <td>21.166667</td>
      <td>October</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>20310</th>
      <td>108220152</td>
      <td>1</td>
      <td>2017-12-21 08:55:39</td>
      <td>2017-12-21 09:04:03</td>
      <td>0</td>
      <td>1.1</td>
      <td>1</td>
      <td>113</td>
      <td>90</td>
      <td>1</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.55</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>9.35</td>
      <td>8.400000</td>
      <td>December</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>21216</th>
      <td>109673439</td>
      <td>1</td>
      <td>2017-12-27 06:21:58</td>
      <td>2017-12-27 06:27:32</td>
      <td>0</td>
      <td>1.3</td>
      <td>1</td>
      <td>237</td>
      <td>262</td>
      <td>2</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>7.30</td>
      <td>5.566667</td>
      <td>December</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>21638</th>
      <td>98705879</td>
      <td>1</td>
      <td>2017-11-21 05:47:11</td>
      <td>2017-11-21 05:58:23</td>
      <td>0</td>
      <td>2.2</td>
      <td>1</td>
      <td>238</td>
      <td>230</td>
      <td>1</td>
      <td>10.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.36</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>14.16</td>
      <td>11.200000</td>
      <td>November</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>21752</th>
      <td>91222179</td>
      <td>1</td>
      <td>2017-10-28 22:25:02</td>
      <td>2017-10-28 22:43:44</td>
      <td>0</td>
      <td>1.6</td>
      <td>1</td>
      <td>113</td>
      <td>246</td>
      <td>1</td>
      <td>12.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>3.45</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>17.25</td>
      <td>18.700000</td>
      <td>October</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.mean(df_clean['passenger_count'][df_clean['passenger_count']>0])
```




    1.6454112303513382




```python
np.median(df_clean['passenger_count'][df_clean['passenger_count']>0])
```




    1.0




```python
# Generally the rest of the data in rows with passenger coynt = 0 looks valid,
#so the passenger count will have passenger count converted to 1(the median)
df_clean.loc[df_clean['passenger_count'] == 0, 'passenger_count'] = 1
df_clean[df_clean['passenger_count']== 0].shape
```




    (0, 20)




```python
df_clean.shape
```




    (22489, 20)



### We have elimionated 210 rows of data with problematic fields

## Creating a new column that includes all of the required fares and tolls but eliminates the tip due to their discretionary nature.


```python
df_clean['total_fare_minus_tip']= df_clean['total_amount'] - df_clean['tip_amount']

```


```python
#Determining likely errors in fares by creating a new column dollars per mile
df_clean['dollars_per_mile'] = df_clean['total_fare_minus_tip']/df_clean['trip_distance']
df_clean.head(3)
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
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>




```python
df_clean.describe()
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
      <td>2.248900e+04</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>2.248900e+04</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
      <td>22489.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.675634e+07</td>
      <td>1.555961</td>
      <td>1.644493</td>
      <td>2.931268</td>
      <td>1.031882</td>
      <td>162.345324</td>
      <td>161.457735</td>
      <td>1.332207</td>
      <td>12.923968</td>
      <td>0.333496</td>
      <td>0.498666</td>
      <td>1.823455</td>
      <td>0.309446</td>
      <td>3.000000e-01</td>
      <td>16.194361</td>
      <td>14.425323</td>
      <td>14.370906</td>
      <td>8.457287</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.273584e+07</td>
      <td>0.496870</td>
      <td>1.284238</td>
      <td>3.654989</td>
      <td>0.229857</td>
      <td>66.594261</td>
      <td>70.092585</td>
      <td>0.489715</td>
      <td>10.819839</td>
      <td>0.461053</td>
      <td>0.025792</td>
      <td>2.430776</td>
      <td>1.385757</td>
      <td>1.169646e-13</td>
      <td>13.403908</td>
      <td>11.648887</td>
      <td>11.780577</td>
      <td>77.684671</td>
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
      <td>2.854340e+07</td>
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
      <td>5.674552e+07</td>
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
      <td>8.537101e+07</td>
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
      <td>8.194444</td>
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
      <td>5856.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clean['dollars_per_mile'].describe()
```




    count    22489.000000
    mean         8.457287
    std         77.684671
    min          0.380000
    25%          4.777328
    50%          6.266667
    75%          8.194444
    max       5856.000000
    Name: dollars_per_mile, dtype: float64




```python
# Exploring data rows where dollars per mile is outlier
plt.figure(figsize=(6,1))
plt.title('dollars_per_mile')
sns.boxplot(data=None, x = df_clean['dollars_per_mile'], fliersize=1)
```




    <Axes: title={'center': 'dollars_per_mile'}, xlabel='dollars_per_mile'>




    
![png](output_82_1.png)
    


### There are several outliers in the data that will be explored further


```python
iqr = 8.19444- 4.777328
iqr
df_clean_high_rate = df_clean[df_clean['dollars_per_mile']>8.19444+1.5*iqr]
df_clean_high_rate
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
      <th>60</th>
      <td>34663272</td>
      <td>2</td>
      <td>2017-04-26 18:08:41</td>
      <td>2017-04-26 18:11:08</td>
      <td>5</td>
      <td>0.38</td>
      <td>1</td>
      <td>236</td>
      <td>236</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.16</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>6.96</td>
      <td>2.450000</td>
      <td>April</td>
      <td>Wednesday</td>
      <td>5.8</td>
      <td>15.263158</td>
    </tr>
    <tr>
      <th>63</th>
      <td>83622942</td>
      <td>2</td>
      <td>2017-10-05 09:39:28</td>
      <td>2017-10-05 09:42:32</td>
      <td>6</td>
      <td>0.22</td>
      <td>3</td>
      <td>48</td>
      <td>48</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>21.80</td>
      <td>3.066667</td>
      <td>October</td>
      <td>Thursday</td>
      <td>21.8</td>
      <td>99.090909</td>
    </tr>
    <tr>
      <th>145</th>
      <td>28327177</td>
      <td>1</td>
      <td>2017-04-06 02:45:19</td>
      <td>2017-04-06 02:47:22</td>
      <td>1</td>
      <td>0.30</td>
      <td>1</td>
      <td>170</td>
      <td>170</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.80</td>
      <td>2.050000</td>
      <td>April</td>
      <td>Thursday</td>
      <td>4.8</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>161</th>
      <td>95729204</td>
      <td>2</td>
      <td>2017-11-11 20:16:16</td>
      <td>2017-11-11 20:17:14</td>
      <td>1</td>
      <td>0.23</td>
      <td>2</td>
      <td>132</td>
      <td>132</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>52.80</td>
      <td>0.966667</td>
      <td>November</td>
      <td>Saturday</td>
      <td>52.8</td>
      <td>229.565217</td>
    </tr>
    <tr>
      <th>170</th>
      <td>46716747</td>
      <td>1</td>
      <td>2017-05-31 20:58:40</td>
      <td>2017-05-31 21:02:51</td>
      <td>2</td>
      <td>0.30</td>
      <td>1</td>
      <td>237</td>
      <td>237</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>5.80</td>
      <td>4.183333</td>
      <td>May</td>
      <td>Wednesday</td>
      <td>5.8</td>
      <td>19.333333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22557</th>
      <td>58013284</td>
      <td>2</td>
      <td>2017-07-07 20:57:59</td>
      <td>2017-07-07 21:01:55</td>
      <td>1</td>
      <td>0.33</td>
      <td>1</td>
      <td>79</td>
      <td>148</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.16</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>6.96</td>
      <td>3.933333</td>
      <td>July</td>
      <td>Friday</td>
      <td>5.8</td>
      <td>17.575758</td>
    </tr>
    <tr>
      <th>22569</th>
      <td>79057964</td>
      <td>1</td>
      <td>2017-09-20 09:18:41</td>
      <td>2017-09-20 09:32:05</td>
      <td>1</td>
      <td>0.60</td>
      <td>1</td>
      <td>90</td>
      <td>234</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.20</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>11.00</td>
      <td>13.400000</td>
      <td>September</td>
      <td>Wednesday</td>
      <td>9.8</td>
      <td>16.333333</td>
    </tr>
    <tr>
      <th>22616</th>
      <td>34075469</td>
      <td>1</td>
      <td>2017-04-25 02:55:40</td>
      <td>2017-04-25 02:56:50</td>
      <td>1</td>
      <td>0.20</td>
      <td>1</td>
      <td>249</td>
      <td>113</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.30</td>
      <td>1.166667</td>
      <td>April</td>
      <td>Tuesday</td>
      <td>4.3</td>
      <td>21.500000</td>
    </tr>
    <tr>
      <th>22617</th>
      <td>82975513</td>
      <td>2</td>
      <td>2017-10-03 08:37:50</td>
      <td>2017-10-03 08:43:39</td>
      <td>1</td>
      <td>0.32</td>
      <td>1</td>
      <td>234</td>
      <td>137</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.26</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>7.56</td>
      <td>5.816667</td>
      <td>October</td>
      <td>Tuesday</td>
      <td>6.3</td>
      <td>19.687500</td>
    </tr>
    <tr>
      <th>22632</th>
      <td>1049724</td>
      <td>2</td>
      <td>2017-01-04 17:02:46</td>
      <td>2017-01-04 17:14:53</td>
      <td>1</td>
      <td>0.77</td>
      <td>1</td>
      <td>162</td>
      <td>230</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>10.30</td>
      <td>12.116667</td>
      <td>January</td>
      <td>Wednesday</td>
      <td>10.3</td>
      <td>13.376623</td>
    </tr>
  </tbody>
</table>
<p>935 rows × 22 columns</p>
</div>



### An excessively large number of rows meet the statistical definition of outliers (935).  This may be an excessive amount of culling if all of thse are deleted.  We will delete the most extreme rows only



```python
df_clean_high_rate.sort_values(by=['dollars_per_mile'],ascending=False).head(10)

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
      <th>19827</th>
      <td>59839344</td>
      <td>2</td>
      <td>2017-07-14 06:09:54</td>
      <td>2017-07-14 06:11:40</td>
      <td>1</td>
      <td>0.01</td>
      <td>2</td>
      <td>132</td>
      <td>132</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>58.56</td>
      <td>1.766667</td>
      <td>July</td>
      <td>Friday</td>
      <td>58.56</td>
      <td>5856.000000</td>
    </tr>
    <tr>
      <th>19644</th>
      <td>105577859</td>
      <td>2</td>
      <td>2017-12-13 12:19:29</td>
      <td>2017-12-13 12:19:39</td>
      <td>1</td>
      <td>0.01</td>
      <td>2</td>
      <td>132</td>
      <td>132</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>17.57</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>76.13</td>
      <td>0.166667</td>
      <td>December</td>
      <td>Wednesday</td>
      <td>58.56</td>
      <td>5856.000000</td>
    </tr>
    <tr>
      <th>3609</th>
      <td>98165974</td>
      <td>2</td>
      <td>2017-11-19 07:17:16</td>
      <td>2017-11-19 07:17:19</td>
      <td>1</td>
      <td>0.01</td>
      <td>2</td>
      <td>264</td>
      <td>239</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>52.80</td>
      <td>0.050000</td>
      <td>November</td>
      <td>Sunday</td>
      <td>52.80</td>
      <td>5280.000000</td>
    </tr>
    <tr>
      <th>8197</th>
      <td>39498898</td>
      <td>2</td>
      <td>2017-05-16 13:33:23</td>
      <td>2017-05-16 13:33:37</td>
      <td>1</td>
      <td>0.01</td>
      <td>2</td>
      <td>100</td>
      <td>100</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>52.80</td>
      <td>0.233333</td>
      <td>May</td>
      <td>Tuesday</td>
      <td>52.80</td>
      <td>5280.000000</td>
    </tr>
    <tr>
      <th>5429</th>
      <td>104244836</td>
      <td>2</td>
      <td>2017-12-09 11:56:56</td>
      <td>2017-12-09 11:58:13</td>
      <td>1</td>
      <td>0.02</td>
      <td>2</td>
      <td>186</td>
      <td>186</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>11.71</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>70.27</td>
      <td>1.283333</td>
      <td>December</td>
      <td>Saturday</td>
      <td>58.56</td>
      <td>2928.000000</td>
    </tr>
    <tr>
      <th>21088</th>
      <td>68563779</td>
      <td>2</td>
      <td>2017-08-13 16:09:35</td>
      <td>2017-08-13 16:10:56</td>
      <td>1</td>
      <td>0.03</td>
      <td>3</td>
      <td>170</td>
      <td>170</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>20.80</td>
      <td>1.350000</td>
      <td>August</td>
      <td>Sunday</td>
      <td>20.80</td>
      <td>693.333333</td>
    </tr>
    <tr>
      <th>5200</th>
      <td>90105531</td>
      <td>1</td>
      <td>2017-10-25 18:08:16</td>
      <td>2017-10-25 18:08:35</td>
      <td>1</td>
      <td>0.10</td>
      <td>2</td>
      <td>132</td>
      <td>132</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>5.00</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>68.06</td>
      <td>0.316667</td>
      <td>October</td>
      <td>Wednesday</td>
      <td>63.06</td>
      <td>630.600000</td>
    </tr>
    <tr>
      <th>4541</th>
      <td>2618392</td>
      <td>2</td>
      <td>2017-01-10 18:25:47</td>
      <td>2017-01-10 18:42:09</td>
      <td>5</td>
      <td>0.02</td>
      <td>1</td>
      <td>236</td>
      <td>239</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>12.30</td>
      <td>16.366667</td>
      <td>January</td>
      <td>Tuesday</td>
      <td>12.30</td>
      <td>615.000000</td>
    </tr>
    <tr>
      <th>3288</th>
      <td>66798269</td>
      <td>1</td>
      <td>2017-08-07 10:20:05</td>
      <td>2017-08-07 10:20:53</td>
      <td>1</td>
      <td>0.10</td>
      <td>2</td>
      <td>162</td>
      <td>163</td>
      <td>1</td>
      <td>...</td>
      <td>0.5</td>
      <td>8.00</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>66.56</td>
      <td>0.800000</td>
      <td>August</td>
      <td>Monday</td>
      <td>58.56</td>
      <td>585.600000</td>
    </tr>
    <tr>
      <th>9188</th>
      <td>26279873</td>
      <td>2</td>
      <td>2017-03-31 05:29:19</td>
      <td>2017-03-31 05:29:32</td>
      <td>1</td>
      <td>0.01</td>
      <td>1</td>
      <td>249</td>
      <td>249</td>
      <td>2</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>3.80</td>
      <td>0.216667</td>
      <td>March</td>
      <td>Friday</td>
      <td>3.80</td>
      <td>380.000000</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 22 columns</p>
</div>



### The top 5 dollars per mile rides are the most extreme and will be deleted.


```python
df_clean.drop(df_clean[df_clean['dollars_per_mile']>1000].index, inplace=True)
```


```python
df_clean.shape
```




    (22484, 22)




```python
df_clean.to_csv('tlc_data_clean.csv', index = False)
```

### The dataframe is now cleaned and is ready for further analysis.


```python

```
