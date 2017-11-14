
<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 15px; height: 80px">

# Project 2

### Exploratory Data Analysis (EDA)

---

Your hometown mayor just created a new data analysis team to give policy advice, and the administration recruited _you_ via LinkedIn to join it. Unfortunately, due to budget constraints, for now the "team" is just you...

The mayor wants to start a new initiative to move the needle on one of two separate issues: high school education outcomes, or drug abuse in the community.

Also unfortunately, that is the entirety of what you've been told. And the mayor just went on a lobbyist-funded fact-finding trip in the Bahamas. In the meantime, you got your hands on two national datasets: one on SAT scores by state, and one on drug use by age. Start exploring these to look for useful patterns and possible hypotheses!

---

This project is focused on exploratory data analysis, aka "EDA". EDA is an essential part of the data science analysis pipeline. Failure to perform EDA before modeling is almost guaranteed to lead to bad models and faulty conclusions. What you do in this project are good practices for all projects going forward, especially those after this bootcamp!

This lab includes a variety of plotting problems. Much of the plotting code will be left up to you to find either in the lecture notes, or if not there, online. There are massive amounts of code snippets either in documentation or sites like [Stack Overflow](https://stackoverflow.com/search?q=%5Bpython%5D+seaborn) that have almost certainly done what you are trying to do.

**Get used to googling for code!** You will use it every single day as a data scientist, especially for visualization and plotting.

#### Package imports


```python
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# this line tells jupyter notebook to put the plots in the notebook rather than saving them to file.
%matplotlib inline

# this line makes plots prettier on mac retina screens. If you don't have one it shouldn't do anything.
%config InlineBackend.figure_format = 'retina'
```

<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 1. Load the `sat_scores.csv` dataset and describe it

---

NOTE: All CSVs are in the same directory as this notebook


```python
# Gave the dataset a variable name "sat"
sat = 'sat_scores.csv'
```

### 1.1 Make a pandas DataFrame object with pandas `.read_csv()` function

Take a look at the `.dtypes` attribute in the DataFrame. 


```python
# Loaded the data set by using pd.read_csv and called the sat variable.
# Gave the data set the variable name "sat_scores".
# Called the variable "sat_scores" to view the data.

sat_scores = pd.read_csv(sat)

sat_scores
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RI</td>
      <td>71</td>
      <td>501</td>
      <td>499</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PA</td>
      <td>71</td>
      <td>500</td>
      <td>499</td>
    </tr>
    <tr>
      <th>7</th>
      <td>VT</td>
      <td>69</td>
      <td>511</td>
      <td>506</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ME</td>
      <td>69</td>
      <td>506</td>
      <td>500</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VA</td>
      <td>68</td>
      <td>510</td>
      <td>501</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DE</td>
      <td>67</td>
      <td>501</td>
      <td>499</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MD</td>
      <td>65</td>
      <td>508</td>
      <td>510</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NC</td>
      <td>65</td>
      <td>493</td>
      <td>499</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GA</td>
      <td>63</td>
      <td>491</td>
      <td>489</td>
    </tr>
    <tr>
      <th>14</th>
      <td>IN</td>
      <td>60</td>
      <td>499</td>
      <td>501</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SC</td>
      <td>57</td>
      <td>486</td>
      <td>488</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DC</td>
      <td>56</td>
      <td>482</td>
      <td>474</td>
    </tr>
    <tr>
      <th>17</th>
      <td>OR</td>
      <td>55</td>
      <td>526</td>
      <td>526</td>
    </tr>
    <tr>
      <th>18</th>
      <td>FL</td>
      <td>54</td>
      <td>498</td>
      <td>499</td>
    </tr>
    <tr>
      <th>19</th>
      <td>WA</td>
      <td>53</td>
      <td>527</td>
      <td>527</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TX</td>
      <td>53</td>
      <td>493</td>
      <td>499</td>
    </tr>
    <tr>
      <th>21</th>
      <td>HI</td>
      <td>52</td>
      <td>485</td>
      <td>515</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AK</td>
      <td>51</td>
      <td>514</td>
      <td>510</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CA</td>
      <td>51</td>
      <td>498</td>
      <td>517</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AZ</td>
      <td>34</td>
      <td>523</td>
      <td>525</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NV</td>
      <td>33</td>
      <td>509</td>
      <td>515</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>31</td>
      <td>539</td>
      <td>542</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>23</td>
      <td>539</td>
      <td>539</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WV</td>
      <td>18</td>
      <td>527</td>
      <td>512</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>17</td>
      <td>543</td>
      <td>542</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>13</td>
      <td>562</td>
      <td>553</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>13</td>
      <td>551</td>
      <td>542</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>12</td>
      <td>576</td>
      <td>589</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>12</td>
      <td>550</td>
      <td>550</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>11</td>
      <td>547</td>
      <td>545</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>11</td>
      <td>561</td>
      <td>572</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>9</td>
      <td>580</td>
      <td>589</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>9</td>
      <td>577</td>
      <td>580</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>9</td>
      <td>559</td>
      <td>554</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>8</td>
      <td>562</td>
      <td>568</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>8</td>
      <td>567</td>
      <td>561</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>8</td>
      <td>577</td>
      <td>577</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>7</td>
      <td>564</td>
      <td>562</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>6</td>
      <td>584</td>
      <td>596</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>6</td>
      <td>562</td>
      <td>550</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
    <tr>
      <th>51</th>
      <td>All</td>
      <td>45</td>
      <td>506</td>
      <td>514</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Used .dtypes to see the types of data.

sat_scores.dtypes
```




    State     object
    Rate       int64
    Verbal     int64
    Math       int64
    dtype: object



### 1.2 Look at the first ten rows of the DataFrame: what does our data describe?

From now on, use the DataFrame loaded from the file using the `.read_csv()` function.

Use the `.head(num)` built-in DataFrame function, where `num` is the number of rows to print out.

You are not given a "codebook" with this data, so you will have to make some (very minor) inference.


```python
# used .head(10) in order to view the first 10 rows of the DataFrame.

sat_scores.head(10)


```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RI</td>
      <td>71</td>
      <td>501</td>
      <td>499</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PA</td>
      <td>71</td>
      <td>500</td>
      <td>499</td>
    </tr>
    <tr>
      <th>7</th>
      <td>VT</td>
      <td>69</td>
      <td>511</td>
      <td>506</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ME</td>
      <td>69</td>
      <td>506</td>
      <td>500</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VA</td>
      <td>68</td>
      <td>510</td>
      <td>501</td>
    </tr>
  </tbody>
</table>
</div>



<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 2. Create a "data dictionary" based on the data

---

A data dictionary is an object that describes your data. This should contain the name of each variable (column), the type of the variable, your description of what the variable is, and the shape (rows and columns) of the entire dataset.


```python
# Used .columns to view the names of all the columns in the DataFrame.

sat_scores.columns
```




    Index(['State', 'Rate', 'Verbal', 'Math'], dtype='object')




```python
# Used .describe to see the attributes of the DataFrame

sat_scores.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>52.000000</td>
      <td>52.000000</td>
      <td>52.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37.153846</td>
      <td>532.019231</td>
      <td>531.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>27.301788</td>
      <td>33.236225</td>
      <td>36.014975</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>482.000000</td>
      <td>439.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>501.000000</td>
      <td>504.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.500000</td>
      <td>526.500000</td>
      <td>521.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>63.500000</td>
      <td>562.000000</td>
      <td>555.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>82.000000</td>
      <td>593.000000</td>
      <td>603.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Used .shape to show the shape (rows x columns) of the DataFrame.

sat_scores.shape
```




    (52, 4)




```python
# Used .dytpes to show the types(object, integer, float, e.g.) of the DataFrame.

sat_scores.dtypes
```




    State     object
    Rate       int64
    Verbal     int64
    Math       int64
    dtype: object




```python
# Created a Data dictionary after calling .describe, .shape and .dtypes
```


```python

# Data Dictionary:

# Variable  Definition        Type
# State      State             object
# Rate       Rate              int64
# Verbal     Verbal Score      int64
# Math       Math Score        int64

# The shape of the DataFrame is 52 rows by 4 columns
```

<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 3. Plot the data using seaborn

---

### 3.1 Using seaborn's `distplot`, plot the distributions for each of `Rate`, `Math`, and `Verbal`

Set the keyword argument `kde=False`. This way you can actually see the counts within bins. You can adjust the number of bins to your liking. 

[Please read over the `distplot` documentation to learn about the arguments and fine-tune your chart if you want.](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.distplot.html#seaborn.distplot)


```python
# Used Seaborn's distplot with each variable separately. This one is for "Rate".

sns.distplot(sat_scores["Rate"], kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1186505c0>




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_17_1.png)



```python
# Used Seaborn's distplot with each variable separately. This one is for "Verbal".

sns.distplot(sat_scores["Verbal"], kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11872db70>




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_18_1.png)



```python
# Used Seaborn's distplot with each variable separately. This one is for "Math".

sns.distplot(sat_scores["Math"], kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1187430b8>




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_19_1.png)


### 3.2 Using seaborn's `pairplot`, show the joint distributions for each of `Rate`, `Math`, and `Verbal`

Explain what the visualization tells you about your data.

[Please read over the `pairplot` documentation to fine-tune your chart.](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html#seaborn.pairplot)


```python
# Used Seaborn's pairplot with the 3 variables ("Rate", "Math", "Verbal") in one double bracket.

sns.pairplot(sat_scores[["Rate", "Math", "Verbal"]])
```




    <seaborn.axisgrid.PairGrid at 0x1185c1550>




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_21_1.png)


<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 4. Plot the data using built-in pandas functions.

---

Pandas is very powerful and contains a variety of nice, built-in plotting functions for your data. Read the documentation here to understand the capabilities:

http://pandas.pydata.org/pandas-docs/stable/visualization.html

### 4.1 Plot a stacked histogram with `Verbal` and `Math` using pandas


```python
# Plotted Stacked Histogram for Verbal and Math
sat_scores[['Math', 'Verbal']].plot.hist(grid=True,by=None)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b1632e8>




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_23_1.png)


### 4.2 Plot `Verbal` and `Math` on the same chart using boxplots

What are the benefits of using a boxplot as compared to a scatterplot or a histogram?

What's wrong with plotting a box-plot of `Rate` on the same chart as `Math` and `Verbal`?


```python
# Plotted Box Blot for Verbal and Math
sat_scores[['Math', 'Verbal']].plot.box(grid=True,by=None)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119e24278>




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_25_1.png)


<img src="http://imgur.com/xDpSobf.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

### 4.3 Plot `Verbal`, `Math`, and `Rate` appropriately on the same boxplot chart

Think about how you might change the variables so that they would make sense on the same chart. Explain your rationale for the choices on the chart. You should strive to make the chart as intuitive as possible. 



```python

```

<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 5. Create and examine subsets of the data

---

For these questions you will practice **masking** in pandas. Masking uses conditional statements to select portions of your DataFrame (through boolean operations under the hood.)

Remember the distinction between DataFrame indexing functions in pandas:

    .iloc[row, col] : row and column are specified by index, which are integers
    .loc[row, col]  : row and column are specified by string "labels" (boolean arrays are allowed; useful for rows)
    
For detailed reference and tutorial make sure to read over the pandas documentation:

http://pandas.pydata.org/pandas-docs/stable/indexing.html



### 5.1 Find the list of states that have an average `Verbal` score greater than the average of `Verbal` scores across the entire dataset

How many states are above the mean? What does this tell you about the distribution of `Verbal` scores?





```python
# Gave the mean of the "Verbal" column throughout the total DataFrame the variable name "sat_verbal_mean".
sat_verbal_mean = sat_scores['Verbal'].mean()

# Gave the mean of the states, whose mean was greater than the mean of the entire DataFrame, the variable name "state_verbal_mean"
state_verbal_mean = sat_scores[sat_scores['Verbal'] > sat_verbal_mean]

# Only showed the Index of the State, the State and the Mean Verbal score.
state_verbal_mean[["State" , "Verbal"]]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Verbal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>539</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>534</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>539</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>543</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>562</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>551</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>576</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>550</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>547</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>561</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>580</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>577</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>559</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>562</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>567</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>577</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>564</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>584</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>562</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>575</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>593</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>577</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>592</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>566</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 24 States are above the Mean, which means 26 States are at or below the mean.  That means the data is normally distributed.

state_verbal_mean.count()


```




    State     24
    Rate      24
    Verbal    24
    Math      24
    dtype: int64



### 5.2 Find the list of states that have a median `Verbal` score greater than the median of `Verbal` scores across the entire dataset

How does this compare to the list of states greater than the mean of `Verbal` scores? Why?


```python
# Gave the median of the "Verbal" column throughout the total DataFrame the variable name "sat_verbal_median".
sat_verbal_median = sat_scores['Verbal'].median()

# Gave the median of the states, whose median was greater than the median of the entire DataFrame, the variable name "state_verbal_median"
state_verbal_median = sat_scores[sat_scores['Verbal'] > sat_verbal_median]

# Only showed the Index of the State, the State and the Median Verbal score.
state_verbal_median[["State" , "Verbal"]]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Verbal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>WA</td>
      <td>527</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>539</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>534</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>539</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WV</td>
      <td>527</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>543</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>562</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>551</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>576</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>550</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>547</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>561</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>580</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>577</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>559</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>562</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>567</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>577</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>564</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>584</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>562</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>575</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>593</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>577</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>592</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>566</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Washington & West Virginia appear on the Median list, but not on the Mean List. They have outliers in their data.

state_verbal_median.count()
```




    State     26
    Rate      26
    Verbal    26
    Math      26
    dtype: int64



### 5.3 Create a column that is the difference between the `Verbal` and `Math` scores

Specifically, this should be `Verbal - Math`.


```python
# Created a column called and "Verbal_Less_Math" and assigned it to "sat_scores" Assigned the formula of Verbal scores - Math scores.

sat_scores['Verbal_Less_Math'] = sat_scores['Verbal'] - sat_scores['Math']
sat_scores
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>Verbal_Less_Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
      <td>-14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
      <td>-10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RI</td>
      <td>71</td>
      <td>501</td>
      <td>499</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PA</td>
      <td>71</td>
      <td>500</td>
      <td>499</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>VT</td>
      <td>69</td>
      <td>511</td>
      <td>506</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ME</td>
      <td>69</td>
      <td>506</td>
      <td>500</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VA</td>
      <td>68</td>
      <td>510</td>
      <td>501</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DE</td>
      <td>67</td>
      <td>501</td>
      <td>499</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MD</td>
      <td>65</td>
      <td>508</td>
      <td>510</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NC</td>
      <td>65</td>
      <td>493</td>
      <td>499</td>
      <td>-6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GA</td>
      <td>63</td>
      <td>491</td>
      <td>489</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>IN</td>
      <td>60</td>
      <td>499</td>
      <td>501</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SC</td>
      <td>57</td>
      <td>486</td>
      <td>488</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DC</td>
      <td>56</td>
      <td>482</td>
      <td>474</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>OR</td>
      <td>55</td>
      <td>526</td>
      <td>526</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>FL</td>
      <td>54</td>
      <td>498</td>
      <td>499</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>WA</td>
      <td>53</td>
      <td>527</td>
      <td>527</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TX</td>
      <td>53</td>
      <td>493</td>
      <td>499</td>
      <td>-6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>HI</td>
      <td>52</td>
      <td>485</td>
      <td>515</td>
      <td>-30</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AK</td>
      <td>51</td>
      <td>514</td>
      <td>510</td>
      <td>4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CA</td>
      <td>51</td>
      <td>498</td>
      <td>517</td>
      <td>-19</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AZ</td>
      <td>34</td>
      <td>523</td>
      <td>525</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NV</td>
      <td>33</td>
      <td>509</td>
      <td>515</td>
      <td>-6</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>31</td>
      <td>539</td>
      <td>542</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
      <td>95</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>23</td>
      <td>539</td>
      <td>539</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WV</td>
      <td>18</td>
      <td>527</td>
      <td>512</td>
      <td>15</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>17</td>
      <td>543</td>
      <td>542</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>13</td>
      <td>562</td>
      <td>553</td>
      <td>9</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>13</td>
      <td>551</td>
      <td>542</td>
      <td>9</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>12</td>
      <td>576</td>
      <td>589</td>
      <td>-13</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>12</td>
      <td>550</td>
      <td>550</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>11</td>
      <td>547</td>
      <td>545</td>
      <td>2</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>11</td>
      <td>561</td>
      <td>572</td>
      <td>-11</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>9</td>
      <td>580</td>
      <td>589</td>
      <td>-9</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>9</td>
      <td>577</td>
      <td>580</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>9</td>
      <td>559</td>
      <td>554</td>
      <td>5</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>8</td>
      <td>562</td>
      <td>568</td>
      <td>-6</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>8</td>
      <td>567</td>
      <td>561</td>
      <td>6</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>8</td>
      <td>577</td>
      <td>577</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>7</td>
      <td>564</td>
      <td>562</td>
      <td>2</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>6</td>
      <td>584</td>
      <td>596</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>6</td>
      <td>562</td>
      <td>550</td>
      <td>12</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
      <td>5</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
      <td>-10</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
      <td>-5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
      <td>-7</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
      <td>15</td>
    </tr>
    <tr>
      <th>51</th>
      <td>All</td>
      <td>45</td>
      <td>506</td>
      <td>514</td>
      <td>-8</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### 5.4 Create two new DataFrames showing states with the greatest difference between scores

1. Your first DataFrame should be the 10 states with the greatest gap between `Verbal` and `Math` scores where `Verbal` is greater than `Math`. It should be sorted appropriately to show the ranking of states.
2. Your second DataFrame will be the inverse: states with the greatest gap between `Verbal` and `Math` such that `Math` is greater than `Verbal`. Again, this should be sorted appropriately to show rank.
3. Print the header of both variables, only showing the top 3 states in each.


```python
# Used .sort_values and assigned it to "Verbal_Less_Math". Made it sort descending by the top 10 States that have a verbal - math score difference 

sat_scores.sort_values('Verbal_Less_Math',ascending=False).head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>Verbal_Less_Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
      <td>95</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
      <td>15</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WV</td>
      <td>18</td>
      <td>527</td>
      <td>512</td>
      <td>15</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>6</td>
      <td>562</td>
      <td>550</td>
      <td>12</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>13</td>
      <td>551</td>
      <td>542</td>
      <td>9</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>13</td>
      <td>562</td>
      <td>553</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VA</td>
      <td>68</td>
      <td>510</td>
      <td>501</td>
      <td>9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DC</td>
      <td>56</td>
      <td>482</td>
      <td>474</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ME</td>
      <td>69</td>
      <td>506</td>
      <td>500</td>
      <td>6</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>8</td>
      <td>567</td>
      <td>561</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## 6. Examine summary statistics

---

Checking the summary statistics for data is an essential step in the EDA process!

### 6.1 Create the correlation matrix of your variables (excluding `State`).


- Use seaborn's `.heatmap` method to add some color to the matrix
- Set `annot=True`



```python
# Created the correlation matrix using .corr and excluded the "State" column.

corr_sat_scores = sat_scores[['Math', 'Verbal']].corr()
corr_sat_scores
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Math</th>
      <th>Verbal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Math</th>
      <td>1.000000</td>
      <td>0.899871</td>
    </tr>
    <tr>
      <th>Verbal</th>
      <td>0.899871</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Created a heatmap with the correlation matrix.

sns.heatmap(corr_sat_scores, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119e6a630>




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_41_1.png)


<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

### 6.2 Use pandas'  `.describe()` built-in function on your DataFrame

Write up what each of the rows returned by the function indicate.


```python
# Count counts the rows
# Mean returns the average of each column
# STD returns the standard deviation of each column
# Min returns the minimun value of each column
# Max returns the maximun value of each column
# The 25%, 50%, and 75% are percentiles : array-like, optional
# The percentiles to include in the output. Should all be in the interval [0, 1]. By default percentiles is [.25, .5, .75], returning the 25th, 50th, and 75th percentiles.

sat_scores.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>Verbal_Less_Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>52.000000</td>
      <td>52.000000</td>
      <td>52.000000</td>
      <td>52.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37.153846</td>
      <td>532.019231</td>
      <td>531.500000</td>
      <td>0.519231</td>
    </tr>
    <tr>
      <th>std</th>
      <td>27.301788</td>
      <td>33.236225</td>
      <td>36.014975</td>
      <td>15.729939</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>482.000000</td>
      <td>439.000000</td>
      <td>-30.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>501.000000</td>
      <td>504.000000</td>
      <td>-6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.500000</td>
      <td>526.500000</td>
      <td>521.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>63.500000</td>
      <td>562.000000</td>
      <td>555.750000</td>
      <td>4.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>82.000000</td>
      <td>593.000000</td>
      <td>603.000000</td>
      <td>95.000000</td>
    </tr>
  </tbody>
</table>
</div>



<img src="http://imgur.com/xDpSobf.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

### 6.3 Assign and print the _covariance_ matrix for the dataset

1. Describe how the covariance matrix is different from the correlation matrix.
2. What is the process to convert the covariance into the correlation?
3. Why is the correlation matrix preferred to the covariance matrix for examining relationships in your data?


```python
# Covariance
# You tend to use the covariance matrix when the variable scales are similar and the correlation matrix when variables are on different scales.

sat_scores.cov()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>Verbal_Less_Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rate</th>
      <td>745.387632</td>
      <td>-804.355958</td>
      <td>-760.803922</td>
      <td>-43.552036</td>
    </tr>
    <tr>
      <th>Verbal</th>
      <td>-804.355958</td>
      <td>1104.646682</td>
      <td>1077.147059</td>
      <td>27.499623</td>
    </tr>
    <tr>
      <th>Math</th>
      <td>-760.803922</td>
      <td>1077.147059</td>
      <td>1297.078431</td>
      <td>-219.931373</td>
    </tr>
    <tr>
      <th>Verbal_Less_Math</th>
      <td>-43.552036</td>
      <td>27.499623</td>
      <td>-219.931373</td>
      <td>247.430995</td>
    </tr>
  </tbody>
</table>
</div>



<img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 7. Performing EDA on "drug use by age" data.

---

You will now switch datasets to one with many more variables. This section of the project is more open-ended - use the techniques you practiced above!

We'll work with the "drug-use-by-age.csv" data, sourced from and described here: https://github.com/fivethirtyeight/data/tree/master/drug-use-by-age.

### 7.1

Load the data using pandas. Does this data require cleaning? Are variables missing? How will this affect your approach to EDA on the data?


```python
# Loaded the dataset

drug_use = pd.read_csv('drug-use-by-age.csv')
drug_use
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>...</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>...</td>
      <td>0.1</td>
      <td>24.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>2757</td>
      <td>8.5</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>15.0</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.1</td>
      <td>41.0</td>
      <td>0.3</td>
      <td>25.5</td>
      <td>0.3</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>2792</td>
      <td>18.1</td>
      <td>5.0</td>
      <td>8.7</td>
      <td>24.0</td>
      <td>0.1</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>-</td>
      <td>...</td>
      <td>0.4</td>
      <td>4.5</td>
      <td>0.9</td>
      <td>5.0</td>
      <td>0.8</td>
      <td>12.0</td>
      <td>0.1</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2956</td>
      <td>29.2</td>
      <td>6.0</td>
      <td>14.5</td>
      <td>25.0</td>
      <td>0.5</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>9.5</td>
      <td>...</td>
      <td>0.8</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>6.0</td>
      <td>0.3</td>
      <td>10.5</td>
      <td>0.4</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>3058</td>
      <td>40.1</td>
      <td>10.0</td>
      <td>22.5</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>2.4</td>
      <td>11.0</td>
      <td>1.8</td>
      <td>9.5</td>
      <td>0.3</td>
      <td>36.0</td>
      <td>0.2</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17</td>
      <td>3038</td>
      <td>49.3</td>
      <td>13.0</td>
      <td>28.0</td>
      <td>36.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>21.0</td>
      <td>...</td>
      <td>1.4</td>
      <td>6.0</td>
      <td>3.5</td>
      <td>7.0</td>
      <td>2.8</td>
      <td>9.0</td>
      <td>0.6</td>
      <td>48.0</td>
      <td>0.5</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>18</td>
      <td>2469</td>
      <td>58.7</td>
      <td>24.0</td>
      <td>33.7</td>
      <td>52.0</td>
      <td>3.2</td>
      <td>5.0</td>
      <td>0.4</td>
      <td>10.0</td>
      <td>...</td>
      <td>1.7</td>
      <td>7.0</td>
      <td>4.9</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0.5</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>19</td>
      <td>2223</td>
      <td>64.6</td>
      <td>36.0</td>
      <td>33.4</td>
      <td>60.0</td>
      <td>4.1</td>
      <td>5.5</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>...</td>
      <td>1.5</td>
      <td>7.5</td>
      <td>4.2</td>
      <td>4.5</td>
      <td>3.3</td>
      <td>6.0</td>
      <td>0.4</td>
      <td>105.0</td>
      <td>0.3</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20</td>
      <td>2271</td>
      <td>69.7</td>
      <td>48.0</td>
      <td>34.0</td>
      <td>60.0</td>
      <td>4.9</td>
      <td>8.0</td>
      <td>0.6</td>
      <td>5.0</td>
      <td>...</td>
      <td>1.7</td>
      <td>12.0</td>
      <td>5.4</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>0.9</td>
      <td>12.0</td>
      <td>0.5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21</td>
      <td>2354</td>
      <td>83.2</td>
      <td>52.0</td>
      <td>33.0</td>
      <td>52.0</td>
      <td>4.8</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>17.0</td>
      <td>...</td>
      <td>1.3</td>
      <td>13.5</td>
      <td>3.9</td>
      <td>7.0</td>
      <td>4.1</td>
      <td>10.0</td>
      <td>0.6</td>
      <td>2.0</td>
      <td>0.3</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>22-23</td>
      <td>4707</td>
      <td>84.2</td>
      <td>52.0</td>
      <td>28.4</td>
      <td>52.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>5.0</td>
      <td>...</td>
      <td>1.7</td>
      <td>17.5</td>
      <td>4.4</td>
      <td>12.0</td>
      <td>3.6</td>
      <td>10.0</td>
      <td>0.6</td>
      <td>46.0</td>
      <td>0.2</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>24-25</td>
      <td>4591</td>
      <td>83.1</td>
      <td>52.0</td>
      <td>24.9</td>
      <td>60.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>0.5</td>
      <td>6.0</td>
      <td>...</td>
      <td>1.3</td>
      <td>20.0</td>
      <td>4.3</td>
      <td>10.0</td>
      <td>2.6</td>
      <td>10.0</td>
      <td>0.7</td>
      <td>21.0</td>
      <td>0.2</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>26-29</td>
      <td>2628</td>
      <td>80.7</td>
      <td>52.0</td>
      <td>20.8</td>
      <td>52.0</td>
      <td>3.2</td>
      <td>5.0</td>
      <td>0.4</td>
      <td>6.0</td>
      <td>...</td>
      <td>1.2</td>
      <td>13.5</td>
      <td>4.2</td>
      <td>10.0</td>
      <td>2.3</td>
      <td>7.0</td>
      <td>0.6</td>
      <td>30.0</td>
      <td>0.4</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>30-34</td>
      <td>2864</td>
      <td>77.5</td>
      <td>52.0</td>
      <td>16.4</td>
      <td>72.0</td>
      <td>2.1</td>
      <td>8.0</td>
      <td>0.5</td>
      <td>15.0</td>
      <td>...</td>
      <td>0.9</td>
      <td>46.0</td>
      <td>3.6</td>
      <td>8.0</td>
      <td>1.4</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>54.0</td>
      <td>0.4</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>35-49</td>
      <td>7391</td>
      <td>75.0</td>
      <td>52.0</td>
      <td>10.4</td>
      <td>48.0</td>
      <td>1.5</td>
      <td>15.0</td>
      <td>0.5</td>
      <td>48.0</td>
      <td>...</td>
      <td>0.3</td>
      <td>12.0</td>
      <td>1.9</td>
      <td>6.0</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>104.0</td>
      <td>0.3</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>50-64</td>
      <td>3923</td>
      <td>67.2</td>
      <td>52.0</td>
      <td>7.3</td>
      <td>52.0</td>
      <td>0.9</td>
      <td>36.0</td>
      <td>0.4</td>
      <td>62.0</td>
      <td>...</td>
      <td>0.4</td>
      <td>5.0</td>
      <td>1.4</td>
      <td>10.0</td>
      <td>0.3</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>30.0</td>
      <td>0.2</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65+</td>
      <td>2448</td>
      <td>49.3</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>-</td>
      <td>...</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>364.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
<p>17 rows × 28 columns</p>
</div>




```python
# Cleaned up values

drug_use.replace(to_replace="-", value=0.0, inplace=True)
```


```python
# viewed the head of the dataset
drug_use.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>...</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0.1</td>
      <td>24.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.2</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>2757</td>
      <td>8.5</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>15.0</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.1</td>
      <td>41.0</td>
      <td>0.3</td>
      <td>25.5</td>
      <td>0.3</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>2792</td>
      <td>18.1</td>
      <td>5.0</td>
      <td>8.7</td>
      <td>24.0</td>
      <td>0.1</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0.4</td>
      <td>4.5</td>
      <td>0.9</td>
      <td>5.0</td>
      <td>0.8</td>
      <td>12.0</td>
      <td>0.1</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2956</td>
      <td>29.2</td>
      <td>6.0</td>
      <td>14.5</td>
      <td>25.0</td>
      <td>0.5</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>9.5</td>
      <td>...</td>
      <td>0.8</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>6.0</td>
      <td>0.3</td>
      <td>10.5</td>
      <td>0.4</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>3058</td>
      <td>40.1</td>
      <td>10.0</td>
      <td>22.5</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>2.4</td>
      <td>11.0</td>
      <td>1.8</td>
      <td>9.5</td>
      <td>0.3</td>
      <td>36.0</td>
      <td>0.2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
# The shape of the dataframe is 17 rows by 28 columns
drug_use.shape
```




    (17, 28)




```python
# all the objects have been converted to float64
drug_use.dtypes
```




    age                         object
    n                            int64
    alcohol-use                float64
    alcohol-frequency          float64
    marijuana-use              float64
    marijuana-frequency        float64
    cocaine-use                float64
    cocaine-frequency           object
    crack-use                  float64
    crack-frequency             object
    heroin-use                 float64
    heroin-frequency            object
    hallucinogen-use           float64
    hallucinogen-frequency     float64
    inhalant-use               float64
    inhalant-frequency          object
    pain-releiver-use          float64
    pain-releiver-frequency    float64
    oxycontin-use              float64
    oxycontin-frequency         object
    tranquilizer-use           float64
    tranquilizer-frequency     float64
    stimulant-use              float64
    stimulant-frequency        float64
    meth-use                   float64
    meth-frequency              object
    sedative-use               float64
    sedative-frequency         float64
    dtype: object




```python
# Change objects to floats


drug_use['cocaine-frequency'] = drug_use['cocaine-frequency'].astype(float) 
drug_use['crack-frequency'] = drug_use['crack-frequency'].astype(float) 
drug_use['heroin-frequency'] = drug_use['heroin-frequency'].astype(float) 
drug_use['inhalant-frequency'] = drug_use['inhalant-frequency'].astype(float) 
drug_use['oxycontin-frequency'] = drug_use['oxycontin-frequency'].astype(float) 
drug_use['meth-frequency'] = drug_use['meth-frequency'].astype(float) 
```


```python
drug_use.dtypes
```




    age                         object
    n                            int64
    alcohol-use                float64
    alcohol-frequency          float64
    marijuana-use              float64
    marijuana-frequency        float64
    cocaine-use                float64
    cocaine-frequency          float64
    crack-use                  float64
    crack-frequency            float64
    heroin-use                 float64
    heroin-frequency           float64
    hallucinogen-use           float64
    hallucinogen-frequency     float64
    inhalant-use               float64
    inhalant-frequency         float64
    pain-releiver-use          float64
    pain-releiver-frequency    float64
    oxycontin-use              float64
    oxycontin-frequency        float64
    tranquilizer-use           float64
    tranquilizer-frequency     float64
    stimulant-use              float64
    stimulant-frequency        float64
    meth-use                   float64
    meth-frequency             float64
    sedative-use               float64
    sedative-frequency         float64
    dtype: object




```python

```

### 7.2 Do a high-level, initial overview of the data

Get a feel for what this dataset is all about.

Use whichever techniques you'd like, including those from the SAT dataset EDA. The final response to this question should be a written description of what you infer about the dataset.

Some things to consider doing:

- Look for relationships between variables and subsets of those variables' values
- Derive new features from the ones available to help your analysis
- Visualize everything!


```python
drug_use[["alcohol-use", "marijuana-use", "cocaine-use", "crack-use","heroin-use","hallucinogen-use","inhalant-use","pain-releiver-use", "oxycontin-use","tranquilizer-use","stimulant-use","meth-use","sedative-use"]].plot.box(rot=45)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11a662ba8>




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_56_1.png)


### 7.3 Create a testable hypothesis about this data

Requirements for the question:

1. Write a specific question you would like to answer with the data (that can be accomplished with EDA).
2. Write a description of the "deliverables": what will you report after testing/examining your hypothesis?
3. Use EDA techniques of your choice, numeric and/or visual, to look into your question.
4. Write up your report on what you have found regarding the hypothesis about the data you came up with.


Your hypothesis could be on:

- Difference of group means
- Correlations between variables
- Anything else you think is interesting, testable, and meaningful!

**Important notes:**

You should be only doing EDA _relevant to your question_ here. It is easy to go down rabbit holes trying to look at every facet of your data, and so we want you to get in the practice of specifying a hypothesis you are interested in first and scoping your work to specifically answer that question.

Some of you may want to jump ahead to "modeling" data to answer your question. This is a topic addressed in the next project and **you should not do this for this project.** We specifically want you to not do modeling to emphasize the importance of performing EDA _before_ you jump to statistical analysis.

** Question and deliverables**


...


```python
# Code

drug_use.plot('age', 'marijuana-use', kind='bar')
plt.ylabel('Marijuana Use')
plt.title('Marijuana Use')
plt.plot()

drug_use.plot('age', 'marijuana-frequency', kind='bar')
plt.ylabel('Marijuana Frequency')
plt.title('Marijuana Frequency')
plt.plot()

# Marijuana use increases from age 12 to 20, then almost symetrically decreases until 65+.
# Marijuana frequency rapidly increases from age 12 to 20, levels off until 29. It spikes up from 30-34
# and then settles back down from age 50 to 65+. 

```




    []




![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_59_1.png)



![png](/images/project-02-v2-starter-code_files/project-02-v2-starter-code_59_2.png)


**Report**



...

<img src="http://imgur.com/xDpSobf.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 8. Introduction to dealing with outliers

---

Outliers are an interesting problem in statistics, in that there is not an agreed upon best way to define them. Subjectivity in selecting and analyzing data is a problem that will recur throughout the course.

1. Pull out the rate variable from the SAT dataset.
2. Are there outliers in the dataset? Define, in words, how you _numerically define outliers._
3. Print out the outliers in the dataset.
4. Remove the outliers from the dataset.
5. Compare the mean, median, and standard deviation of the "cleaned" data without outliers to the original. What is different about them and why?


```python

```

<img src="http://imgur.com/GCAf1UX.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

### 9. Percentile scoring and spearman rank correlation

---

### 9.1 Calculate the spearman correlation of sat `Verbal` and `Math`

1. How does the spearman correlation compare to the pearson correlation? 
2. Describe clearly in words the process of calculating the spearman rank correlation.
  - Hint: the word "rank" is in the name of the process for a reason!



```python

```

### 9.2 Percentile scoring

Look up percentile scoring of data. In other words, the conversion of numeric data to their equivalent percentile scores.

http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html

http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html

1. Convert `Rate` to percentiles in the sat scores as a new column.
2. Show the percentile of California in `Rate`.
3. How is percentile related to the spearman rank correlation?


```python

```

### 9.3 Percentiles and outliers

1. Why might percentile scoring be useful for dealing with outliers?
2. Plot the distribution of a variable of your choice from the drug use dataset.
3. Plot the same variable but percentile scored.
4. Describe the effect, visually, of coverting raw scores to percentile.


```python

```
