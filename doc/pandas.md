#Pandas

##Data Structures


###DataFrames

* display first/ last data `df.head()` resp. `df.tail()`

* display columns `df.columns`

* select single column `df['a']` or `df.a`

* select multiple columns `df[['c', 'd']]` 

* select by rows and/or column names `df.loc['columnName']` or `df.loc[rows, ['a', 'b']]`

* select by rows and/or column index ``df.iloc[rows, [1, 2]]`

* filter with logic `df[df['columnname'] > 5]`

* filter string `df.columnna

* select row ranges by index (range is inclusive!) `df[:3]` 

* add a colum by assignment 

~~~
s = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
df['a'] = s

~~~

* transpose rows and columns `df.T`

* create two-dimensianal ndarray `df.values` --> if data types are different, the dType will chosen to match all of them (likely *object*)

* reindex `df = df.reindex(['a', 'b', 'c', 'd', 'e'])`

* remove rows by index from df / s `df.drop(['d', 'c'])`

* remove columns `df.drop(['two', 'four'], axis='columns')` *option* `inplace=True` will not create a new object

**Transform DF to Something**

![Df to other Types](./doc/df_to_something.png)

**Useful Index Methods**
![Index Methods](./doc/index_methods.png)



###Series

* Can be thought of as a fixed-length ordered dict

* get values `s.values`

* get index `s.index`

* select by index `s[0]` or `s['a']`

* select multiple `s[['a', 'b', 'c']]`

* filter numeric values `s[s > 0]`

* check for key `'a' in s`

* creating series `s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])` *index optional*f

* creating series from dict `s = pd.Series(dict)` *possible to pass dict key in wantend order. Default is sorted order.*
 

##Essential Functionality

* replace data by condition `df[df < 5] = 0`

* display unique values in column `pd.DataFrame(df.column.unique())`

* count unique values in df `pd.value_counts(df.columnName)`
* 




