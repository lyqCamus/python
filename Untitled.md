
本项目提供了两份数据：train.csv文件作为训练构建与生存相关的模型；另一份test.csv文件则用于测试集，用我们构建出来的模型预测生存情况；

PassengerId --Id,具有唯一标识的作用，即每个人对应一个Id
survived --是否幸存 1表示是 0表示否
pclass --船舱等级 1:一等舱 2:二等舱 3:三等舱
Name --姓名，通常西方人的姓名
Sex --性别，female女性 male 男性
Age --年龄
SibSp --同船配偶以及兄弟姐妹的人数
Parch --同船父母或子女的人数
Ticket --船票
Fare --票价
Cabin --舱位
Embarked --登船港口


```python
#读取数据
import pandas as pd
df_train,df_test = pd.read_csv('train.csv'),pd.read_csv('test.csv')
```

从训练集开始


```python
#查看前五行数据
df_train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看后5行数据
df_train.tail()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看数据信息，其中包含数据纬度、数据类型、所占空间等信息
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB


数据纬度：891行 X 12列
缺失字段：Age，Cabin，Embarked
数据类型：两个64位浮点型，5个64位整型，5个python对象



```python
#描述性统计
df_train.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



1.除了python对象之外的数据类型，均参与了计算
2.38.4%的人幸存，死亡率很高
3.年龄现有数据714，缺失占比714/891 = 20%
4.同船兄弟姐妹与配偶人数最大为8，同船父母或子女最大数则为6，看来有大家庭小家庭之分；
5.票价最小为0，最大为512.3，均值为32.20，中位数为14.45，正偏，贫富差距不小；


```python
#那么python对象对应的数据查看
df_train[['Name','Sex','Ticket','Cabin','Embarked']].describe()
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
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>891</td>
      <td>2</td>
      <td>681</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Lam, Mr. Ali</td>
      <td>male</td>
      <td>1601</td>
      <td>B96 B98</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>577</td>
      <td>7</td>
      <td>4</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>



# 特征分析


```python
# 1.PassengerId，id仅仅是来标识乘客的唯一性，必然与幸存无关
```


```python
# 2.Pclass
#船舱等级，一等级是整个船最昂贵奢华的地方，有钱人才能享受，有没有可能一等舱有钱人比三等舱的穷人更容易幸存呢？
```


```python
import numpy as np
import matplotlib.pyplot as plt
#生成pclass-survive的列联表
Pclass_Survived = pd.crosstab(df_train['Pclass'],df_train['Survived'])
```


```python
Pclass_Survived
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
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>80</td>
      <td>136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>97</td>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>372</td>
      <td>119</td>
    </tr>
  </tbody>
</table>
</div>




```python
Pclass_Survived.count()
```




    Survived
    0    3
    1    3
    dtype: int64




```python
Pclass_Survived.index
```




    Int64Index([1, 2, 3], dtype='int64', name='Pclass')




```python
#绘制堆积柱形图
Pclass_Survived.plot(kind = 'bar',stacked = True)
Survived_len = len(Pclass_Survived.count())
print(Survived_len)
Pclass_index = np.arange(len(Pclass_Survived.index))
print(Pclass_index)

plt.xticks(Pclass_Survived.index-1,Pclass_Survived.index,rotation = 360)
plt.title('Survived status by pclass')
```

    2
    [0 1 2]





    Text(0.5,1,'Survived status by pclass')




![png](output_18_2.png)


其中列联表就等于一下操作


```python
#生成Survived为0时，每个Pclass的总计数
Pclass_Survived_0 = df_train.Pclass[df_train['Survived'] == 0].value_counts()
#生成Survived为1时，每个Pclass的总计数
Pclass_Survived_1 = df_train.Pclass[df_train['Survived'] == 1].value_counts()
#将两个状况合并为一个dataFram
Pclass_Survived = pd.DataFrame({0:Pclass_Survived_0,1:Pclass_Survived_1})
Pclass_Survived
```

    3    372
    2     97
    1     80
    Name: Pclass, dtype: int64





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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>80</td>
      <td>136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>97</td>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>372</td>
      <td>119</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Name
#姓名，总数为891个且有891种不同的结果，没多大意义，但值得注意的是性命中有头衔存在的，头衔是身份地位的象征，是否身份地位越高更容易生存？
#首先提取出头衔
import re
df_train['Appellation'] = df_train.Name.apply(lambda x : re.search('\w+\.',x).group()).str.replace('.','')
#查看多钟不同的结果
df_train.Appellation.unique()
```




    array(['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
           'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
           'Jonkheer'], dtype=object)



#头衔解读：Mr既可用于已婚男性，也可用于未婚男性；Mrs已婚女性；Miss通常用来称呼未婚女性，但有时也用于称呼自己不了解的年龄较大的妇女；
#Master： 男童或男婴；Don：大学教师；Rev： 牧师；Dr：医生或博士；Mme：女士；Ms：既可用于已婚女性也可用于未婚女性；Major：陆军少校；
#Lady ： 公侯伯爵的女儿；Sir：常用来称呼上级长官；Mlle：小姐；Col：上校；Capt：船长；Countess：伯爵夫人；Jonkheer：乡绅；


```python
#性别与头衔的对应的人数
Appellation_Sex = pd.crosstab(df_train.Appellation,df_train.Sex)
Appellation_Sex.T
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
      <th>Appellation</th>
      <th>Capt</th>
      <th>Col</th>
      <th>Countess</th>
      <th>Don</th>
      <th>Dr</th>
      <th>Jonkheer</th>
      <th>Lady</th>
      <th>Major</th>
      <th>Master</th>
      <th>Miss</th>
      <th>Mlle</th>
      <th>Mme</th>
      <th>Mr</th>
      <th>Mrs</th>
      <th>Ms</th>
      <th>Rev</th>
      <th>Sir</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>182</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>125</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>male</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>517</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Appellation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将少数部分用Rare表示，将‘Mlle’，‘Ms’用‘MIss’代替，将‘Mme’用‘Mrs’代替
df_train['Appellation'] = df_train['Appellation'].replace(['Capt','Col','Countess','Don',
                                                           'Dr','Jonkheer','Lady','Major','Rev','Sir'],'Rare')
df_train['Appellation'] = df_train['Appellation'].replace(['Mlle','Ms'],'Miss')
df_train['Appellation'] = df_train['Appellation'].replace('Mme','Mrs')
df_train.Appellation.unique()
```




    array(['Mr', 'Mrs', 'Miss', 'Master', 'Rare'], dtype=object)




```python
#头衔和幸存者相关吗？
#绘制柱状图
Appellation_Survived = pd.crosstab(df_train['Appellation'],df_train['Survived'])
Appellation_Survived.plot(kind = 'bar')
print(np.arange(len(Appellation_Survived.index)-1))
plt.xticks(np.arange(len(Appellation_Survived.index)),Appellation_Survived.index,rotation = 360)
plt.title('Survived status by Appelation')
```

    [0 1 2 3]





    Text(0.5,1,'Survived status by Appelation')




![png](output_26_2.png)


# Sex


```python
#性别，女士优先，但这种紧急关头，会让女士先上救生艇吗
```


```python
#生成列联表
Sex_Survived = pd.crosstab(df_train['Sex'],df_train['Survived'])
Survived_len = len(Sex_Survived.count())
print(Survived_len)
Sex_index = np.arange(len(Sex_Survived.index))
print(Sex_survived.index)
print(Sex_index)
single_width = 0.35

for i in range(Survived_len):
    SurvivedName = Sex_Survived.columns[i]
    print(SurvivedName)
    SexCount = Sex_Survived[SurvivedName]
    print(SexCount)
    SexLocation = Sex_index * 1.05 + (i - 1/2)*single_width
    print(SexLocation)
    
    #绘制柱状图
    plt.bar(SexLocation,SexCount,width = single_width)
    for x,y in zip(SexLocation,SexCount):
        #添加数据标签
        plt.text(x,y,'%.0f'%y,ha = 'center',va= 'bottom')
index = Sex_index * 1.05
plt.xticks(index,Sex_Survived.index,rotation = 360)
plt.title('Survived status by sex')
    
```

    2
    Index(['female', 'male'], dtype='object', name='Sex')
    [0 1]
    0
    Sex
    female     81
    male      468
    Name: 0, dtype: int64
    [-0.175  0.875]
    1
    Sex
    female    233
    male      109
    Name: 1, dtype: int64
    [0.175 1.225]





    Text(0.5,1,'Survived status by sex')




![png](output_29_2.png)



```python
#结果可以看出，女性的幸存率远高于男性
```

# Age


```python
#由于Age特征存在缺失值，处理完缺失值，再对其进行分析

```

# SibSp --同船配偶以及兄弟姐妹的人数


```python
#从之前的描述性统计了解到，兄弟姐妹与配偶的人数最多的为8，最少为0，哪个更容易生存呢？

```


```python
#生成列联表
SibSp_Survived = pd.crosstab(df_train['SibSp'],df_train['Survived'])
#print(SibSp_Survived)
#print(np.arange(len(SibSp_Survived.index)))
SibSp_Survived.plot(kind = 'bar')
plt.xticks(np.arange(len(SibSp_Survived.index)),SibSp_Survived.index,rotation = 360)
plt.title('Survived status by SibSp')

```




    Text(0.5,1,'Survived status by SibSp')




![png](output_35_1.png)


# Parch --同船父母或子女的人数


```python
#通过上面的描述性统计了解到，同样也可以分为大家庭，小家庭

Parch_Survived = pd.crosstab(df_train['Parch'],df_train['Survived'])
Parch_Survived.plot(kind = 'bar')
plt.xticks(np.arange(len(Parch_Survived.index)),Parch_Survived.index,rotation = 360)
plt.title('Survived status by Parch')
```




    Text(0.5,1,'Survived status by Parch')




![png](output_37_1.png)



```python
Parch_Survived = pd.crosstab(df_train[df_train.Parch >= 3]['Parch'],df_train['Survived'])
Parch_Survived.plot(kind = 'bar')
plt.xticks(np.arange(len(Parch_Survived.index)),Parch_Survived.index,rotation = 360)
plt.title('Survived status by Parch')
```




    Text(0.5,1,'Survived status by Parch')




![png](output_38_1.png)



```python
##可以看到，大部分Parch为0，幸存率不大，当为1，2，3时，有所增加，再往上又有所减小
```

# Ticket --船票


```python
#总人数891，船票有681种，说明部分人共用一张票，什么人能共用一张票呢？需要对他们进行归类，共用票的归位一类，独自使用的归位一类；
```


```python
#计算每张船票的使用的人数
Ticket_Count = df_train.groupby('Ticket',as_index = False)['PassengerId'].count()
#获取使用人数为1的船票
Ticket_Count_0 = Ticket_Count[Ticket_Count.PassengerId == 1]['Ticket']
#当船票在已经筛选出使用人数为1的船票中时，将0赋值给GroupTicket，否则将1赋值给GroupTicket
df_train['GroupTicket'] = np.where(df_train.Ticket.isin(Ticket_Count_0),0,1)
#绘制柱形图
GroupTicket_Survived = pd.crosstab(df_train['GroupTicket'],df_train['Survived'])
GroupTicket_Survived.plot(kind = 'bar')
plt.xticks(GroupTicket_Survived.index,rotation = 360)
plt.title('Survived status by GroupTicket')

```




    Text(0.5,1,'Survived status by GroupTicket')




![png](output_42_1.png)



```python
#很明显，船上有同伴比孤身一人幸存机会大
```

# Fare --票价


```python
#对Fare进行分组，2**10>891 分成10组，组距为（最大值-最小值）/10取值60
bins = [0,60,120,180,240,300,360,420,480,540,600]
df_train['GroupFare'] = pd.cut(df_train.Fare,bins,right = False)
GroupFare_Survived = pd.crosstab(df_train['GroupFare'],df_train['Survived'])
GroupFare_Survived
GroupFare_Survived.plot(kind = 'bar')
plt.title('Survived status by GroupFare')

GroupFare_Survived.iloc[2:].plot(kind = 'bar')
plt.title('Survived status by GroupFare(Fare > 120)')
```




    Text(0.5,1,'Survived status by GroupFare(Fare > 120)')




![png](output_45_1.png)



![png](output_45_2.png)



```python
#可以看到随着票价的增长，幸存机会也会变大
```

# Cabin --舱位   #Embarked --登船港口



```python
#由于含有大量缺失值，处理完缺失值再对其进行分析
```

# 四.特征工程


缺失值主要是由人为原因和机械原因造成的数据缺失，在pandas中用NaN或者NaT表示，它的处理方式有多种：
1.用某些集中趋势度量（平均数，众数）进行对缺失值进行填充；
2.用统计模型来预测缺失值，比如回归模型、决策树、随即森林；
3.删除缺失值；
4.保留缺失值；



```python
#在处理缺失值之前，应当将数据拷贝一份，以保证原始数据的完整性；
train = df_train.copy()
```

1.Embarked缺失值处理
通过以上，我们已经知道Embarked字段中缺失2个，且数据中S最多，达到644，占比644/891=72%，那么就采用众数进行填充；


```python
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0]) 
# 0 or index :获取列的众数；1 or columns ：获取行的众数
```

2.Cabin缺失值处理
Cabin缺失值687个，占比687/891=77%，缺失数据太多，是否删除呢？舱位缺失可能代表这些人没有舱位，不妨用‘NO’来填充；


```python
train['Cabin'] = train['Cabin'].fillna('NO')
```

3.Age缺失值处理
Age缺失177个，占比177/891=20%，缺失数据也不少，而且Age在本次分析中也尤其重要，孩子和老人属于弱势群体，应当更容易获救，不能删除也不能保留；
采用头衔相对应的年龄中位数进行填充


```python
#求出每个头衔对应的年龄的中位数
Age_Appellation_median = train.groupby('Appellation')['Age'].median()
#在当前表设置Appellation为索引
train.set_index('Appellation',inplace = True)
#在当前表填充缺失值
train.Age.fillna(Age_Appellation_median,inplace = True)
#重置索引
train.reset_index(inplace = True)

```

#检查一下是否有缺失值


```python
#第一种方法：返回0即表示没有缺失值
train.Age.isnull().sum()
```




    0




```python
#第二种方法：返回False即表示没有缺失值
train.Age.isnull().any()
```




    False




```python
#第三种方法：描述性统计
train.Age.describe()
```




    count    891.000000
    mean      29.390202
    std       13.265322
    min        0.420000
    25%       21.000000
    50%       30.000000
    75%       35.000000
    max       80.000000
    Name: Age, dtype: float64



# 对缺失特征分析

Embarked --登船港口


```python
#绘制柱状图
Embarked_Survived = pd.crosstab(train['Embarked'],train['Survived'])
Embarked_Survived.plot(kind = 'bar')
plt.xticks(np.arange(len(Embarked_Survived.index)),Embarked_Survived.index,rotation = 360)
plt.title('Survived status by Embarked')
```




    Text(0.5,1,'Survived status by Embarked')




![png](output_64_1.png)



```python
#C港生存几率会明显高于Q，S港
```

#Cabin


```python
#将没有舱位的归为0，有舱位的归位1
train['GroupCabin'] = np.where(train.Cabin == 'NO',0,1)
#绘制柱状图
GroupCabin_Survived = pd.crosstab(train['GroupCabin'],train['Survived'])
GroupCabin_Survived.plot(kind = 'bar')
plt.xticks(np.arange(len(GroupCabin_Survived.index)),GroupCabin_Survived.index,rotation = 360)
plt.title('Survived ststus by GroupCabin')
```




    Text(0.5,1,'Survived ststus by GroupCabin')




![png](output_67_1.png)



```python
#有舱位比没有舱位的生存几率大
```

Age


```python
#对Age进行分组
bins = [0,9,18,27,36,45,54,63,72,81,90]
train['GroupAge'] = pd.cut(train.Age,bins)
GroupAge_Survived = pd.crosstab(train['GroupAge'],train['Survived'])
#绘制柱状图
GroupAge_Survived.plot(kind = 'bar')
plt.xticks(np.arange(len(GroupAge_Survived.index)),GroupAge_Survived.index,rotation = 90)
plt.title('Survived status by GroupAge')
```




    Text(0.5,1,'Survived status by GroupAge')




![png](output_70_1.png)



```python
#如图，孩子的幸存几率很大
```
