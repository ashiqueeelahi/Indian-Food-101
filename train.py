import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import pickle;

Let us import the data

data = pd.read_csv('../input/indian-food-101/indian_food.csv')

How does the data look? Wanna Explore?

data.head(30)

Hmm. Looks nice. Are there any null values? Let us check

data.isnull().sum()/data.shape[0]*100

Haah!Very Few. We can easily drop them. It will not impact the datasets

data = data.dropna()

Nice. Let us look at our data again.

data.head(20)

**Wait! There are some -1 values. What to do?
I guess we can replace them.**

data = data.replace(to_replace = '-1', method = 'ffill')

data.head(30)

**There are still some -1 values. But we just replaced them. Why they are still in our dataframe?
Humm. Looks like we have only replaced the strings. We now need to replace the int values as well**

data = data.replace(to_replace = -1, method = 'ffill')

data.head(30)

Now the data looks perfect. We can now start to visualize our data
* Let us start with the diet section

data['diet'].unique()

plt.figure(figsize = (16,9))
df_diet_type = data.diet.value_counts().reset_index()
plt.pie(df_diet_type.diet, labels = df_diet_type['index'],autopct='%1.1f%%')
plt.title("Vegetarian vs Non-Vegetarian recipes in dataset")
plt.show()

#plt.show()

**Haha HUge! Looks like everyone is becoming vegetarian these days.
Next, let us explore the cuisine section**

data['course'].unique()

plt.figure(figsize = (20,10))

course = data.groupby('course').size().to_frame(name = "count").reset_index()
sns.barplot(x = 'count', y='course', data = course )

plt.title("Type of Course")
plt.ylabel("course")
plt.xlabel("Count")

plt.show()

**Great. People liking the main course pretty much.
Let us have a look at the flavor section now**

data['flavor_profile'].unique()

plt.figure(figsize = (16,9))

flavor = data.groupby('flavor_profile').size().to_frame(name = "count").reset_index()
sns.barplot(x = 'count', y='flavor_profile', data = flavor )
#sub_index = np.arange(len(flavor))

plt.title("Type of Flavors")
plt.ylabel("Flavor Profile")
plt.xlabel("Count")

plt.show()

**Yeah! Surely we like spices. Thats why, we really dont enjoy arab or traditional europian meals. ;)
How about the regions?**

data['region'].unique()

plt.figure(figsize = (16,9))

region = data.groupby('region').size().to_frame(name = "count").reset_index()
sns.barplot(x = 'count', y='region', data = region )

plt.title("Foods Belonging to Different Regions")
plt.ylabel("Region")
plt.xlabel("Count")

plt.show()

**West is winning. :D
How about the states? Which state seems to be more foody?**

data['state'].unique()

plt.figure(figsize = (20,10))

state = data.groupby('state').size().to_frame(name = "count").reset_index()
sns.barplot(x = 'count', y='state', data = state )

plt.title("Foods Belonging to Different States")
plt.ylabel("State")
plt.xlabel("Count")

plt.show()

**Hah! PANJABIS! Balle balle!!
We Indians spend an imense amount of time in cooking. But are all dishes like that? Let us check**

data['cook_time'].unique()

plt.figure(figsize = (16,9))

df_cook_time = (data.prep_time + data.cook_time).to_frame('total_time').reset_index()
plt.hist(df_cook_time['total_time'],np.arange(5,150,10), rwidth = 0.9)

plt.title("Cooking time")
plt.ylabel("Number of recipes")
plt.xlabel("Time in minutes")

plt.show()

