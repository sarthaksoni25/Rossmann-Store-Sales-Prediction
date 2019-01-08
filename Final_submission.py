
# coding: utf-8

# In[176]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pickle
plt.rcParams['figure.figsize'] = (12.0, 10.0)


# In[177]:


types = {'StateHoliday': np.dtype(str)}
train = pd.read_csv("train.csv", parse_dates=[2], dtype=types,nrows = 70000)
store = pd.read_csv("store.csv")


# In[178]:


class Information:
    def __init__(self):
        """
        This class give some brief information about the datasets.
        """
        print("Information object created")

    def _get_missing_values(self,data):
        """
        Find missing values of given datad
        :param data: checked its missing value
        :return: Pandas Series object
        """
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)

        #Returning missing values
        return missing_values


# In[179]:


class Preprocess:
    def __init_(self):
        print("Preprocess object created")

    def remove_no_sales(self,train):
        not_open = train[(train['Open'] == 0) & (train['Sales'] != 0)]
        print("No closed store with sales: " + str(not_open.size == 0))
        train = train.loc[train['Sales'] > 0]
        return train

    def date_range(self,train):
        dates = pd.to_datetime(train['Date'], format="%Y%m%d:%H:%M:%S.%f").sort_values()
        dates = dates.unique()
        start_date = dates[0]
        end_date = dates[-1]
        print("Start date: ", start_date)
        print("End Date: ", end_date)
        date_range = pd.date_range(start_date, end_date).values


# In[180]:


p1 = Preprocess()
p1.remove_no_sales(train)
p1.date_range(train)


# In[181]:


class Datavisualisation:
    def __init__(self):
        print ("DataVisualisation object created")

    def sales_per_day(self,train):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        f, ax = plt.subplots(7, sharex=True, sharey=True)
        plt.rcParams['figure.figsize'] = (10.0, 50.0)
        for i in range(1, 8):
            mask = train[train['DayOfWeek'] == i]
            ax[i - 1].set_title("Day {0}".format(i))
            ax[i - 1].scatter(mask['Customers'], mask['Sales'], label=i)

        plt.legend()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.savefig('output1.png', dpi=300, bbox_inches='tight')
        plt.show()



    def sales_per_customer(self,train):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        plt.scatter(train['Customers'], train['Sales'], c=train['DayOfWeek'], alpha=0.8, cmap=plt.cm.get_cmap('plasma'))
        plt.colorbar()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.savefig('output2.png', dpi=300, bbox_inches='tight')
        plt.show()


    def state_holiday(self,train):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        for i in ["0", "a", "b", "c"]:
            data = train[train['StateHoliday'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['Customers'], data['Sales'], label=i)
        plt.legend()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.savefig('output3.png', dpi=300, bbox_inches='tight')
        plt.show()

    def school_holiday(self,train):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        for i in [0, 1]:
            data = train[train['SchoolHoliday'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['Customers'], data['Sales'], label=i)

        plt.legend()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.savefig('output4.png', dpi=300, bbox_inches='tight')
        plt.show()


    def promo(self,train):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        for i in [0, 1]:
            data = train[train['Promo'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['Customers'], data['Sales'], label=i)

        plt.legend()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.savefig('output5.png', dpi=300, bbox_inches='tight')
        plt.show()

    def add_store(self,train,store):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        train['SalesPerCustomer'] = train['Sales'] / train['Customers']
        avg_store = train.groupby('Store')[['Sales', 'Customers', 'SalesPerCustomer']].median()
        avg_store.rename(columns=lambda x: 'Avg' + x, inplace=True)
        store = pd.merge(avg_store.reset_index(), store, on='Store')
        return store
    def store_type(self,store):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        for i in ['a', 'b', 'c', 'd']:
            data = store[store['StoreType'] == i]
            if(len(data) == 0):
                continue
            plt.scatter(data['AvgCustomers'], data['AvgSales'], label=i)
        plt.legend()
        plt.xlabel('Average Customers')
        plt.ylabel('Average Sales')
        plt.savefig('output6.png', dpi=300, bbox_inches='tight')
        plt.show()

    def assortment(self,store):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        for i in ['a', 'b', 'c']:
            data = store[store['Assortment'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['AvgCustomers'], data['AvgSales'], label=i)
        plt.legend()
        plt.xlabel('Average Customers')
        plt.ylabel('Average Sales')
        plt.savefig('output7.png', dpi=300, bbox_inches='tight')
        plt.show()

    def promo2(self,store):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        for i in [0, 1]:
            data = store[store['Promo2'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['AvgCustomers'], data['AvgSales'], label=i)

        plt.legend()
        plt.xlabel('Average Customers')
        plt.ylabel('Average Sales')
        plt.savefig('output8.png', dpi=300, bbox_inches='tight')
        plt.show()

    def fill_na_values(self,store):
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        # fill NaN values
        store["CompetitionDistance"].fillna(-1)
        plt.scatter(store['CompetitionDistance'], store['AvgSales'])

        plt.xlabel('CompetitionDistance')
        plt.ylabel('Average Sales')
        plt.savefig('output9.png', dpi=300, bbox_inches='tight')
        plt.show()
        return store


# In[182]:


class Features:
    def __init__(self):
        print ("Features object created")
    def string_to_int(self,store,train):
        store['StoreType'] = store['StoreType'].astype('category').cat.codes
        store['Assortment'] = store['Assortment'].astype('category').cat.codes
        train["StateHoliday"] = train["StateHoliday"].astype('category').cat.codes
        merged = pd.merge(train, store, on='Store', how='left')
        return merged
    def remove_nan(self,merged):
        NaN_replace = 0
        merged.fillna(NaN_replace, inplace=True)
        merged['Year'] = merged.Date.dt.year
        merged['Month'] = merged.Date.dt.month
        merged['Day'] = merged.Date.dt.day
        merged['Week'] = merged.Date.dt.week
        return merged
    def Month_Competetions(self,merged):
        # Number of months that competition has existed for
        NaN_replace = 0
        merged['MonthsCompetitionOpen'] = 12 * (merged['Year'] - merged['CompetitionOpenSinceYear']) +         (merged['Month'] - merged['CompetitionOpenSinceMonth'])
        merged.loc[merged['CompetitionOpenSinceYear'] == NaN_replace, 'MonthsCompetitionOpen'] = NaN_replace
        return merged
    def Weeks_promo_open(self,merged):
        # Number of weeks that promotion has existed for
        NaN_replace = 0
        merged['WeeksPromoOpen'] = 12 * (merged['Year'] - merged['Promo2SinceYear']) +         (merged['Date'].dt.weekofyear - merged['Promo2SinceWeek'])
        merged.loc[merged['Promo2SinceYear'] == NaN_replace, 'WeeksPromoOpen'] = NaN_replace
        return merged
    def to_int(self,merged):
        toInt = [
            'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'MonthsCompetitionOpen',
            'WeeksPromoOpen'
        ]
        merged[toInt] = merged[toInt].astype(int)
        return merged
    def add_mean(self,train,store):
        med_store = train.groupby('Store')[['Sales', 'Customers', 'SalesPerCustomer']].mean()
        med_store.rename(columns=lambda x: 'Med' + x, inplace=True)
        store = pd.merge(med_store.reset_index(), store, on='Store')
        return store


# In[183]:


# rmpse_scorer = make_scorer(rmspe, greater_is_better = False) # Loss function


# In[184]:


train_info = Information()
train_info._get_missing_values(train)


# In[185]:


store_info = Information()
store_info._get_missing_values(store)


# In[186]:


train_preprocess = Preprocess()
train = train_preprocess.remove_no_sales(train)
train_preprocess.date_range(train)


# In[187]:


EDA = Datavisualisation()
EDA.sales_per_day(train)


# In[188]:


EDA.sales_per_customer(train)


# In[189]:


EDA.state_holiday(train)


# In[190]:


EDA.state_holiday(train)


# In[191]:


EDA.school_holiday(train)


# In[192]:


EDA.promo(train)


# In[193]:


store = EDA.add_store(train,store)


# In[194]:


EDA.store_type(store)


# In[195]:


EDA.assortment(store)


# In[196]:


EDA.promo2(store)


# In[197]:


store = EDA.fill_na_values(store)


# In[198]:


feature_selection = Features()


# In[199]:


store = feature_selection.add_mean(train,store)


# In[200]:


merge = feature_selection.string_to_int(store,train)


# In[201]:


merge = feature_selection.remove_nan(merge)


# In[202]:


merge = feature_selection.Month_Competetions(merge)


# In[203]:


merge = feature_selection.Weeks_promo_open(merge)


# In[204]:


merge = feature_selection.to_int(merge)


# In[205]:


merge.shape
merge['CompetitionDistance'] = np.log(merge['CompetitionDistance'] + 1)



# In[206]:


X = [
    'Store',
    'Customers',
    'CompetitionDistance',

    'Promo',
    'Promo2',

#     'SchoolHoliday',
    'StateHoliday',
    'StoreType',
    'Assortment',

    'AvgSales',
    'AvgCustomers',
    'AvgSalesPerCustomer',

    'MedSales',
    'MedCustomers',
    'MedSalesPerCustomer',

    'DayOfWeek',
    'Week',
    'Day',
    'Month',
    'Year',

    'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear',
    'Promo2SinceWeek',
    'Promo2SinceYear',

#     'MonthsCompetitionOpen',
#     'WeeksPromoOpen'
]
X_train, X_test, y_train, y_test = train_test_split(merge[X], merge['Sales'], test_size=0.1, random_state=10)


# In[207]:


# Error calculating function according to kaggle
def rmspe(y, y_hat):
    return np.sqrt(np.mean(((y - y_hat) / y) ** 2))

rmpse_scorer = make_scorer(rmspe, greater_is_better = False) # Loss function

def score(model, X_train, y_train, y_test, y_hat):
    score = cross_val_score(model, X_train, y_train, scoring=rmpse_scorer, cv=5)
    print('Mean', score.mean())
    print('Variance', score.var())
    print('RMSPE', rmspe(y_test, y_hat))

def plot_importance(model):
    k = list(zip(X, model.feature_importances_))
    k.sort(key=lambda tup: tup[1])

    labels, vals = zip(*k)

    plt.barh(np.arange(len(X)), vals, align='center')
    plt.yticks(np.arange(len(X)), labels)


# In[208]:


import xgboost as xgb

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

xgboost_tree = xgb.XGBRegressor(
    n_estimators = 1000,
    max_depth = 2,
    tree_method = 'exact',
    reg_alpha = 0.05,
    silent = 0,
    random_state = 1023
)
xgboost_tree.fit(X_train[X], np.log1p(y_train),
                 eval_set = [(X_train[X], np.log1p(y_train)), (X_test[X], np.log1p(y_test))],
                 eval_metric = rmspe_xg,
                 early_stopping_rounds = 300
                )
filename="Private_score.sav"
pickle.dump(xgboost_tree, open(filename,"wb"))


# In[209]:


print("Note that this is not in percentage, thus not to scale of graphs above")
xgb.plot_importance(xgboost_tree)


# In[211]:


types = {'StateHoliday': np.dtype(str)}
test = pd.read_csv("test.csv", parse_dates=[3], dtype=types)
feature_selection = Features()
test_merge = feature_selection.string_to_int(store,test)
test_merge = feature_selection.remove_nan(test_merge)
test_merge = feature_selection.Month_Competetions(test_merge)
test_merge = feature_selection.Weeks_promo_open(test_merge)
test_merge = feature_selection.to_int(test_merge)
test_merge.shape
test_merge['CompetitionDistance'] = np.log(test_merge['CompetitionDistance'] + 1)
test = test_merge
y_hat = np.expm1(xgboost_tree.predict(test[X]))
ids = test.Id
df = pd.DataFrame({"Id": ids, 'Sales': y_hat})
df.loc[test['Open'] == 0, 'Sales'] = 0
print "Predicting Sales:"
df.to_csv('submission.csv', index=False)
print "Sales predicted and are stored in submission.csv"
