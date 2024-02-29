#Creating a dummy dataset for products with columns such as product_id, product_name, product_price, product_category, product_description, product_rating, product_review_count using faker library and saving it to a csv file using pandas library

# from faker import Faker
# import pandas as pd

# fake = Faker()
# create a list of dictionaries
# products = []
# generated the dummy dataset for products
# for _ in range(1000):  # about 1000 products
#     product = {
#         'product_id': fake.unique.random_number(digits=5),
#         'product_name': fake.catch_phrase(),
#         'product_price': fake.random_int(min=1, max=1000),
#         'product_category': fake.bs(),
#         'product_description': fake.text(),
#         'product_rating': fake.random_int(min=1, max=5),
#         'product_review_count': fake.random_int(min=0, max=1000),
#     }
#     products.append(product)

# convert the list of dictionaries to a pandas DataFrame using DataFrame constructor
# df_products = pd.DataFrame(products)

# df_products
# df_products.to_csv("products.csv")

# store this dataset into mysql database
# import mysql.connector
# from mysql.connector import Error

# creating and connecting to the database
# mydb = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='Py.Programmers.70',
#     database='Products',
#     auth_plugin='mysql_native_password'
# )

# Inserting the data into the database 'products' table
# cursor = mydb.cursor()
# for data in products:
#     cursor.execute("INSERT INTO products(product_id, product_name, product_price, product_category, product_description, product_rating, product_review_count) VALUES(%s, %s, %s, %s, %s, %s, %s)", tuple(data.values()))

# mydb.commit()
# cursor.close()
# mydb.close()

# Importing the necessary modules for traing the model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

data = pd.read_csv('products.csv')

# Selection features and target
X = data[['product_description', 'product_rating', 'product_review_count']]
y = data['product_price']

# There are mostly categoricat values in the features, so we need to convert them to numeric values

# Converting the categorical values to numeric values
# we can use the get_dummies() function from pandas to convert the categorical values to numeric values
#ABOUT get_dummies() function:

# The get_dummies() function in pandas is used for converting categorical variables into dummy/indicator variables. This process is also known as one-hot encoding.
# It will work as follows:

# Input: The function takes a DataFrame or a Series containing categorical variables as input.

# Output: It returns a DataFrame where each categorical variable is replaced by one or more binary columns (dummy variables), indicating the presence of a particular category in each observation.

# Process: For each categorical variable, get_dummies() creates new binary columns, where each column corresponds to a unique category in the original variable. The value in each column is 1 if the category is present for that observation, and 0 otherwise.

X = pd.get_dummies(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

if __name__ == "__main__":
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(mse)

    mae = mean_absolute_error(y_test, y_pred)
    print(mae)
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse = np.sqrt(mse)
    print(rmse)

    # Accuracy of the model
    r2 = r2_score(y_test, y_pred)
    print(r2)

# Decision Tree Regressor
# model2 = DecisionTreeRegressor()
# model2.fit(X_train, y_train)

# y_pred2 = model2.predict(X_test)
# y_pred2

# mse2 = mean_squared_error(y_test, y_pred2)
# mae2 = mean_absolute_error(y_test, y_pred2)
# rmse2 = np.sqrt(mse2)
# r2_2 = r2_score(y_test, y_pred2)
# mse2, mae2, rmse2, r2_2

