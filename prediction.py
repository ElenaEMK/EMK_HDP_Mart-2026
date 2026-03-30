#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import joblib


# 1. Загрузка модели, колонок и threshold

model = joblib.load("MODEL/best_model.pkl")
threshold = joblib.load("MODEL/threshold.pkl")
train_cols = joblib.load("MODEL/columns.pkl")


# 2. Загрузка тестового набора данных

df_test = pd.read_csv("DATA/test.csv")

ids = df_test["ID"]


# 3. Preprocessing

# chest
df_test['chest'] = df_test['chest'].round().clip(1, 4).astype(int)

# удаление лишнего
cols_to_drop = ['ID', 'fasting_blood_sugar']
df_test = df_test.drop(columns=cols_to_drop)

# категориальные признаки
cat_cols = [
    'chest',
    'resting_electrocardiographic_results',
    'slope',
    'thal']

# кодирование one-hot
df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)

# типы колонок

cols = df_test.filter(regex='chest|resting_electrocardiographic_results|slope|thal').columns
df_test[cols] = df_test[cols].astype(int)


# 4. Выравнивание колонок

for col in train_cols:
    if col not in df_test.columns:
        df_test[col] = 0

df_test = df_test[train_cols]


# 5. Предсказание с threshold

proba = model.predict_proba(df_test)[:, 1]
preds = (proba >= threshold).astype(int)


# 6. Сохранение

submission = pd.DataFrame({
    "ID": ids,
    "target": preds
})

submission.to_csv("DATA/submission1.csv", index=False)

