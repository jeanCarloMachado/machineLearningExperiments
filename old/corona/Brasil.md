---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python pycharm={"is_executing": false}

import pandas as pd
import matplotlib.pyplot as plt


data = [
    {"date": "03-21", "cases":1000},
    {"date": "03-22", "cases":1500},
    {"date": "03-23", "cases":1900},
    {"date": "03-24", "cases":2200},
    {"date": "03-25", "cases":2600},
    {"date": "03-26", "cases":3000},
    {"date": "03-27", "cases":3400},
    {"date": "03-28", "cases":3900},
]


df = pd.DataFrame.from_dict(data)

plt.plot(df['date'], df['cases'])
plt.show()

```

<!-- #region pycharm={"is_executing": false} -->
# 4 days to double 
in pycharm

<!-- #endregion -->


```python
from datetime import datetime,timedelta


def growth(cases):
    return (2 * cases) / 3

total_cases = 3500

def pop_affected(total_cases):
    now = datetime.now()
    data = []
    for i in range(1, 15):
        new_cases = growth(total_cases)
        total_cases += new_cases


        date = now + timedelta(days=i)
        data.append({
            "date": date.strftime("%m-%d"), "cases": total_cases
        })
        if total_cases >=  200000000:
            break

    return data


plt.rcParams["figure.figsize"] = (20,7)

affected_data = pop_affected(total_cases)
df2 = pd.DataFrame.from_dict(affected_data)

dates = df['date'].to_list() + df2['date'].to_list()
cases = df['cases'].to_list() + ([None] * len(df2['date'].to_list()))
pred = ([None] * len(df['date'].to_list())) + df2['cases'].to_list() 
plt.plot(dates, cases)
plt.plot(dates, pred)
plt.show()
```

```python pycharm={"is_executing": false}
from statsmodels.tsa.arima_model import ARMA

model = ARMA(df["cases"], order=(0,1), dates=df["date"], freq="D")
fit=model.fit(disp=False)
result = fit.predict("03-29", "04-15")
print(result)

```
```python pycharm={"is_executing": false}
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(df['cases'])
fit=model.fit()
result = fit.predict(20, 1000)
print(result)

```

```python

```
