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

```python

import pandas as pd

data = [
    {"date": "28-03-2020", "cases":3900},
    {"date": "27-03-2020", "cases":3400},
    {"date": "26-03-2020", "cases":3000},
    {"date": "25-03-2020", "cases":2600},
    {"date": "24-03-2020", "cases":2200},
    {"date": "23-03-2020", "cases":1900},
    {"date": "22-03-2020", "cases":1500},
    {"date": "21-03-2020", "cases":1000},
]


df = pd.DataFrame.from_dict(data)

df.hist()


```

```python pycharm={"is_executing": false}
# 4 days to double



```
