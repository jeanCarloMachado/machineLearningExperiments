#!/usr/bin/env python
import grimoire as g
import random
import pandas as pd


colors = {'Yellow', 'Green', 'Red'}
states = {f'state{i}' for i in range(0, 25)}


lstates, lcolors, lcounts = [], [], []

for i in range(0, 1000000):
    lstates.append(random.sample(states, 1)[0])
    lcolors.append(random.sample(colors, 1)[0])
    lcounts.append(random.randint(0, 20000))



df = pd.DataFrame(data={'state': lstates, 'color': lcolors, 'count': lcounts})
df.to_csv('./data.csv')

