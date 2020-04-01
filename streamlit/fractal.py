import streamlit as st
import numpy as np
import time

image = st.empty()




def get_at(array, index, default):
    if index < 0: index += len(array)
    if index < 0: raise IndexError('list index out of range')
    return array[index] if index < len(array) else default

def get_state(arr, position):
    prev = get_at(arr, position-1, 1)
    same = get_at(arr, position, 0)
    following = get_at(arr, position+1, 1)

    if prev == following:
        return 255

    return 0


size = 300
array = np.zeros([size, size], dtype=np.uint8)
array[0] = list(map(lambda x: 0 if x < 0.5 else 255, np.random.rand(size, 1)))

for i in range(0, size):
    time.sleep(0.1)
    for j in range(0, size):
        array[i+1,j] = get_state(array[i], j)

    image.image(array, use_column_width=True)


st.button("Re-run")
