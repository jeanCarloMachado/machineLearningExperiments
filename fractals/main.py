import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


"""
# Play with fractals

"""


for i in range(0, 100):
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    circle1 = plt.Circle((0.5, 0.5), 0.01 * i)
    ax.add_artist(circle1)
    st.pyplot()
