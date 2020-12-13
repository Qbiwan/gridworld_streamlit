import streamlit as st
from agent import Agent
from maze import Maze
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

a = [0, 0, 0, 0, 0, 0,
     0, 0, 1, 1, 1, 1,
     0, 0, 1, 0, 0, 1,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 2]

st.set_page_config(layout="wide")
col1 = st.sidebar
col2, col3 = st.beta_columns([5,5])

with col1:
    st.subheader("User Input Features")
    num_rows = st.selectbox("Number of rows", list((range(0, 10))), 6)
    num_cols = st.selectbox("Number of columns", list((range(0, 10))), 6)
total_range = np.arange(num_rows*num_cols).reshape(num_rows,
                                                   num_cols
                                                   ).astype(np.float)

select_wall = col1.multiselect("Build wall",
                               list((range(1, (num_rows*num_cols)-1))),
                               [8, 9, 10, 11, 14, 17, 23, 20, 31, 32, 33, 34]
                               )


wall_row = [i // num_rows for i in select_wall]
wall_col = [i % num_rows for i in select_wall]
total_range[wall_row, wall_col] = np.nan

df = pd.DataFrame(total_range)
f, ax = plt.subplots(figsize=(5, 3))
mask = df.isnull()
ax = sns.heatmap(df,
                 mask=mask,
                 linewidths=0.5,
                 vmin=0,
                 vmax=num_rows*num_cols,
                 cmap="autumn",
                 annot=True,
                 cbar=False,
                 )
ax.set_facecolor("black")
col1.pyplot(f)

#col2.write(mask)
maze = Maze(num_rows, num_cols)
# agent = Agent(maze)
# f, ax = agent.pretraining_heatmap()
# col2.pyplot(f)

maze.build_wall(mask=mask)
#col2.write(maze.grid)
agent = Agent(maze)
f, ax = agent.pretraining_heatmap()
col2.pyplot(f)

#col3.pyplot(maze.grid)
#maze = Maze(6, 6)
#maze.set_wall(a)

# agent = Agent(maze)
# f, ax = agent.pretraining_heatmap()
# col2.pyplot(f)

#agent.learn()
#f2, ax2 = agent.post_training_heatmap()
#col3.pyplot(f2)

if col2.button('Start Reinforcement Learning'):
    col2.header('Training bot to find shortest path...')
    # progress_bar = col2.progress(0)
    # status_text = st.empty()
    # status_text.text("Starting training...")

    # for i in range(100):
    #     progress_bar.progress(i + 1)
    #     agent.learn_one_episode()

    agent.learn()
    f2, ax2 = agent.post_training_heatmap()
    col3.pyplot(f2)

