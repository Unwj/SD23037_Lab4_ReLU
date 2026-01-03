import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Modular function to compute ReLU
def relu(x):
    """Compute ReLU: f(x) = max(0, x)"""
    return np.maximum(0, x)

# App title and AI relevance explanation
st.title("ReLU Activation Function Visualizer")
st.write("""
ReLU (Rectified Linear Unit) introduces non-linearity by outputting x if x > 0, else 0. 
In neural networks, it helps solve complex problems like image classification by preventing vanishing gradients, 
making deep networks trainable. Without it, the model would be linear and limited in capability.
""")

# Display mathematical formula using LaTeX
st.latex(r"f(x) = \max(0, x)")

# Interactive inputs in columns for better layout
col1, col2 = st.columns(2)
with col1:
    min_x = st.slider("Minimum Input (x)", -20.0, 0.0, -10.0)
with col2:
    max_x = st.slider("Maximum Input (x)", 0.0, 20.0, 10.0)

# Generate and plot data
x = np.linspace(min_x, max_x, 500)
y = relu(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='ReLU', color='blue')
ax.set_xlabel('Input (x)')
ax.set_ylabel('Output f(x)')
ax.set_title('ReLU Function Plot')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax.grid(True)
ax.legend()
st.pyplot(fig)