import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Regression Intuition Game", layout="wide", initial_sidebar_state="expanded")  # Options: "auto", "expanded", "collapsed")
st.title("🎯 The Linear Regression Challenge")
st.write("Can you find the hidden 'True Line' just by looking at the noise?")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Generate the Data")
n_points = st.sidebar.slider("Number of points", 10, 200, 50)
noise_level = st.sidebar.slider("Noise (Error) Level", 0.1, 5.0, 1.5)

# Secret Formula parameters (Hidden from user or fixed)
if 'true_a' not in st.session_state:
    st.session_state.true_a = np.round(np.random.uniform(-3, 3), 2)
    st.session_state.true_b = np.round(np.random.uniform(-5, 5), 2)

st.sidebar.header("2. Your Guess")
user_a = st.sidebar.slider("Your Slope (a)", -5.0, 5.0, 0.0, 0.1)
user_b = st.sidebar.slider("Your Intercept (b)", -10.0, 10.0, 0.0, 0.1)

# --- DATA GENERATION ---
np.random.seed(42) # For consistent points while sliding
x = np.linspace(0, 10, n_points)
error = np.random.normal(0, noise_level, n_points)
y_true_dots = st.session_state.true_a * x + st.session_state.true_b + error

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter the dots
ax.scatter(x, y_true_dots, color='lightgray', alpha=0.7, label="Data Points (Secret Formula + Noise)")

# Calculate plot bounds
x_range = np.array([0, 10])
y_user = user_a * x_range + user_b

# Plot the user's line
ax.plot(x_range, y_user, color='#FF4B4B', linewidth=3, label="Your Regression Line")

# Formatting
ax.set_ylim(min(y_true_dots) - 5, max(y_true_dots) + 5)
ax.set_xlim(0, 10)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig)

# --- CALCULATING THE SCORE ---
# Mean Squared Error (MSE)
y_pred = user_a * x + user_b
mse = np.mean((y_true_dots - y_pred)**2)

st.divider()
st.subheader(f"Current Error (MSE): {mse:.4f}")

if st.button("Reveal Secret Formula"):
    st.write(f"The hidden formula was:  **y = {st.session_state.true_a}x + {st.session_state.true_b}**")
    if mse < (noise_level**2 + 0.5):
        st.balloons()
        st.success("Incredible fit! You've got a human-brain neural network.")

import plotly.graph_objects as go

st.title("🎢 The Error Surface Explorer")



# --- CALCULATE ERROR SURFACE ---
# Create a grid of possible a and b values
a_range = np.linspace(st.session_state.true_a - 3, st.session_state.true_a + 3, 40)
b_range = np.linspace(st.session_state.true_b - 5, st.session_state.true_b + 5, 40)
A, B = np.meshgrid(a_range, b_range)

# Function to calculate MSE for the entire grid
def calculate_mse_grid(A_grid, B_grid, x_vals, y_vals):
    mse_grid = np.zeros(A_grid.shape)
    for i in range(A_grid.shape[0]):
        for j in range(A_grid.shape[1]):
            predictions = A_grid[i,j] * x_vals + B_grid[i,j]
            mse_grid[i,j] = np.mean((y_vals - predictions)**2)
    return mse_grid

Z = calculate_mse_grid(A, B, x, y_true_dots)

# Current User Error
current_mse = np.mean(((user_a * x + user_b) - y_true_dots)**2)
from plotly.subplots import make_subplots

# --- 1. DEFINE THE DUAL-LAYOUT ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.5, 0.5],
    # Important: This tells Plotly which subplot handles 3D vs 2D
    specs=[[{"type": "surface"}, {"type": "xy"}]],
    horizontal_spacing=0.08,
    subplot_titles=("3D Error Surface", "2D Precision Map")
)

# --- 2. ADD THE 3D SURFACE (Left Side) ---
fig.add_trace(go.Surface(
    x=a_range, y=b_range, z=Z, 
    colorscale='Viridis', 
    showscale=False,
    contours_z=dict(show=True, usecolormap=True, project_z=True)
), row=1, col=1)

# Add the 3D Red Marker
fig.add_trace(go.Scatter3d(
    x=[user_a], y=[user_b], z=[current_mse],
    mode='markers',
    marker=dict(size=10, color='red', symbol='diamond', line=dict(color='white', width=2))
), row=1, col=1)

# --- 3. ADD THE 2D CONTOUR (Right Side) ---
fig.add_trace(go.Contour(
    x=a_range, y=b_range, z=Z, 
    colorscale='Viridis',
    showlegend=False,
    contours=dict(coloring='heatmap', showlabels=True)
), row=1, col=2)

# Add the 2D "X" Marker
fig.add_trace(go.Scatter(
    x=[user_a], y=[user_b],
    mode='markers',
    marker=dict(size=15, color='red', symbol='x', line=dict(color='white', width=2))
), row=1, col=2)

# Optional: Add a target dot at the mathematical minimum
fig.add_trace(go.Scatter(
    x=[st.session_state.true_a], y=[st.session_state.true_b],
    mode='markers',
    marker=dict(size=12, color='lime', symbol='circle-open', line=dict(width=3)),
    name="Minimum"
), row=1, col=2)

# --- 4. FORMATTING ---
fig.update_layout(
    height=550,
    margin=dict(l=10, r=10, t=50, b=10),
    uirevision='constant', # Keeps the camera from jumping
    showlegend=False
)

fig.update_xaxes(title_text="Slope (a)", row=1, col=2)
fig.update_yaxes(title_text="Intercept (b)", row=1, col=2)

st.plotly_chart(fig, use_container_width=True)