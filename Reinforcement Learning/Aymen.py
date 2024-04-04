import tkinter as tk
from tkinter import ttk
import numpy as np

grid_size = (4, 3)
end_states = [(1, 2), (3, 1)]
dangerous_states = [(1, 1), (2, 2)]
slip_probability = 0.1
epsilon = 0.1
alpha = 0.2
gamma = 0.9

# Initialize Q-values
Q_values = np.zeros((grid_size[0], grid_size[1], 4))

def take_action(state, action, grid_size):
    # All possible actions
    MOVE_N = 0
    MOVE_S = 1
    MOVE_W = 2
    MOVE_E = 3

    # current position
    current_row, current_col = state

    # Simulate action
    if action == MOVE_N and current_row > 0:
        next_state = (current_row - 1, current_col)
    elif action == MOVE_S and current_row < grid_size[0] - 1:
        next_state = (current_row + 1, current_col)
    elif action == MOVE_W and current_col > 0:
        next_state = (current_row, current_col - 1)
    elif action == MOVE_E and current_col < grid_size[1] - 1:
        next_state = (current_row, current_col + 1)
    else:
        # If the action is not possible, stay in the current state
        next_state = state 

    return next_state

def calculate_reward(state, end_states, dangerous_states):
    if state in end_states:
        reward = 20.0  # Positive reward
    elif state in dangerous_states:
        reward = -50.0  # Negative reward
    else:
        reward = 0.0  # No additional reward for being in a safe state

    return reward

def run_q_learning():
    slip_probability = float(slip_entry.get())
    epsilon = float(epsilon_entry.get())
    num_episodes = int(episodes_entry.get())

    # Q-learning algorithm implementation
    for episode in range(num_episodes):
        # Initialize state
        state = (0, 0)

        while state not in end_states:
            # Choose action based on epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q_values[state[0], state[1]])

            # Simulate slip
            if np.random.rand() < slip_probability:
                action = np.random.randint(4)

            # Take action and observe next state and reward
            next_state = take_action(state, action, grid_size)
            reward = calculate_reward(next_state, end_states, dangerous_states)

            # Update Q-value
            Q_values[state[0], state[1], action] = (1 - alpha) * Q_values[state[0], state[1], action] + \
                                                   alpha * (reward + gamma * np.max(Q_values[next_state[0], next_state[1]]))

            # Move to the next state
            state = next_state

    result_label.config(text="Q-learning completed!")

    # Update the grid display
    update_grid()

    # Display the utility values below the grid
    display_utility()

def update_grid():
    # Update the grid display based on Q-values or other relevant information
    # This is a simple example; you might want to customize it based on your needs
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            q_values_str = [f"{q:.2f}" for q in Q_values[row, col, :]]
            label_text = f"({row}, {col})\nQ-values: {', '.join(q_values_str)}"
            grid_labels[row][col].config(text=label_text)

def display_utility():
    # Calculate utility values as the maximum Q-value for each state
    utility_values = np.max(Q_values, axis=2)

    # Display utility values below the grid
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            label_text = f"({row}, {col})\nUtility: {utility_values[row, col]:.2f}"
            ttk.Label(main_frame, text=label_text, borderwidth=2, relief="solid", width=20, height=5).grid(row=row + 6, column=col, padx=5, pady=5)

# Create main application window
app = tk.Tk()
app.title("Q-learning GUI")

# Create and pack frames
main_frame = ttk.Frame(app, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Slip Probability
slip_label = ttk.Label(main_frame, text="Slip Probability:")
slip_label.grid(row=0, column=0, padx=5, pady=5)

slip_entry = ttk.Entry(main_frame)
slip_entry.grid(row=0, column=1, padx=5, pady=5)

# Epsilon
epsilon_label = ttk.Label(main_frame, text="Epsilon:")
epsilon_label.grid(row=1, column=0, padx=5, pady=5)

epsilon_entry = ttk.Entry(main_frame)
epsilon_entry.grid(row=1, column=1, padx=5, pady=5)

# Number of Episodes
episodes_label = ttk.Label(main_frame, text="Number of Episodes:")
episodes_label.grid(row=2, column=0, padx=5, pady=5)

episodes_entry = ttk.Entry(main_frame)
episodes_entry.grid(row=2, column=1, padx=5, pady=5)

# Run Button
run_button = ttk.Button(main_frame, text="Run Q-learning", command=run_q_learning)
run_button.grid(row=3, column=0, columnspan=2, pady=10)

# Result Label
result_label = ttk.Label(main_frame, text="")
result_label.grid(row=4, column=0, columnspan=2, pady=5)

# Grid Display
grid_frame = ttk.Frame(main_frame)
grid_frame.grid(row=5, column=0, columnspan=2, pady=10)

# Create labels for the grid
grid_labels = [[ttk.Label(grid_frame, text="", borderwidth=2, relief="solid", width=20, height=5) for _ in range(grid_size[1