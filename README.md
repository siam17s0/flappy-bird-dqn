# flappy-bird-dqn

# Flappy Bird AI using Deep Q-Network (DQN)
This project implements an AI agent that learns to play Flappy Bird using Deep Q-Learning (DQN) with Experience Replay and Target Network stabilization.
The agent starts with random actions and gradually learns an optimal policy by maximizing cumulative rewards through interaction with the environment.

# Key Features
- ✅ Deep Q-Network (DQN) implementation using PyTorch
- ✅ Experience Replay for stable learning
- ✅ Target Network to reduce training instability
- ✅ Epsilon-Greedy exploration strategy
- ✅ Configurable hyperparameters using YAML
- ✅ Automatic model saving based on best performance
- ✅ Training logs for tracking progress

# Algorithm Overview
## Reinforcement Learning Setup
- Environment: FlappyBird-v0 (via Gymnasium)
- State Space: Bird position, velocity, and pipe information
- Action Space:
  - 0 → Do nothing
  - 1 → Flap

## DQN Components
### 1. Experience Replay
Stores past experiences:
(state, action, reward, next_state, done)  

- Helps break correlation between samples
- Improves learning stability

### 2. Target Network
A separate network (target_dqn) is used to compute target Q-values:

- Updated periodically
- Prevents oscillations and divergence

### 3. Epsilon-Greedy Policy
- Starts with high exploration
- Gradually shifts toward exploitation
epsilon = max(epsilon * decay, epsilon_min

### 4. Q-Learning Update Rule
- Q_target = reward + gamma * max(Q_target(next_state))

- Loss is computed using Mean Squared Error (MSE):

Loss = MSE(Q_current, Q_target)

Tech Stack
- Python
- PyTorch
- Gymnasium
- flappy-bird-gymnasium
- NumPy

## Project Structure
.
├── agent.py
├── dqn.py # Neural Network model 
├── experience_replay.py
├── parameters.yaml # Hyperparameters 
├── runs/ │ 
├── *.pt # Saved models
│ └── *.log # Training logs 
├── README.md


# Installation
- git clone https://github.com/siam17s0/flappy-bird-dqn.git  
-  cd flappy-bird-dqn
-  pip install -r requirements.txt

# Usage

## Train the Agent
-python main.py flappyBirdv0 --train

## Run Trained Model
- python main.py flappyBirdv0

## Hyperparameters

All hyperparameters are defined in:

parameters.yaml

### Example parameters:

- Learning rate (alpha)
- Discount factor (gamma)
- Epsilon decay settings
- Replay memory size
- Mini-batch size
- Target network sync rate
### 📊 Training Behavior
- Initially performs random actions
- Gradually learns to survive longer
- Achieves higher rewards over time

Model is automatically saved when a new best reward is achieved.

## Model Saving
- Best model saved at:
runs/flappyBirdv0.pt
-Training logs:
runs/flappyBirdv0.log

## Future Improvements
- Implement Double DQN
- Add Prioritized Experience Replay
- Use CNN for visual input instead of state vectors
- Hyperparameter tuning automation

## Contributing

Feel free to fork and improve the project!


