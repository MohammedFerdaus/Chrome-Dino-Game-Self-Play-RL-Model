# Chrome-Dino-Game-Self-Play-RL-Model

Features
- CNN-based DQN with 3 convolutional layers
- Visual input processing - 4 stacked grayscale 80x80 frames
- Target network for stable training
- Experience replay with 50,000 memory buffer
- Automatic checkpoint saving every 50 games
- Real-time stats - score, high score, games played, and training time

Architecture

DQN Network
- Input: 4 stacked grayscale frames (4, 80, 80)
- Conv Layer 1: 32 filters, 8x8 kernel, stride 4
- Conv Layer 2: 64 filters, 4x4 kernel, stride 2
- Conv Layer 3: 64 filters, 3x3 kernel, stride 1
- Fully Connected 1: 2304 → 512 neurons
- Output Layer: 512 → 3 Q-values (do nothing, jump, duck)

Hyperparameters
- Learning rate: 0.00025
- Gamma (discount): 0.99
- Epsilon start: 1.0, min: 0.1, decay: 0.9995
- Batch size: 32
- Memory size: 50,000
- Target network update: every 1000 steps

Reward System
- +1 for each obstacle passed
- -1 for collision (game over)

Training Details
- Training time: Requires thousands of episodes for good performance
- Expected results: Progressive improvement visible after ~500-1000 games
- Convergence: May take several hours to days depending on hardware
- GPU recommended for faster training
