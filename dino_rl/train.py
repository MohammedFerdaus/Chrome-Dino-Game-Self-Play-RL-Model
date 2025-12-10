import pygame
from game import Game
from agent import Agent

def train():
    agent = Agent()
    game = Game()
    
    obstacles_passed = 0
    last_obstacle_count = 0
    
    while game.running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                agent.save()
                game.running = False
                pygame.quit()
                return
        
        # Get current state (screenshot)
        state_old = agent.get_state(game.screen)
        
        # Get action
        action = agent.get_action(state_old)
        
        # Perform action
        # 0 = do nothing, 1 = jump, 2 = duck
        if action == 1 and not game.dino.jumping:
            game.dino.jump()
        elif action == 2:
            game.dino.duck(True)
        else:
            game.dino.duck(False)
        
        # Update game
        game.update()
        
        # Count obstacles passed (simple and effective reward)
        current_obstacle_count = 0
        for obstacle in game.obstacles:
            if obstacle.x + obstacle.width < game.dino.x:
                current_obstacle_count += 1
        
        # Reward for passing an obstacle
        reward = 0
        if current_obstacle_count > last_obstacle_count:
            reward = 1
            obstacles_passed += 1
        
        last_obstacle_count = current_obstacle_count
        
        # Get new state
        state_new = agent.get_state(game.screen)
        
        done = False
        if game.game_over:
            reward = -1
            done = True
        
        # Remember
        agent.remember(state_old, action, reward, state_new, done)
        
        # Train
        if len(agent.memory) > agent.batch_size:
            agent.train_long_memory()
        
        if game.game_over:
            agent.n_games += 1
            
            # Save model periodically
            if agent.n_games % 50 == 0:
                agent.save(f'model_game_{agent.n_games}.pth')
            
            # Reset game and frame stack
            game.reset_game()
            agent.reset_frame_stack()
            obstacles_passed = 0
            last_obstacle_count = 0
        
        # Draw
        game.draw()
        game.clock.tick(60)

if __name__ == "__main__":
    train()