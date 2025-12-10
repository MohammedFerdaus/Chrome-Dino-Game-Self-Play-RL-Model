import pygame
import random
import sys
import time
from constants import *

pygame.init()

class Dino:
    def __init__(self, x, y, image_path):
        self.x = x
        self.y = y
        self.width = 88 // 2  # 44
        self.height = 85 // 2  # 42
        self.vel_y = 0
        self.jumping = False
        self.ducking = False
        self.ground_y  = y

        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (self.width, self.height))

    def jump(self):
        if not self.jumping:
            self.vel_y = jump_strength
            self.jumping = True

    def duck(self, is_ducking):
        self.ducking = is_ducking

    def update(self):
        if self.jumping:
            self.vel_y += gravity
            self.y += self.vel_y

            if self.y >= self.ground_y:
                self.y = self.ground_y
                self.vel_y = 0
                self.jumping = False
    
    def draw(self, screen):
        if self.image:
            if self.ducking:
                ducked_height = self.height // 2
                ducked_image = pygame.transform.scale(self.image, (self.width, ducked_height))
                screen.blit(ducked_image, (self.x, self.y + self.height // 2))
            else:
                screen.blit(self.image, (self.x, self.y))

        else:
            height = self.height // 2 if self.ducking else self.height
            pygame.draw.rect(screen, gray, (self.x, self.y if not self.ducking else self.y + self.height // 2, self.width, height))
    
    def get_rect(self):
        height = self.height // 2 if self.ducking else self.height
        return pygame.Rect(self.x, self.y if not self.ducking else self.y + self.height // 2, self.width, height)

class Cactus:
    def __init__(self, x, y, speed, image_path):
        self.x = x
        self.y = y 
        self.width = 50 // 2  # 25
        self.height = 105 // 2  # 52
        self.speed = speed
        
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (self.width, self.height))

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def is_off_screen(self):
        return self.x + self.width < 0

class Pterodactyl:
    def __init__(self, x, y, speed, image_path):
        self.x = x
        self.y = y
        self.width = 194 // 3  # 64
        self.height = 127 // 3  # 42
        self.speed = speed
        self.frame_count = 0

        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (self.width, self.height))

    def update(self):
        self.x -= self.speed
        self.frame_count += 1

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def get_rect(self):
        return pygame.Rect(self.x + 40, self.y + 30, self.width - 80, self.height - 60)
    
    def is_off_screen(self):
        return self.x + self.width < 0
    
class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Chrome Dino Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_over = False

        self.dino_image = "assets/Chrome_Trex.png"
        self.cactus_image = "assets/Chrome_Cactus.png"
        self.pterodactyl_image = "assets/Chrome_Pterodactyl.png"

        self.ground_y = screen_height - 120

        self.dino = Dino(50, self.ground_y - 42 + 15, self.dino_image)  # Added +15 to lower dino to ground

        self.obstacles = []
        self.spawn_timer = 0
        self.spawn_delay = 90
        
        # Score
        self.score = 0
        self.frame_count = 0  # Add frame counter for score
        self.font = pygame.font.Font(None, 30)
        self.game_speed = game_speed
        
        # New stats
        self.high_score = 0
        self.games_played = 0
        self.start_time = time.time()

    def spawn_obstacle(self):
        obstacle_type = random.choice(['cactus', 'cactus', 'pterodactyl'])  # More cacti
        
        if obstacle_type == 'cactus':
            obstacle = Cactus(screen_width, self.ground_y - 52 + 15, self.game_speed, self.cactus_image)  # Added +15 to lower cactus
        else:
            # Pterodactyl at different heights
            heights = [self.ground_y - 42, self.ground_y - 70, self.ground_y - 30]  # Adjusted for 1/3 size pterodactyl
            height = random.choice(heights)
            obstacle = Pterodactyl(screen_width, height, self.game_speed, self.pterodactyl_image)
        
        self.obstacles.append(obstacle)

    def check_collision(self):
        dino_rect = self.dino.get_rect()
        for obstacle in self.obstacles:
            obstacle_rect = obstacle.get_rect()
            if dino_rect.colliderect(obstacle_rect):
                return True
        return False
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.game_over:
                    self.dino.jump()
                elif event.key == pygame.K_SPACE and self.game_over:
                    self.reset_game()
                elif event.key == pygame.K_DOWN:
                    self.dino.duck(True)
            
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    self.dino.duck(False)

    def update(self):
        if self.game_over:
            return
        
        # Update dino
        self.dino.update()
        
        # Increase score over time (every 6 frames = ~10 times per second at 60fps)
        self.frame_count += 1
        if self.frame_count >= 6:
            self.score += 1
            self.frame_count = 0
        
        # Update obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update()
            if obstacle.is_off_screen():
                self.obstacles.remove(obstacle)
        
        # Spawn obstacles
        self.spawn_timer += 1
        if self.spawn_timer > self.spawn_delay:
            self.spawn_obstacle()
            self.spawn_timer = 0
            self.spawn_delay = random.randint(60, 120)
        
        # Check collision
        if self.check_collision():
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
        
        # Increase difficulty
        self.game_speed = game_speed + (self.score // 100) * 0.5
    
    def draw(self):
        # Clear screen
        self.screen.fill(white)
        
        # Draw ground
        pygame.draw.line(self.screen, black, (0, self.ground_y + 20), (screen_width, self.ground_y + 20), 2)
        
        # Draw dino
        self.dino.draw(self.screen)
        
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        # Calculate elapsed time
        elapsed_time = int(time.time() - self.start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        
        # Draw stats
        score_text = self.font.render(f"Score: {self.score}", True, gray)
        high_score_text = self.font.render(f"High Score: {self.high_score}", True, gray)
        games_text = self.font.render(f"Games: {self.games_played}", True, gray)
        time_text = self.font.render(f"Time: {minutes:02d}:{seconds:02d}", True, gray)
        
        self.screen.blit(score_text, (15, 20))
        self.screen.blit(high_score_text, (15, 50))
        self.screen.blit(games_text, (15, 80))
        self.screen.blit(time_text, (15, 110))
        
        if self.game_over:
            game_over_text = self.font.render("GAME OVER", True, black)
            restart_text = self.font.render("Press SPACE to restart", True, black)
            self.screen.blit(game_over_text, (screen_width // 2 - 100, screen_height // 2 - 40))
            self.screen.blit(restart_text, (screen_width // 2 - 160, screen_height // 2 + 10))
        
        pygame.display.flip()
    
    def reset_game(self):
        self.dino = Dino(50, self.ground_y - 42 + 15, self.dino_image)  # Added +15 to match initial position
        self.obstacles = []
        self.spawn_timer = 0
        self.score = 0
        self.frame_count = 0  # Reset frame counter
        self.game_over = False
        self.game_speed = game_speed
        self.games_played += 1
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(fps)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()