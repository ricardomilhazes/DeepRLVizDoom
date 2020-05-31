import random
import math
import matplotlib.pyplot as plt
import deepQNetwork as dqn

possible_learning_rate = [0.1, 0.01, 0.001, 0.0001]
bounds = [(0,1),(0,3)] # 1st variable -> discount_factor, 2nd variable -> activation_function
possible_number_of_fm = [4,8,12,16]
nv = 4

initial_fitness = -1000 # maximization problem

particle_size = 5 # how many particles do we want?
iteration = 5 # how many iterations do we want?
w = 0.729 # used to determine inertia
c1 = 2.05 # weight given to individual solutions
c2 = 2.05 # weight given to colective solutions

class Particle:
    def __init__(self,bounds):
        self.particle_position = []
        self.particle_velocity = []
        self.local_best_particle_position = []
        self.fitness_local_best_particle_position = initial_fitness
        self.fitness_particle_position = initial_fitness

        self.particle_position.append(random.choice(possible_learning_rate)) # learning_rate
        self.particle_velocity.append(random.choice([0.1,10]))

        self.particle_position.append(random.uniform(bounds[0][0],bounds[0][1])) # discount_factor
        self.particle_velocity.append(random.uniform(-1,1))

        self.particle_position.append(random.randint(bounds[1][0],bounds[1][1])) # activation_func
        self.particle_velocity.append(random.randint(-1,1))
        
        self.particle_position.append(random.choice(possible_number_of_fm)) # feature_maps
        self.particle_velocity.append(random.choice([-4,4]))

    def evaluate(self):
        print(self.particle_position[2])
        agent = dqn.Agent(self.particle_position[0], self.particle_position[1], self.particle_position[3], self.particle_position[2])
        self.fitness_particle_position = agent.play(verbose=1)

        if self.fitness_particle_position > self.fitness_local_best_particle_position:
            self.local_best_particle_position = self.particle_position
            self.fitness_local_best_particle_position = self.fitness_particle_position

    def update_velocity(self,global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = c1*r1*(self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2*r2*(global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w*self.particle_velocity[i] + cognitive_velocity + social_velocity

    def update_position(self,bounds):
        self.particle_position[0] = self.particle_position[0] * self.particle_velocity[0]

        i = 1
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

        if(self.particle_position[0] > 0.1):
            self.particle_position[0] = 0.1
        if(self.particle_position[0] >= 0.055 and self.particle_position[0] <= 0.1):
            self.particle_position[0] = 0.1
        if(self.particle_position[0] >= 0.0055 and self.particle_position[0] < 0.55):
            self.particle_position[0] = 0.01
        if(self.particle_position[0] >= 0.00055 and self.particle_position[0] < 0.0055):
            self.particle_position[0] = 0.001
        if(self.particle_position[0] >= 0.0001 and self.particle_position[0] < 0.00055):
            self.particle_position[0] = 0.0001
        if(self.particle_position[0] < 0.0001):
            self.particle_position[0] = 0.0001

        if(self.particle_position[1] > bounds[0][1]):
            self.particle_position[1] = bounds[0][1]
        if(self.particle_position[1] < bounds[0][0]):
            self.particle_position[1] = bounds[0][0]

        if(self.particle_position[2] > bounds[1][1]):
            self.particle_position[2] = bounds[1][1]
        if(self.particle_position[2] >= 2.5 and self.particle_position[2] <= 3.0):
            self.particle_position[2] = 3
        if(self.particle_position[2] >= 1.5 and self.particle_position[2] < 2.5):
            self.particle_position[2] = 2
        if(self.particle_position[2] >= 0.5 and self.particle_position[2] < 1.5):
            self.particle_position[2] = 1
        if(self.particle_position[2] >= 0 and self.particle_position[2] < 0.5):
            self.particle_position[2] = 0
        if(self.particle_position[2] < bounds[1][0]):
            self.particle_position[2] = bounds[1][0]
        
        if(self.particle_position[3] > 16):
            self.particle_position[3] = 16
        if(self.particle_position[3] >= 14 and self.particle_position[3] <= 16):
            self.particle_position[3] = 16
        if(self.particle_position[3] >= 10 and self.particle_position[3] < 14):
            self.particle_position[3] = 12
        if(self.particle_position[3] >= 6 and self.particle_position[3] < 10):
            self.particle_position[3] = 8
        if(self.particle_position[3] >= 4 and self.particle_position[3] < 6):
            self.particle_position[3] = 4
        if(self.particle_position[3] < 4):
            self.particle_position[3] = 4
    
class PSO():
    def __init__(self,bounds,particle_size,iteration):

        fitness_global_best_particle_position = initial_fitness
        global_best_particle_position = []

        swarm_particle = []
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        A = []

        for i in range(iteration):
            for j in range(particle_size):
                swarm_particle[j].evaluate()

                if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                    global_best_particle_position = list(swarm_particle[j].particle_position)
                    fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)

            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)
            
            A.append(fitness_global_best_particle_position)
    
        print('Optimal Solution: ', global_best_particle_position)
        print('Objective Function Value: ', fitness_global_best_particle_position)
        plt.plot(A)
        plt.show()

PSO(bounds,particle_size,iteration)