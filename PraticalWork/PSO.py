import random
import math
import matplotlib.pyplot as plt
import deepQNetwork as dqn

bounds=[(0,1),(0,1)] # 1st variable -> learning rate, 2nd variable -> discount_factor
nv = 2

initial_fitness = -1000 # maximization problem

particle_size = 10 # how many particles do we want?
iteration = 10 # how many iterations do we want?
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

        for i in range(nv): 
            self.particle_position.append(random.uniform(bounds[i][0],bounds[i][1]))
            self.particle_velocity.append(random.uniform(-1,1))

    def evaluate(self):
        # self.fitness_particle_position = average_score
        agent = dqn.Agent(self.particle_position[0], self.particle_position[1])
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
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

            if(self.particle_position[i] > bounds[i][1]):
                self.particle_position[i] = bounds[i][1]
            if(self.particle_position[i] < bounds[i][0]):
                self.particle_position[i] = bounds[i][0]
    
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