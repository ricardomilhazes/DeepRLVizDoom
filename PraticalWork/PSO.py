import random
import math
import matplotlib.pyplot as plt
import deepQNetwork as dqn

possible_lr = [0.1, 0.01, 0.001, 0.0001]
possible_fm = [4,8,12,16]
possible_atv_func = [0, 1, 2, 3]
# df pode ser qql valore entre [0, 1]

nv = 4

initial_fitness = -1000 # maximization problem

particle_size = 3 # how many particles do we want?
iteration = 2 # how many iterations do we want?
w = 0.729 # used to determine inertia
c1 = 2.05 # weight given to individual solutions
c2 = 2.05 # weight given to colective solutions


def convertPositionToValue(positionValue, possibleValues):
    if(positionValue < 0.25):
        return possibleValues[0]
    elif positionValue < 0.5:
        return possibleValues[1]
    elif positionValue < 0.75:
        return possibleValues[2]
    else:
       return possibleValues[3]

def convertBestSolution(best_particle_position):
    lr = convertPositionToValue(best_particle_position[0], possible_lr)
    df = best_particle_position[1]
    fm = convertPositionToValue(best_particle_position[2], possible_fm)
    af = convertPositionToValue(best_particle_position[3], possible_atv_func)

    best_solution = []
    best_solution.append(lr)
    best_solution.append(df)
    best_solution.append(fm)
    best_solution.append(af)

    return best_solution


class Particle:
    def __init__(self, id):
        self.id = id
        self.iteration = 0
        self.particle_position = []
        self.particle_velocity = []
        self.local_best_particle_position = []
        self.fitness_local_best_particle_position = initial_fitness
        self.fitness_particle_position = initial_fitness

        for i in range(nv):
            # generate random initial position
            self.particle_position.append(random.uniform(0, 1))
            # generate random initial velocity
            self.particle_velocity.append(random.uniform(-1, 1))

        #self.particle_position.append(random.choice(possible_learning_rate)) # learning_rate
        #self.particle_velocity.append(random.choice([0.1,10]))

        #self.particle_position.append(random.uniform(bounds[0][0],bounds[0][1])) # discount_factor
        #self.particle_velocity.append(random.uniform(-1,1))

        #self.particle_position.append(random.randint(bounds[1][0],bounds[1][1])) # activation_func
        #self.particle_velocity.append(random.randint(-1,1))
        
        #self.particle_position.append(random.choice(possible_number_of_fm)) # feature_maps
        #self.particle_velocity.append(random.choice([-4,4]))

    def evaluate(self):
        lr = convertPositionToValue(self.particle_position[0], possible_lr)
        df = self.particle_position[1]
        fm = convertPositionToValue(self.particle_position[2], possible_fm)
        af = convertPositionToValue(self.particle_position[3], possible_atv_func)

        self.iteration += 1

        print("############################################")
        print("Particle " + str(self.id) + ", iteration " + str(self.iteration) + " (lr: " + str(lr) + ", df: " + str(df) + ", fm: " + str(fm) + ", af: " + str(af)+ ")")
        agent = dqn.Agent(lr, df, fm, af)
        score, _, _, _, _ = agent.play(verbose=0)
        self.fitness_particle_position = score
        print("Particle " + str(self.id) + ", iteration " + str(self.iteration) + ", score " + str(self.fitness_particle_position))
        print("############################################")

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

    def update_position(self):
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > 1:
                self.particle_position[i] = 1
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < 0:
                self.particle_position[i] = 0
    
class PSO():
    def __init__(self,particle_size, iteration):

        fitness_global_best_particle_position = initial_fitness
        global_best_particle_position = []

        swarm_particle = []
        for i in range(particle_size):
            swarm_particle.append(Particle(i+1))
        A = []

        for i in range(iteration):
            for j in range(particle_size):
                swarm_particle[j].evaluate()

                if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                    global_best_particle_position = list(swarm_particle[j].particle_position)
                    fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)

            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position()
            
            A.append(fitness_global_best_particle_position)
    
        print('Optimal Solution: ', convertBestSolution(global_best_particle_position))
        print('Objective Function Value: ', fitness_global_best_particle_position)
        plt.plot(A)
        plt.show()

PSO(particle_size,iteration)