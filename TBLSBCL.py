# Addressing Constrained Engineering Problems and Feature Selection with a Time-Based Leadership Salp-Based Algorithm with Competitive Learning 
# More details about the algorithm are in [please cite the original paper ]
# Mohammed Qaraad , Souad Amjad, Nazar K. Hussein , and Mostafa A. Elhosseini, "Addressing Constrained Engineering Problems and Feature Selection with a Time-Based Leadership Salp-Based Algorithm with Competitive Learning"
# Journal of Computational Design and Engineering, 2022


import random
import numpy
import math
import time
import numpy as np
import matplotlib.pyplot as plt

def objective_Fun (x):
    return 20+x[0]**2-10.*np.cos(2*3.14159*x[0])+x[1]**2-10*np.cos(2*3.14159*x[1])

def TBLSBCL(objf, lb, ub, dim, N, Max_iteration):

    # Max_iteration=1000
    # lb=-100
    # ub=100
    # dim=30
    #N = 50  # Number of search agents
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    Convergence_curve = numpy.zeros(Max_iteration)

    # Initialize the positions of salps
    v     = numpy.zeros((N, dim))
    SalpPositions = numpy.zeros((N, dim))
    for i in range(dim):
        SalpPositions[:, i] = numpy.random.uniform(0, 1, N) * (ub[i] - lb[i]) + lb[i]
    SalpFitness = numpy.full(N, float("inf"))

    FoodPosition = numpy.zeros(dim)
    FoodFitness = float("inf")
    # Moth_fitness=numpy.fell(float("inf"))

    

    #print('TBLSBCL is optimizing  "' + objf.__name__ + '"')

    
   

    for i in range(0, N):
        # evaluate moths
        SalpFitness[i] = objf(SalpPositions[i, :])

    sorted_salps_fitness = numpy.sort(SalpFitness)
    I = numpy.argsort(SalpFitness)

    Sorted_salps = numpy.copy(SalpPositions[I, :])

    FoodPosition = numpy.copy(Sorted_salps[0, :])
    FoodFitness = sorted_salps_fitness[0]

    Iteration = 1

    # Main loop
    while Iteration < Max_iteration:

        # Number of flames Eq. (3.14) in the paper
        # Flame_no=round(N-Iteration*((N-1)/Max_iteration));
        center = SalpPositions.mean(axis=0)
        c1 = 2 * math.exp(-((4 * Iteration / Max_iteration) ** 2))
        L  = numpy.ceil(N * (Iteration / (Max_iteration + 1)))           
        
        # Eq. (3.2) in the paper

        for i in range(0, N):

            SalpPositions = numpy.transpose(SalpPositions)
            v     = numpy.transpose(v)
            if i < L:
                for j in range(0, dim):
                    c2 = random.random()
                    c3 = random.random()
                    # Eq. (3.1) in the paper
                    if c3 < 0.5:
                        SalpPositions[j, i] = FoodPosition[j] + c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )
                    else:
                        SalpPositions[j, i] = FoodPosition[j] - c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )

                    ####################

            else :
                point1 = SalpPositions[:, i - 1]
                point2 = SalpPositions[:, i]
                Positions= SalpPositions[:, i]
                WPositions =  (point2 + point1) / 2
                for d in range(0, dim):
                    r1 = random.random()
                    r2 = random.random()
                    r3 = random.random()                
                    v[d,i] = r1*v[d][i]+ r2*( WPositions[d] - Positions[d])+ r3*0.3*(center[d] - Positions[d])
                    SalpPositions[d,i] =  (SalpPositions[d,i] + v[d,i]) 


            SalpPositions = numpy.transpose(SalpPositions)
            v     = numpy.transpose(v)
        for i in range(0, N):

            # Check if salps go out of the search spaceand bring it back
            for j in range(dim):
                SalpPositions[i, j] = numpy.clip(SalpPositions[i, j], lb[j], ub[j])

            SalpFitness[i] = objf(SalpPositions[i, :])

            if SalpFitness[i] < FoodFitness:
                FoodPosition = numpy.copy(SalpPositions[i, :])
                FoodFitness = SalpFitness[i]
        #N = round(SearchAgents_no + Iteration * ((n_min - SearchAgents_no)/Max_iteration))
        #print("N", N)
        # Display best fitness along the iteration
#        if Iteration % 1 == 0:
#             print(
#                 [
#                     "At iteration "
#                     + str(Iteration)
#                     + " the best fitness is "
#                     + str(FoodFitness)
#                 ]
#             )

        Convergence_curve[Iteration] = FoodFitness

        Iteration = Iteration + 1


    

    return Convergence_curve


Max_iterations=50  # Maximum Number of Iterations
swarm_size = 30 # Number of salps
LB=-10  #lower bound of solution
UB=10   #upper bound of solution
Dim=2 #problem dimensions
NoRuns=100  # Number of runs
ConvergenceCurve=np.zeros((Max_iterations,NoRuns))
for r in range(NoRuns):
    result = TBLSBCL(objective_Fun, LB, UB, Dim, swarm_size, Max_iterations)
    ConvergenceCurve[:,r]=result
# Plot the convergence curves of all runs
idx=range(Max_iterations)
fig= plt.figure()

#3-plot
ax=fig.add_subplot(111)
for i in range(NoRuns):
    ax.plot(idx,ConvergenceCurve[:,i])
plt.title('Convergence Curve of the TBLSBCL Optimizer', fontsize=12)
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.show()
