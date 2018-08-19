# Car position minimal and maximal values
pmin = -1.2
pmax = 0.6

# Car velocity minimal and maximal values
vmin = -0.07
vmax = 0.07

# Maximum width of area 
wmax = 0.07

# Discretization of states space
dS = 40

# Number of success to achieve using random policies before learn neural network 
randomNR = 50

# Repeating the learning NN process
learnN = 6

# Maximal value of noise adding in each learnig step
maxRandomNoise = vmax/dS

# Learnig rate for optimalization algoritm
learningRate = 0.002
