#!/usr/bin/python3.6
import mlrose
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def my_random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0,
                      init_state=None):
    """Use randomized hill climbing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    Brownlee, J (2011). *Clever Algorithms: Nature-Inspired Programming
    Recipes*. `<http://www.cleveralgorithms.com>`_.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    best_fitness = -1*np.inf
    best_state = None

    ''' My modification '''
    all_states = []
    all_fitness = []
    ''' end '''

    for _ in range(restarts + 1):
        # Initialize optimization problem and attempts counter
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        attempts = 0
        iters = 0

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1
            

            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If best neighbor is an improvement,
            # move to that state and reset attempts counter
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0
                all_states.append(next_state) #
                all_fitness.append(problem.get_maximize()*next_fitness) # 
            else:
                attempts += 1
                all_states.append(problem.get_state()) #
                all_fitness.append(problem.get_maximize()*problem.get_fitness()) #

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness
    #return best_state, best_fitness
    return all_states, all_fitness #


def my_simulated_annealing(problem, schedule=mlrose.decay.GeomDecay(), max_attempts=10,
                        max_iters=np.inf, init_state=None):
    """Use simulated annealing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    schedule: schedule object, default: :code:`mlrose.GeomDecay()`
        Schedule used to determine the value of the temperature parameter.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    attempts = 0
    iters = 0

    ''' My modification '''
    all_states = []
    all_fitness = []
    ''' end '''

    while (attempts < max_attempts) and (iters < max_iters):
        temp = schedule.evaluate(iters)
        iters += 1

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e/temp)

            # If best neighbor is an improvement or random value is less
            # than prob, move to that state and reset attempts counter
            if (delta_e > 0) or (np.random.uniform() < prob):
                problem.set_state(next_state)
                attempts = 0
                all_states.append(next_state) #
                all_fitness.append(problem.get_maximize()*next_fitness) # 
            else:
                attempts += 1
                all_states.append(problem.get_state()) #
                all_fitness.append(problem.get_maximize()*problem.get_fitness()) #

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    #return best_state, best_fitness
    return all_states, all_fitness # 

def my_genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                max_iters=np.inf):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    ''' My modification '''
    all_states = []
    all_fitness = []
    ''' end '''

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        for _ in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0
            all_states.append(next_state) #
            all_fitness.append(problem.get_maximize()*next_fitness) # 
        else:
            attempts += 1
            all_states.append(problem.get_state()) #
            all_fitness.append(problem.get_maximize()*problem.get_fitness()) #

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    #return best_state, best_fitness
    return all_states, all_fitness #


class MyNeuralNetwork(mlrose.NeuralNetwork):
	def __init__(self, hidden_nodes, activation='relu',algorithm='random_hill_climb', max_iters=100, bias=True,is_classifier=True, learning_rate=0.1, early_stopping=False,clip_max=1e+10, schedule=mlrose.decay.GeomDecay(1E12,0.99,1), pop_size=200,mutation_prob=0.1, max_attempts=10):
		super().__init__(hidden_nodes, activation, algorithm, max_iters, bias, is_classifier, learning_rate, early_stopping, clip_max, schedule, pop_size, mutation_prob, max_attempts)

	# def predict(self, X):
	# 	super().predict(X)

	def fit(self, X, y, validationX, validationY, init_weights=None):
		input_y = y
		y = np.array(y)

		# Convert y to 2D if necessary
		if len(np.shape(y)) == 1:
			y = np.reshape(y, [len(y), 1])

		# Verify X and y are the same length
		if not np.shape(X)[0] == np.shape(y)[0]:
			raise Exception('The length of X and y must be equal.')

        # Determine number of nodes in each layer
		input_nodes = np.shape(X)[1] + self.bias
		output_nodes = np.shape(y)[1]
		node_list = [input_nodes] + self.hidden_nodes + [output_nodes]

		num_nodes = 0

		for i in range(len(node_list) - 1):
			num_nodes += node_list[i]*node_list[i+1]

		if init_weights is not None and len(init_weights) != num_nodes:
			raise Exception("""init_weights must be None or have length %d""" % (num_nodes,))

        # Initialize optimization problem
		fitness = mlrose.neural.NetworkWeights(X, y, node_list, self.activation, self.bias,
                                 self.is_classifier, learning_rate=self.lr)

		problem = mlrose.opt_probs.ContinuousOpt(num_nodes, fitness, maximize=False,
                                min_val=-1*self.clip_max,
                                max_val=self.clip_max, step=self.lr)

		if self.algorithm == 'random_hill_climb':
			if init_weights is None:
				init_weights = np.random.uniform(-1, 1, num_nodes)

			fitted_weights, loss = my_random_hill_climb(
                problem,
                max_attempts=self.max_attempts, max_iters=self.max_iters,
                restarts=0, init_state=init_weights)

		elif self.algorithm == 'simulated_annealing':
			if init_weights is None:
				init_weights = np.random.uniform(-1, 1, num_nodes)
			fitted_weights, loss = my_simulated_annealing(
                problem,
                schedule=self.schedule, max_attempts=self.max_attempts,
                max_iters=self.max_iters, init_state=init_weights)

		elif self.algorithm == 'genetic_alg':
			fitted_weights, loss = my_genetic_alg(
                problem,
                pop_size=self.pop_size, mutation_prob=self.mutation_prob,
                max_attempts=self.max_attempts, max_iters=self.max_iters)

		else:  # Gradient descent case
			if init_weights is None:
				init_weights = np.random.uniform(-1, 1, num_nodes)
			fitted_weights, loss = gradient_descent(
                problem,
                max_attempts=self.max_attempts, max_iters=self.max_iters,
                init_state=init_weights)

        # Save fitted weights and node list
		for i in range(self.max_iters):
			self.node_list = node_list
			self.fitted_weights = fitted_weights[i]
			self.loss = loss[i]
			self.output_activation = fitness.get_output_activation()
			y_train_pred = self.predict(X)
			y_train_accuracy = accuracy_score(input_y, y_train_pred)
			y_validation_pred = self.predict(validationX)
			y_validation_accuracy = accuracy_score(validationY, y_validation_pred)
			print(i, ",", y_train_accuracy, ",", y_validation_accuracy)


def loadData(filename):
	# load adult data and drop missing information
	data = pd.read_csv(filename, names=['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country','result'],
		engine="python", skipinitialspace=True, na_values=['?'])
	data = data.dropna()

	# encode the labels to numbers
	eData = data.copy()
	for column in eData.columns:
		if eData.dtypes[column] == 'object':
			le = preprocessing.LabelEncoder()
			le.fit(eData[column])
			eData[column] = le.transform(eData[column])
	return eData
	
def prepareData(cross):
	trainset = loadData("./data/adult.data")
	testset = loadData("./data/adult.test")

	# seperate data for cross validation
	trainX, validationX, trainY, validationY = train_test_split(trainset[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']], trainset['result'], test_size=cross)
	testX = testset[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
	testY = testset["result"]

	scaler = preprocessing.MinMaxScaler()

	trainX_scaled = scaler.fit_transform(trainX)
	validationX_scaled = scaler.transform(validationX)
	testX_scaled = scaler.transform(testX)

	return trainX_scaled, validationX_scaled, trainY, validationY, testX_scaled, testY

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: ./randomized.py <random_hill_climb OR simulated_annealing OR genetic_alg>'''")
		exit(0)
	trainX, validationX, trainY, validationY, testX, testY = prepareData(0.2)

	algorithm = sys.argv[1]
	np.random.seed(3)
	models = []
	accs = []


	nn_model1 = MyNeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = algorithm, max_iters = 100,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts = 100)



	nn_model1.fit(trainX, trainY, validationX, validationY)
	
	# y_train_pred = nn_model1.predict(trainX)
	# y_train_accuracy = accuracy_score(trainY, y_train_pred)
	# y_validation_pred = nn_model1.predict(validationX)
	# y_validation_accuracy = accuracy_score(validationY, y_validation_pred)

	# models.append(nn_model1)
	# accs.append(y_validation_accuracy)

	# #print(i, ',', y_train_accuracy, ',', y_validation_accuracy)

	# maxScore = max(accs)
	# index = accs.index(maxScore)
	# model = models[index]
	y_test_pred = nn_model1.predict(testX)
	y_test_accuracy = accuracy_score(testY, y_test_pred)

	print(y_test_accuracy)
