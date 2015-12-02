import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.stats import norm
import math

def generate_data(input_range):
    y1 = map(lambda x: 3*x + 2, input_range)
    y2 = map(lambda x: 2*x - 3, input_range)
    jitter = 2*np.random.randn(50)
    return (np.array(y1) + jitter, np.array(y2) + jitter)

def initialize_components(all_data):
    component_one = []
    component_two = []
    for point in all_data:
        if random.random() <= 0.5:
            component_one.append(point)
        else:
            component_two.append(point)
    return component_one, component_two

def maximization(components):
    '''
    This step simply calculates the regression parameters using least-squares, which is equivalent to MLE,
    from the components that are passed in.
    If number of components isn't known, we can pass in a list of components and do something like this
    for x,y in zip(*[iter(component_list)]*2):
        print x,y
    to iterate 2 at a time. Here this example is only of 2 components.
    '''
    comp1_x, comp1_y = zip(*components[0])
    comp1_slope, comp1_intercept, _, _, _ = stats.linregress(np.array(comp1_x), np.array(comp1_y))
    comp2_x, comp2_y = zip(*components[1])
    comp2_slope, comp2_intercept, _, _, _ = stats.linregress(np.array(comp2_x), np.array(comp2_y))
    return [(comp1_slope, comp1_intercept), (comp2_slope, comp2_intercept)]
    
def get_sigma(component, params):
    sigma = 0.0
    n = len(component)
    for xi, yi in component:
        mu_i = params[1] + xi*params[0]
        sigma += (yi - mu_i)**2
    sigma = (1.0/n)*sigma
    return sigma
        

def get_ll(xi, yi, params, sigma):
    mu_i = params[1] + xi*params[0]
    likelihood = norm.pdf(yi, loc=mu_i, scale=sigma)
    return likelihood

def get_likelihood(components, parameters):
    '''
    The log likelihood is used as a convergence criterion for the EM algorithm. The reason it is used is
    because generally the maximum likelihood estimation method is used to to determine the parameters of the component
    we're interested in. For this reason, it's convenient to use as a convergence criterion as we're seeking to maximize
    it anyway.
    If Y = b0 + b1X + err is our model, with err being a random noise variable, and err ~ N(0, s^2), and
    is independent of X, and is independent across observations the likelihood is the product over i of
    scipy.stats.norm.pdf(y_i, loc=mu_i, scale=sigma) where mu_i is b0 + b1x_i, and sigma (1/n) sum over i (y_i - mu_i)**2
    '''
    likelihoods = []
    for comp, params in zip(components, parameters):
        ll = 1
        sigma = get_sigma(comp, params)
        for xi, yi in comp:
            ll *= get_ll(xi, yi, params, sigma)
        likelihoods.append(math.log(ll))
    return sum(likelihoods)

def get_distance(xi, yi, params):
    func = lambda x: params[0]*x + params[1]
    return abs(yi - func(xi)) 

def expectation(components, parameters):
    '''
    This step assigns the data points to a component based on the calculated parameters. There are two ways to do this:
    the specific way here is to calculate the difference Y-Y' where Y' is the estimated value based on the parameters we
    calculated from the maximization step for both components and assign that point to the component it's closest to. The
    more general approach is to calculate the likelihood that it belongs either component, and assign it to the one that yields
    the greater likelihood.
    We can also employ a stochastic method to assign the components using the likelihood as a starting point to generate
    a probability distribution from which a pseudo monte carlo method may be used to assign the points. This can be helpful
    in instances where the algorithm converges to a local minimum.
    '''
    new_components = [[], []]
    for comp in components:
        for xi, yi in comp:
            likelihoods = []
            for params in parameters:
                sigma = get_sigma(comp, params)
                #likelihoods.append(get_ll(xi, yi, params, sigma))
                likelihoods.append(get_distance(xi, yi, params))
            print likelihoods
            #max_index = likelihoods.index(max(likelihoods))
            max_index = likelihoods.index(min(likelihoods))
            new_components[max_index].append((xi, yi))
    return new_components

def main():
    input_range = range(50)
    y1, y2 = generate_data(input_range)
    all_data = [x for x in zip(input_range, y1)] + [x for x in zip(input_range, y2)]
    random.shuffle(all_data) # randomize the order so we don't know which point belongs to which component
    
    # initialize components by randomly assigning points to them
    component_one, component_two = initialize_components(all_data)
    components = [component_one, component_two]
    parameters = maximization(components)
    prev_ll = 0
    new_ll = get_likelihood(components, parameters)
    counter = 0
    while abs(new_ll - prev_ll) >= 10**-3:
        prev_ll = new_ll
        components = expectation(components, parameters)
        parameters = maximization(components)
        new_ll = get_likelihood(components, parameters)
        print new_ll, prev_ll
        counter += 1
    print parameters
    print counter
    x1, y1 = zip(*components[0])
    x2, y2 = zip(*components[1])
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)
    ypred1 = parameters[0][1] + parameters[0][0]*x1_arr
    ypred2 = parameters[1][1] + parameters[1][0]*x2_arr
    plt.plot(x1_arr, ypred1)
    plt.plot(x2_arr, ypred2)
    plt.show()
        
    
    

if __name__ == '__main__':
    main()