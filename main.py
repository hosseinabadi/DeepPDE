import numpy as np
from Infrastructure.utils import *
from nn_models import *
from models import *
from exact_solutions import *
import matplotlib.pyplot as plt

np.random.seed(42)
# User settings

load_model = True
nr_samples_surface_plot = 21
nr_samples_scatter_plot = 1000
nr_samples_error_calculation = 10000


# Model parameters. Re-train model after any changes.
s_min_interest = 25
s_max_interest = 150
t_min_interest = 0.5
t_max_interest = 4.

riskfree_rate_min = 0.1
riskfree_rate_max = 0.3
riskfree_rate_eval = 0.2

volatility_min = 0.1
volatility_max = 0.3
volatility1_eval = 0.1
volatility2_eval = 0.3

correlation_min = 0.2
correlation_max = 0.8
correlation_eval = 0.5

strike_price = 100.

# neural network parameters
nr_nodes_per_layer = 90
initial_learning_rate = 0.001
localisation_parameter = 1/10.

n_train = 10000
nr_epochs = 601

# Internal Parameters
dimension_state = 2
dimension_parameter = 4
dimension_total = 1 + dimension_state + dimension_parameter

t_min = 0.
t_max = t_max_interest
s_max = strike_price * (1 + 3*volatility_max*t_max)
x_max = np.log(s_max)
x_min = 2*np.log(strike_price) - x_max

normalised_max = 1
normalised_min = -1

t_min_interest_normalised = normalise_time(t_min_interest)
t_max_interest_normalised = normalise_time(t_max_interest)

diff_dx = (normalised_max-normalised_min) / (x_max-x_min)
diff_dt = (normalised_max-normalised_min) / (t_max-t_min)

riskfree_rate_eval_normalised = normalise_riskfree_rate(riskfree_rate_eval)
volatility1_eval_normalised = normalise_volatility(volatility1_eval)
volatility2_eval_normalised = normalise_volatility(volatility2_eval)
correlation_eval_normalised = normalise_correlation(correlation_eval)


def create_network(inputs):
    """ Creates the neural network by creating three highway layers and an
    output layer. Returns the output of these layers as a tensorflow variable.

    Keyword arguments:
    inputs -- Tensorflow variable of the input layer
    """
    layer0 = keras.layers.Dense(nr_nodes_per_layer, activation="tanh")

    layer1 = HighwayLayer(units=nr_nodes_per_layer,
                          original_input=dimension_total)
    layer2 = HighwayLayer(units=nr_nodes_per_layer,
                          original_input=dimension_total)
    layer3 = HighwayLayer(units=nr_nodes_per_layer,
                          original_input=dimension_total)

    last_layer = keras.layers.Dense(1)

    outputs_layer0 = layer0(inputs)
    outputs_layer1 = layer1({'previous_layer': outputs_layer0,
                             'original_variable': inputs})
    outputs_layer2 = layer2({'previous_layer': outputs_layer1,
                             'original_variable': inputs})
    outputs_layer3 = layer3({'previous_layer': outputs_layer2,
                             'original_variable': inputs})

    outputs_dnn = last_layer(outputs_layer3)

    inputs_t_normalised = inputs[:, 0:1]
    inputs_x1_normalised = inputs[:, 1:2]
    inputs_x2_normalised = inputs[:, 2:3]
    inputs_p1_normalised = inputs[:, 3:4]

    inputs_t = transform_to_time(inputs_t_normalised)
    inputs_x1 = transform_to_logprice(inputs_x1_normalised)
    inputs_x2 = transform_to_logprice(inputs_x2_normalised)
    inputs_s_mean = (tf.math.exp(inputs_x1) + tf.math.exp(inputs_x2))/2.
    riskfree_rate = transform_to_riskfree_rate(inputs_p1_normalised)

    localisation = tf.math.log(1+tf.math.exp(localisation_parameter * (
            inputs_s_mean - strike_price * tf.exp( - riskfree_rate * inputs_t)
              )))/localisation_parameter

    return outputs_dnn + localisation

%%time
if load_model:
    # Load model from local folder. If it is not availabe, download it.
    os.makedirs('model/variables', exist_ok=True)
    url_base = 'https://github.com/LWunderlich/DeepPDE/raw/main/TwoAssetsExample/'
    filename = 'model/saved_model.pb'
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url_base + filename, filename)

    filename = 'model/variables/variables.data-00000-of-00001'
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url_base + filename, filename)

    filename = 'model/variables/variables.index'
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url_base + filename, filename)

    model = keras.models.load_model('model')
else:
    # Create and train model from scratch.
    inputs = keras.Input(shape=(dimension_total,))
    outputs = create_network(inputs)
    model = DPDEModel(inputs=inputs, outputs=outputs)
    batch_generator = DPDEGenerator(n_train)
    model.compile(optimizer=tf.keras.optimizers.Adam(initial_learning_rate))
    callback = tf.keras.callbacks.EarlyStopping(
        'loss', patience=50, restore_best_weights=True)

    model.fit(x=batch_generator, epochs=nr_epochs, steps_per_epoch=10,
                          callbacks=[callback])

test_solution = exact_solution(t=4., s1=100., s2=100., riskfree_rate=0.2,
                               volatility1=0.1, volatility2=0.3, correlation=0.5)
assert (np.abs(test_solution - 55.096796282039364) < 1e-10)

s1_plot_mesh, s2_plot_mesh, x_plot_normalised = \
    get_points_for_plot_at_fixed_time()

DPDE_solution = model.predict(x_plot_normalised).reshape(
    nr_samples_surface_plot, nr_samples_surface_plot)

exact_solution_evaluated = [exact_solution(t=t_max, s1=s1[0], s2=s2[0],
                                riskfree_rate=riskfree_rate_eval,
                                volatility1=volatility1_eval,
                                volatility2=volatility2_eval,
                                correlation=correlation_eval)
                  for s1, s2 in zip(
                      s1_plot_mesh.reshape(-1, 1), s2_plot_mesh.reshape(-1, 1))

                  ]
exact_solution_evaluated = np.array(exact_solution_evaluated)
exact_solution_evaluated = exact_solution_evaluated.reshape(
    nr_samples_surface_plot, nr_samples_surface_plot)

localisation_plot = localisation(4., s1_plot_mesh, s2_plot_mesh, riskfree_rate_eval)


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(s1_plot_mesh, s2_plot_mesh, DPDE_solution, cmap='viridis')
ax.set_title('DPDE Solution')
ax.set_xlabel('$s_1$')
ax.set_ylabel('$s_2$')
plt.show()

####

fig = plt.figure()
ax = plt.axes(projection='3d')

dnn_part = DPDE_solution - localisation_plot
ax.plot_surface(s1_plot_mesh, s2_plot_mesh, dnn_part, cmap='viridis')
ax.set_title('DNN part')
ax.set_xlabel('$s_1$')
ax.set_ylabel('$s_2$')
plt.show()

###

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(s1_plot_mesh, s2_plot_mesh, localisation_plot, cmap='viridis')
ax.set_title('Localisation')
ax.set_xlabel('$s_1$')
ax.set_ylabel('$s_2$')
plt.show()

###
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(s1_plot_mesh, s2_plot_mesh, exact_solution_evaluated, cmap='viridis')
ax.set_title('Exact solution')
ax.set_xlabel('$s_1$')
ax.set_ylabel('$s_2$')
plt.show()

###
