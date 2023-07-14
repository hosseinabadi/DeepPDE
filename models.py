class DPDEGenerator(keras.utils.Sequence):
    """ Create batches of random points for the network training. """

    def __init__(self, batch_size):
        """ Initialise the generator by saving the batch size. """
        self.batch_size = batch_size

    def __len__(self):
        """ Describes the number of points to create """
        return self.batch_size

    def __getitem__(self, idx):
        """ Get one batch of random points in the interior of the domain to
        train the PDE residual and with initial time to train the initial value.
        """
        data_train_interior = np.random.uniform(
            normalised_min, normalised_max, [self.batch_size, dimension_total])

        t_train_initial = normalised_min * np.ones((self.batch_size, 1))
        s_and_p_train_initial = np.random.uniform(
            normalised_min, normalised_max,
            [self.batch_size, dimension_state + dimension_parameter])

        data_train_initial = np.concatenate(
            (t_train_initial, s_and_p_train_initial), axis=1)

        return [data_train_interior, data_train_initial]


class DPDEModel(keras.Model):
    """ Create a keras model with the deep param. PDE loss function """

    def train_step(self, data):
        """ Create one optimisation stop based on the deep param. PDE loss function. """
        data_interior, data_initial = data[0]

        riskfree_rate_interior = transform_to_riskfree_rate(
            data_interior[:, 3:4])
        volatility1_interior = transform_to_volatility(data_interior[:, 4:5])
        volatility2_interior = transform_to_volatility(data_interior[:, 5:6])
        correlation_interior = transform_to_correlation(data_interior[:, 6:7])

        x1_initial = transform_to_logprice(data_initial[:, 1:2])
        x2_initial = transform_to_logprice(data_initial[:, 2:3])

        with tf.GradientTape() as tape:
            v_interior = self(data_interior, training=True)  # Forward pass
            v_initial = self(data_initial, training=True)  # Forward pass bdry

            gradient = K.gradients(v_interior, data_interior)[0]

            v_dt = diff_dt * gradient[:, 0:1]
            v_dx1 = diff_dx * gradient[:, 1:2]
            v_dx2 = diff_dx * gradient[:, 2:3]

            grad_v_dx1 = K.gradients(v_dx1, data_interior)[0]
            grad_v_dx2 = K.gradients(v_dx2, data_interior)[0]

            v_dx1dx1 = diff_dx * grad_v_dx1[:, 1:2]
            v_dx2dx2 = diff_dx * grad_v_dx2[:, 2:3]
            v_dx1dx2 = diff_dx * grad_v_dx1[:, 2:3]
            v_dx2dx1 = diff_dx * grad_v_dx2[:, 1:2]

            residual_interior = (
                    v_dt + riskfree_rate_interior * v_interior
                    - (riskfree_rate_interior - volatility1_interior ** 2 / 2) * v_dx1
                    - (riskfree_rate_interior - volatility2_interior ** 2 / 2) * v_dx2
                    - 0.5 * volatility1_interior ** 2 * v_dx1dx1
                    - 0.5 * volatility2_interior ** 2 * v_dx2dx2
                    - 0.5 * correlation_interior
                    * volatility1_interior * volatility2_interior * v_dx1dx2
                    - 0.5 * correlation_interior
                    * volatility2_interior * volatility1_interior * v_dx2dx1
            )

            s_mean_initial = 0.5 * (
                    tf.math.exp(x1_initial) + tf.math.exp(x2_initial))
            payoff_initial = K.maximum(s_mean_initial - strike_price, 0)

            loss_interior = K.mean(K.square(residual_interior))
            loss_initial = K.mean(K.square(v_initial - payoff_initial))

            loss = loss_initial + loss_interior

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss,
                "loss initial": loss_initial,
                "loss interior": loss_interior}
