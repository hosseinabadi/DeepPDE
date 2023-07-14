def transform_ab_to_cd(x, a, b, c, d):
    """
    Perform a linear transformation of a scalar from the souce interval
    to the target interval.

    Keyword arguments:
    x -- scalar point(s) to transform
    a, b -- interval to transform from
    c, d -- interval to transform to
    """
    return c + (x - a) * (d - c) / (b - a)


def transform_to_logprice(x):
    """ Transform normalised variable to the log-price. """
    return transform_ab_to_cd(x, normalised_min, normalised_max, x_min, x_max)


def transform_to_time(t):
    """ Transform normalised variable to the time variable. """
    return transform_ab_to_cd(t, normalised_min, normalised_max, t_min, t_max)


def normalise_logprice(x):
    """ Transform log-price to its corresponding normalised variable. """
    return transform_ab_to_cd(x, x_min, x_max, normalised_min, normalised_max)


def normalise_time(t):
    """ Transform time to its corresponding normalised variable. """
    return transform_ab_to_cd(t, t_min, t_max, normalised_min, normalised_max)


def transform_to_riskfree_rate(mu_1):
    """ Transform normalised variable to the risk-free rate. """
    return transform_ab_to_cd(mu_1, normalised_min, normalised_max,
                              riskfree_rate_min, riskfree_rate_max)


def transform_to_volatility(mu_2):
    """ Transform normalised variable to the volatility. """
    return transform_ab_to_cd(mu_2, normalised_min, normalised_max,
                              volatility_min, volatility_max)


def transform_to_correlation(mu_3):
    """ Transform normalised variable to the correlation. """
    return transform_ab_to_cd(mu_3, normalised_min, normalised_max,
                              correlation_min, correlation_max)


def normalise_riskfree_rate(riskfree_rate):
    """ Transform risk-free rate to its corresponding normalised variable. """
    return transform_ab_to_cd(riskfree_rate,
                              riskfree_rate_min, riskfree_rate_max,
                              normalised_min, normalised_max)


def normalise_volatility(volatility):
    """ Transform volatility to its corresponding normalised variable. """
    return transform_ab_to_cd(volatility, volatility_min, volatility_max,
                              normalised_min, normalised_max)


def normalise_correlation(correlation):
    """ Transform correlation to its corresponding normalised variable. """
    return transform_ab_to_cd(correlation, correlation_min, correlation_max,
                              normalised_min, normalised_max)


def localisation(t, s1, s2, riskfree_rate=riskfree_rate_eval):
    """ Return the value of the localisation used in the network. """
    return 1/localisation_parameter * np.log(1 +
                    np.exp(localisation_parameter * (
                        0.5*(s1+s2) - np.exp(-riskfree_rate*t)*strike_price))
                    )

def get_random_points_of_interest(nr_samples,
                    t_min_interest=t_min_interest,
                    t_max_interest=t_max_interest,
                    s_min_interest=s_min_interest,
                    s_max_interest=s_max_interest,
                    parameter_min_interest_normalised=normalised_min,
                    parameter_max_interest_normalised=normalised_max):
    """ Get a number of random points within the defined domain of interest. """
    t_sample = np.random.uniform(t_min_interest, t_max_interest,
                                 [nr_samples, 1])
    t_sample_normalised = normalise_time(t_sample)

    s_sample = np.random.uniform(
        s_min_interest, s_max_interest, [nr_samples, dimension_state])
    s1_sample = s_sample[:, 0:1]
    s2_sample = s_sample[:, 1:2]
    x_sample_normalised = normalise_logprice(np.log(s_sample))

    parameter_sample_normalised = np.random.uniform(
        normalised_min, normalised_max, [nr_samples, dimension_parameter])
    data_normalised = np.concatenate(
        (t_sample_normalised, x_sample_normalised, parameter_sample_normalised),
        axis=1
        )

    riskfree_rate_sample = transform_to_riskfree_rate(
        parameter_sample_normalised[:, 0])
    volatility1_sample = transform_to_volatility(
        parameter_sample_normalised[:, 1])
    volatility2_sample = transform_to_volatility(
        parameter_sample_normalised[:, 2])
    correlation_sample = transform_to_correlation(
        parameter_sample_normalised[:, 3])

    return data_normalised, t_sample.reshape(-1), s1_sample.reshape(-1), \
            s2_sample.reshape(-1), riskfree_rate_sample, volatility1_sample, \
            volatility2_sample, correlation_sample


def get_points_for_plot_at_fixed_time(t_fixed=t_max,
                s_min_interest=s_min_interest, s_max_interest=s_max_interest,
                riskfree_rate_fixed=riskfree_rate_eval,
                volatility1_fixed=volatility1_eval,
                volatility2_fixed=volatility2_eval,
                correlation_fixed=correlation_eval,
                n_plot=nr_samples_surface_plot):
    """ Get the spacial and normalised values for surface plots
    at fixed time and parameter, varying both asset prices.
    """
    s1_plot = np.linspace(s_min_interest, s_max_interest, n_plot).reshape(-1,1)
    s2_plot = np.linspace(s_min_interest, s_max_interest, n_plot).reshape(-1,1)
    [s1_plot_mesh, s2_plot_mesh] = np.meshgrid(s1_plot, s2_plot, indexing='ij')

    x1_plot_mesh_normalised = normalise_logprice(
        np.log(s1_plot_mesh)).reshape(-1,1)

    x2_plot_mesh_normalised = normalise_logprice(
        np.log(s2_plot_mesh)).reshape(-1,1)

    t_mesh = t_fixed  * np.ones((n_plot**2, 1))
    t_mesh_normalised = normalise_time(t_mesh)

    parameter1_mesh_normalised = (normalise_riskfree_rate(riskfree_rate_fixed)
                                                      * np.ones((n_plot**2, 1)))
    parameter2_mesh_normalised = (normalise_volatility(volatility1_fixed)
                                                      * np.ones((n_plot**2, 1)))
    parameter3_mesh_normalised = (normalise_volatility(volatility2_fixed)
                                                      * np.ones((n_plot**2, 1)))
    parameter4_mesh_normalised = (normalise_correlation(correlation_fixed)
                                                      * np.ones((n_plot**2, 1)))

    x_plot_normalised = np.concatenate((t_mesh_normalised,
                                        x1_plot_mesh_normalised,
                                        x2_plot_mesh_normalised,
                                        parameter1_mesh_normalised,
                                        parameter2_mesh_normalised,
                                        parameter3_mesh_normalised,
                                        parameter4_mesh_normalised), axis=1)


    return s1_plot_mesh, s2_plot_mesh, x_plot_normalised



