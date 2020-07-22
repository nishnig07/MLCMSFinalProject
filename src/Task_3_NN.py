# Approximation using a Neural Network

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# The NN-model constructed in the paper has two hidden dimensions of width 16
# It uses Tan-H activation functions mostly
# Data-set is derived by random sampling


def hamiltonian_function_pendulum(q, p):
    return np.square(p)*0.5 + (1-np.cos(q))


# dp/dt and dq/dt
def get_p_dot(data):
    return -np.sin(data)


def get_q_dot(data):
    return data


def step(data, c_1, c_2, c_3, c_4, model, opt, qdot, pdot, training=True):
    """
    Step function for the model
    :param opt: Optimizer used for training the model
    :param model: The NN model of type tf.keras.Model
    :param c_1: Float values putting on a weight over the loss terms f1.
    :param c_2: Float values putting on a weight over the loss terms f2.
    :param c_3: Float values putting on a weight over the loss terms f3.
    :param c_4: Float values putting on a weight over the loss terms f4.
    :param data: Training data of shape (n, 2)
    :param training: Boolean to check if this function is being used for training data or validation data.
                     For validation data, the weights aren't updated.
    :return: loss
    """
    train_data_tf = data
    if training:
        n_data = data.numpy()
    else:
        n_data = data
        train_data_tf = tf.convert_to_tensor(data, dtype=float)

    # Calculate some arbitrary H0 value
    h_0 = hamiltonian_function_pendulum(np.mean(n_data), np.mean(n_data[:, 1]))

    # Automatic differentiation for calculating gradients required by the loss function
    with tf.GradientTape() as tape:
        with tf.GradientTape() as tape2:
            tape2.watch(train_data_tf)
            h_hat = model(train_data_tf)
        grad_h = tape2.gradient(h_hat, train_data_tf)
        dh_hat_dq = grad_h[:, 0]
        dh_hat_dp = grad_h[:, 1]
        q_dot = qdot(data[:, 1])
        p_dot = pdot(data[:, 0])
        f1 = (dh_hat_dp - q_dot) ** 2
        f2 = (dh_hat_dq + p_dot) ** 2
        f3 = (h_hat - h_0) ** 2
        f4 = (dh_hat_dq * q_dot + dh_hat_dp * p_dot) ** 2
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        loss = (c_1 * f1 + c_2 * f2 + c_3 * f3 + c_4 * f4) / 4
        loss = tf.reduce_mean(loss)
        # print(loss)
    if training:
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def get_model(layer_size):
    return tf.keras.models.Sequential([
                            tf.keras.layers.InputLayer(input_shape=(2,)),
                            tf.keras.layers.Dense(layer_size, activation='softplus'),
                            tf.keras.layers.Dense(layer_size, activation='softplus'),
                            tf.keras.layers.Dense(1, activation='softplus')
                        ])


def short_trajectories(p_val, q_val):
    dpdt = solve_ivp(lambda t, y, q: -np.sin(q), [0, 1], p_val, args=(q_val,), t_eval=[0.1])
    dqdt = solve_ivp(lambda t, y, p: p, [0, 1], q_val, args=(p_val,), t_eval=[0.1])
    return dpdt.y[:, 0], dqdt.y[:, 0]


if __name__ == '__main__':
    # Generate data
    n = 20000   # Number of training samples

    # Training set
    np.random.seed(12345)
    train_p = np.random.uniform(-1, 1, n)
    train_q = np.random.uniform(-2*np.pi, 2*np.pi, n)

    # Validation set
    np.random.seed(12345)
    val_p = np.random.uniform(-1, 1, 200)
    val_q = np.random.uniform(-2*np.pi, 2*np.pi, 200)

    # Small trajectories from initial points
    dp_dt, dq_dt = short_trajectories(train_p, train_q)
    dp_dt_val, dq_dt_val = short_trajectories(val_p, val_q)

    # Plot the initial contour
    p, q = np.meshgrid(np.linspace(-9, 9), np.linspace(-1.5, 1.5))
    plt.figure()
    plt.contour(p, q, hamiltonian_function_pendulum(p, q), linewidths=4)
    plt.colorbar()
    plt.scatter(train_q[::100], train_p[::100])
    plt.scatter(dq_dt[::100], dp_dt[::100])
    plt.show()

    # confirm the shapes
    # print(train_q.shape, train_p.shape)
    # print(q_dot.shape, p_dot.shape)

    # Hyper-params:
    # More hyper params can be added to the lists for comparisons.
    learning_rate = [5e-3]
    epochs = 15
    batch_size = 256
    hidden_layer_size = 16
    c1_range = [0.1, 1]
    c2_range = [1, 10]
    c3_range = [0]
    c4_range = [1]
    for lr in learning_rate:
        for c1 in c1_range:
            for c2 in c2_range:
                for c3 in c3_range:
                    for c4 in c4_range:
                        params = "learning_rate = " +str(lr) + " c1 = "+str(c1)+ " c2 = "+str(c2)+ "c3 = "+str(c3)+ "c4 = "+ str(c4)
                        train_data = np.vstack((train_q, train_p)).T
                        val_data = np.vstack((dq_dt_val, dp_dt_val)).T

                        # Prepare the model as per the model architecture.
                        # Model outputs the predicted hamiltonian function.
                        model = get_model(hidden_layer_size)

                        # Use Adam-Optimizer
                        opt = tf.keras.optimizers.Adam(lr)

                        num_updates = int(train_data.shape[0] / batch_size)
                        loss_epoch = []
                        train_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_data, dtype=float))
                        for epoch in range(epochs):
                            print("[INFO] starting epoch {}/{}...".format(
                                epoch + 1, epochs))
                            train_data = train_data.shuffle(n)
                            batch_data = train_data.batch(batch_size)
                            loss = np.array([step(i, c_1=c1, c_2=c2, c_3=c3, c_4=c4, model=model, opt=opt,
                                                  pdot=get_p_dot, qdot=get_q_dot)
                                             for i in batch_data])

                            print('loss = ', np.mean(loss), ", val_loss = ",
                                  step(val_data, c_1=c1,
                                       c_2=c2, c_3=c3, c_4=c4, model=model, opt=opt, pdot=get_p_dot,
                                       qdot=get_q_dot, training=False).numpy())
                            loss_epoch.append(np.mean(loss))

                        # Plot the loss vs. epoch graph for a quick analysis of the hyper params
                        plt.figure()
                        plt.plot(range(epochs), loss_epoch)
                        plt.show()

                        # Test the trained model over a validation set
                        plt.figure()

                        # Test the trained model over a validation set
                        temp = model.predict(np.vstack((p.ravel(), q.ravel())).T)[:, 0]

                        # The following line is an experimental thought. Since we got good results when
                        # c3 was set to 0, that in turn resulted in the existence of an additive constant.
                        # We observed for quite a few different hyper params and found out that the additive constant
                        # would be the only thing that was different and the "true" range for the H predictions would be
                        # the same. For eg: 0-3.2 was what we got for the original plot as well as for th plot
                        # in the paper, and when we set c3 to 0, we would get around 3.2-6.4 or 12.4-15.6 or something
                        # around that according to the different hyper-params. Since the additive constant itself
                        # doesn't matter, we thought of ust subtracting the minimum from all our predictions so as to
                        # keep our values bound to 0-3.2. We do not know if doing this will have further implications,
                        # but it seemed to be logical at present and useful for our comparisons. Un-comment it to see
                        # the change or else the additive constant is kept.
                        # temp -= np.min(temp)
                        temp = np.reshape(temp, (-1, p.shape[1]))

                        # Plot the contour
                        plt.contour(p, q, temp, linewidths=4)
                        plt.colorbar()
                        # Display the training points if necessary or ust comment the next line!
                        plt.scatter(dq_dt[::100], dp_dt[::100], c='black')
                        plt.xlabel("q")
                        plt.ylabel("p")
                        plt.title(params)
                        plt.show()
