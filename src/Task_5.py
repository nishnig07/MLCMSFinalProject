# Approximation using a Neural Network

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Task_3_4_NN import step, get_model, short_trajectories


# The NN-model constructed in the paper has two hidden dimensions of width 16
# It uses Tan-H activation functions mostly
# Data-set is derived by random sampling


def hamiltonian_function_pendulum(q, p):
    return np.square(p) * 0.5 + 0.5 * np.square(q) * 0.010


# dp/dt and dq/dt
def get_p_dot(q):
    return -q * 0.010


def get_q_dot(p):
    return p


if __name__ == "__main__":

    # Generate data
    n = 20000  # Number of training samples

    # Training set
    np.random.seed(12345)
    train_p = np.random.uniform(-1, 1, n)
    train_q = np.random.uniform(-6.28, 6.28, n)

    # Validation set
    np.random.seed(12345)
    val_p = np.random.uniform(-1, 1, 200)
    val_q = np.random.uniform(-6.28, 6.28, 200)

    # Small trajectories from initial points
    dp_dt, dq_dt = short_trajectories(train_p, train_q)
    dp_dt_val, dq_dt_val = short_trajectories(val_p, val_q)
    p, q = np.meshgrid(np.linspace(-9, 9), np.linspace(-1.5, 1.5))

    # Plot the original hamiltonian function
    plt.figure()

    # Here q and p are mesh grid points for the contour plot
    plt.contour(p, q, hamiltonian_function_pendulum(p, q), linewidths=4)
    plt.colorbar()

    # Since the training size is big, plot a few of the points
    plt.scatter(train_q[::100], train_p[::100])
    # Short trajectories from the training points
    plt.scatter(dq_dt[::100], dp_dt[::100])
    plt.show()

    # confirm the shapes
    # print(train_q.shape, train_p.shape)
    # print(q_dot.shape, p_dot.shape)

    # Hyper-params:
    # More hyper params can be added to the lists for comparisons
    learning_rate = [1e-3]
    epochs = 15
    batch_size = 256
    hidden_layer_size = 16
    c1_range = [0, 1]
    c2_range = [0, 1]
    c3_range = [0]
    c4_range = [1]
    for lr in learning_rate:
        for c1 in c1_range:
            for c2 in c2_range:
                for c3 in c3_range:
                    for c4 in c4_range:
                        params = "learning_rate = " + str(lr) + " c1 = " + str(c1) + " c2 = " + str(c2) + "c3 = " + str(
                            c3) + "c4 = " + str(c4)
                        train_data = np.vstack((train_q, train_p)).T
                        val_data = np.vstack((dq_dt_val, dp_dt_val)).T

                        # Define the model as per the paper's architecture
                        # Model outputs the predicted hamiltonian function value
                        model = get_model(hidden_layer_size)

                        # Use Adam's optimizer
                        opt = tf.keras.optimizers.Adam(lr)

                        # Train the model over the data
                        num_updates = int(train_data.shape[0] / batch_size)
                        loss_epoch = []
                        train_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_data, dtype=float))
                        for epoch in range(epochs):
                            print("Starting epoch {}/{}...".format(
                                epoch + 1, epochs))
                            train_data = train_data.shuffle(n)
                            batch_data = train_data.batch(batch_size)
                            loss = np.array([step(i, c1, c2, c3, c4, model, opt,
                                                  pdot=get_p_dot, qdot=get_q_dot) for i in batch_data])
                            print('loss = ', np.mean(loss), ", val_loss = ",
                                  step(val_data, c1, c2, c3, c4, model, opt, pdot=get_p_dot,
                                       qdot=get_q_dot, training=False).numpy())
                            loss_epoch.append(np.mean(loss))

                        # Plot the loss vs. epoch graph for a quick analysis of the hyper params
                        plt.figure()
                        plt.plot(range(epochs), loss_epoch)
                        plt.show()

                        # Test the trained model over a validation set
                        temp = model.predict(np.vstack((p.ravel(), q.ravel())).T)[:, 0]

                        # Plot the contour for the trained model
                        temp = np.reshape(temp, (-1, p.shape[1]))
                        plt.figure()
                        plt.contour(p, q, temp, linewidths=4)
                        plt.colorbar()
                        # Display the points if necessary or just comment them!
                        plt.scatter(dq_dt[::100], dp_dt[::100], c='black')
                        plt.xlabel("q")
                        plt.ylabel("p")
                        plt.title(params + " loss = " + str(loss_epoch[-1]))
                        plt.show()
