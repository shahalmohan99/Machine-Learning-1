import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from plot_utils import (plot_scatterplot_and_line, plot_scatterplot_and_polynomial, 
                        plot_logistic_regression, plot_datapoints, plot_3d_surface, plot_2d_contour,
                        plot_function_over_iterations)
from linear_regression import (fit_univariate_lin_model, 
                               fit_multiple_lin_model, 
                               univariate_loss, multiple_loss,
                               calculate_pearson_correlation,
                               compute_design_matrix,
                               compute_polynomial_design_matrix)
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)
from gradient_descent import rastrigin, gradient_rastrigin, gradient_descent, finite_difference_gradient_approx


def task_1(use_linalg_formulation=False):
    print('---- Task 1 ----')
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3, 
                    "duration": 4, "exercise_intensity": 5, 
                    "fitness_level": 6, "calories": 7}

    # After loading the data, you can for example access it like this: 
    # `smartwatch_data[:, column_to_id['hours_sleep']]`
    # Make sure the 'plots' folder exists, otherwise, saving plots will raise a FileNotFoundError.
    smartwatch_data = np.load('data/smartwatch_data.npy')

    # TODO: Implement Task 1.1.2: Find 3 pairs of features that have a linear relationship.
    # For each pair, fit a univariate linear regression model: If ``use_linalg_formulation`` is False,
    # call `fit_univariate_lin_model`, otherwise use the linalg formulation in `fit_multiple_lin_model` (Task 1.2.2).
    # For each pair, also calculate and report the Pearson correlation coefficient, the theta vector you found, 
    # the MSE, and plot the data points together with the linear function.
    # Repeat the process for 3 pairs of features that do not have a meaningful linear relationship.
    
    print("\nTask 1.1.2: Univariate Linear Regression")
    correlated_pairs = [("duration", "calories"), ("exercise_intensity", "calories"), ("fitness_level", "max_pulse")]
    uncorrelated_pairs = [("hours_sleep", "avg_pulse"), ("hours_work", "fitness_level"), ("avg_pulse", "calories")]

    for pair_list, label in [(correlated_pairs, "Correlated"), (uncorrelated_pairs, "Uncorrelated")]:
        print(f"\n{label} Pairs:")
        for x_col, y_col in pair_list:
            x = smartwatch_data[:, column_to_id[x_col]]
            y = smartwatch_data[:, column_to_id[y_col]]

            if use_linalg_formulation:
                X_design = compute_design_matrix(x.reshape(-1, 1))
                theta = fit_multiple_lin_model(X_design, y)
            else:
                theta = fit_univariate_lin_model(x, y)

            mse = univariate_loss(x, y, theta)
            p = calculate_pearson_correlation(x, y)

            print(f"{x_col} vs {y_col}:")
            print(f"  Theta: {theta}")
            print(f"  MSE: {mse:.4f}")
            print(f"  Pearson r: {p:.4f}")

            plot_scatterplot_and_line(x, y, theta, xlabel=x_col, ylabel=y_col,
                                      title=f'{x_col} vs {y_col} ({label})',
                                      figname=f'{x_col}_vs_{y_col}_{label.lower()}')


    # TODO: Implement Task 1.2.3: Multiple linear regression
    # Select two additional features, compute the design matrix, and fit the multiple linear regression model.
    # Report the MSE and the theta vector.

    print("Task 1.2.3: Multiple Linear Regression")
    X_multi = smartwatch_data[:, [column_to_id["duration"], column_to_id["exercise_intensity"]]]
    y_multi = smartwatch_data[:, column_to_id["calories"]]

    X_design = compute_design_matrix(X_multi)
    theta_multi = fit_multiple_lin_model(X_design, y_multi)
    mse_multi = np.mean((X_design @ theta_multi - y_multi) ** 2)

    print(f"  Theta: {theta_multi}")
    print(f"  MSE: {mse_multi:.4f}")
    


    # TODO: Implement Task 1.3.2: Polynomial regression
    # For the feature-target pair of choice, compute the polynomial design matrix with an appropriate degree K, 
    # fit the model, and plot the data points together with the polynomial function.
    # Report the MSE and the theta vector.

    print("\nTask 1.3.2: Polynomial Regression")
    x_poly = smartwatch_data[:, column_to_id["duration"]]
    y_poly = smartwatch_data[:, column_to_id["calories"]]

    for K in [1, 2, 3, 5]:
        X_poly = compute_polynomial_design_matrix(x_poly, K)
        theta_poly = fit_multiple_lin_model(X_poly, y_poly)
        y_pred = X_poly @ theta_poly
        mse_poly = np.mean((y_poly - y_pred) ** 2)

        print(f"  K = {K} => MSE: {mse_poly:.4f}, Theta shape: {theta_poly.shape}")
        plot_scatterplot_and_polynomial(x_poly, y_poly, theta_poly,
                                        xlabel="duration", ylabel="calories",
                                        title=f"Polynomial Fit K={K}",
                                        figname=f"poly_duration_calories_K{K}")
        


    # TODO: Implement Task 1.3.3: Use x_small and y_small to fit a polynomial model.
    # Find and report the smallest K that gets zero loss. Plot the data points and the polynomial function.
    
    print("\nTask 1.3.3: Polynomial Fit on Small Data")
    x_small = smartwatch_data[:5, column_to_id['duration']]
    y_small = smartwatch_data[:5, column_to_id['calories']]
    
    for K in range(1, 10):
        X_small_poly = compute_polynomial_design_matrix(x_small, K)
        theta_small = fit_multiple_lin_model(X_small_poly, y_small)
        y_pred_small = X_small_poly @ theta_small
        loss = np.mean((y_pred_small - y_small) ** 2)
        if loss < 1e-10:
            print(f"  Smallest K with zero loss: {K}")
            print(f"  Theta: {theta_small}")
            plot_scatterplot_and_polynomial(x_small, y_small, theta_small,
                                            xlabel="duration", ylabel="calories",
                                            title=f"Small Data Polynomial Fit (K={K})",
                                            figname=f"poly_small_K{K}")
            break


def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # TODO: Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load("data/X-1-data.npy") # TODO: change me
            y = np.load("data/targets-dataset-1.npy") # TODO: change me
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            # TODO: Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load("data/X-1-data.npy") # TODO: change me
            y = np.load("data/targets-dataset-2.npy") # TODO: change me
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.load("data/X-2-data.npy") # TODO: change me
            y = np.load("data/targets-dataset-3.npy") # TODO: change me
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        # TODO: Split the dataset using the `train_test_split` function.
        # The parameter `random_state` should be set to 0.
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        # TODO: Fit the model to the data using the `fit` method of the classifier `clf`
        
        clf.fit(X_train, y_train)
        
        # TODO: Use the `score` method of the classifier `clf` to calculate accuracy
        acc_train, acc_test = clf.score(X_train, y_train), clf.score(X_test, y_test) 

        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')
        
        # TODO: Use the `predict_proba` method of the classifier `clf` to
        # calculate the predicted probabilities on the training set
        yhat_train = clf.predict_proba(X_train)[:, 1]
        
        # TODO: Use the `predict_proba` method of the classifier `clf` to
        # calculate the predicted probabilities on the test set
        yhat_test = clf.predict_proba(X_test)[:, 1] 

        # TODO: Use the `log_loss` function to calculate the cross-entropy loss
        #  (once on the training set, once on the test set).
        #  You need to pass (1) the true binary labels and (2) the probability of the *positive* class to `log_loss`.
        #  Since the output of `predict_proba` is of shape (n_samples, n_classes), you need to select the probabilities
        #  of the positive class by indexing the second column (index 1).
        loss_train, loss_test = log_loss(y_train, yhat_train), log_loss(y_test, yhat_test)
        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test,  f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        # TODO: Print theta vector (and also the bias term). Hint: Check the attributes of the classifier
        classifier_weights, classifier_bias = clf.coef_, clf.intercept_
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3(initial_plot=True):
    print('\n---- Task 3 ----')
    # Do *not* change this seed
    np.random.seed(46)

    # TODO: Choose a random starting point using samples from a standard normal distribution
    x0 = np.random.randn()
    y0 = np.random.randn()
    print(f'Starting point: {x0:.4f}, {y0:.4f}')

    if initial_plot:
        # Plot the function to see how it looks like
        plot_3d_surface(rastrigin)
        plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0))

    # TODO: Check if gradient_rastrigin is correct at (x0, y0). 
    # To do this, print the true gradient and the numerical approximation.
    
    true_grad = gradient_rastrigin(x0, y0)
    approx_grad = finite_difference_gradient_approx(rastrigin, x0, y0)
    print(f"True gradient at ({x0:.4f}, {y0:.4f}): {true_grad}")
    print(f"Numerical approximation: {approx_grad}")

    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.
    x_list, y_list, f_list = gradient_descent(
        f=rastrigin,
        df=gradient_rastrigin,
        learning_rate=0.1,
        lr_decay=0.99,
        num_iters=100,
        x0=x0,
        y0=y0
    )

    print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}' )
    print(f'Global optimum: f(0, 0)= {rastrigin(0, 0):.4f}')

    plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0), 
                    x_list=x_list, y_list=y_list)

    # TODO: Create a plot f(x_t, y_t) over iterations t by calling `plot_function_over_iterations` with `f_list`
    plot_function_over_iterations(f_list)


def main():
    np.random.seed(46)

    task_1(use_linalg_formulation=False)
    task_2()
    task_3(initial_plot=True)


if __name__ == '__main__':
    main()
