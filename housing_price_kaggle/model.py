def solve_parameters(X, Y, Wi):
    dL_dW_atWi = 2 * X.T @ (X @ Wi - Y)
    alpha = 1e-2
    sum_dLdW_atWi = dL_dW_atWi.mean()
    print(sum_dLdW_atWi)
    i = 1
    while i < 50000:  # sum_dLdW_atWi != 0
        W_i1 = Wi - alpha * dL_dW_atWi
        dL_dW_atWi1 = 2 * X.T @ (X @ W_i1 - Y)
        sum_dldW_atWi1 = dL_dW_atWi1.mean()
        if i % 500 == 0:
            print(
                f"======iteration {i}\n, Gradient-mean: {sum_dldW_atWi1} -- Alpha: { alpha} -- Weight: {Wi}"
            )
        if abs(sum_dldW_atWi1) > abs(sum_dLdW_atWi):
            alpha = alpha * 1e-2
        else:
            dL_dW_atWi = dL_dW_atWi1
            Wi = W_i1
            i += 1
    else:
        return Wi


def gradient_descent(
    alpha,
    num_iterations,
    X,
    Y,
    W,
    X_validate,
    Y_validate,
    early_stopping_tolerance=1000,
    verbose=True,
    display_interval=100,
):

    # num_iterations = 30000
    # alpha = 1e-02
    previous_validation_loss = float("inf")
    validation_tolerance = 0
    for i in range(num_iterations):
        loss = ((X @ W - Y) ** 2).mean()
        loss_validated = ((X_validate @ W - Y_validate) ** 2).mean()
        gradient = 2 * X.T @ (X @ W - Y) / len(Y)
        W = W - alpha * gradient
        if verbose:
            if i % (num_iterations / display_interval) == 0:
                print(
                    f"Iteration {i} -- loss: {loss} -- gradient:{gradient.mean()} -- loss for validation: {loss_validated}"
                )

        # Validation tolerance STOP TRAINING
        if loss_validated > previous_validation_loss:
            validation_tolerance += 1
        if validation_tolerance > early_stopping_tolerance:
            return W
        previous_validation_loss = loss_validated

    return W
