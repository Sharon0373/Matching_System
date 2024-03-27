import numpy as np
from itertools import permutations
from sklearn.metrics import accuracy_score

def getdata1(scenario="homo", obs_prob=0.3, seed=None):

    if seed is not None:
        np.random.seed(seed)

    n_task, n_worker = 150, 150

    alpha_0 = np.array([2, 0, 0])
    alpha_1 = np.array([0, 2, 0])
    alpha_2 = np.array([0, 0, 2])

    beta11 = np.array([2, 0, 0])
    beta12 = np.array([0, 1, 1])
    beta13 = np.array([1, 1, 0])

    beta21 = np.array([1, 1, 1])
    beta22 = np.array([0, 2, 0])
    beta23 = np.array([1, 1, 0])

    beta31 = np.array([1, 1, 0])
    beta32 = np.array([1, 1, 0])
    beta33 = np.array([0, 0, 2])

    A = np.zeros((n_task, 3))
    B1 = np.zeros((n_worker, 3))
    B2 = np.zeros((n_worker, 3))
    B3 = np.zeros((n_worker, 3))

    if scenario == "homo":

        A[:n_task//3, :] = np.random.multivariate_normal(alpha_0, 0.5 * np.eye(3), n_task//3)
        A[n_task//3: 2*n_task//3, :] = np.random.multivariate_normal(alpha_1, 0.5 * np.eye(3), n_task//3)
        A[2*n_task//3:, :] = np.random.multivariate_normal(alpha_2, 0.5 * np.eye(3), n_task//3)

        B1[:n_worker//3, :] = np.random.multivariate_normal(beta11, 0.5 * np.eye(3), n_worker//3)
        B1[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta12, 0.5 * np.eye(3), n_worker//3)
        B1[2*n_worker//3:, :] = np.random.multivariate_normal(beta13, 0.5 * np.eye(3), n_worker//3)

        B2[:n_worker//3, :] = np.random.multivariate_normal(beta21, 0.5 * np.eye(3), n_worker//3)
        B2[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta22, 0.5 * np.eye(3), n_worker//3)
        B2[2*n_worker//3:, :] = np.random.multivariate_normal(beta23, 0.5 * np.eye(3), n_worker//3)

        B3[:n_worker//3, :] = np.random.multivariate_normal(beta31, 0.5 * np.eye(3), n_worker//3)
        B3[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta32, 0.5 * np.eye(3), n_worker//3)
        B3[2*n_worker//3:, :] = np.random.multivariate_normal(beta33, 0.5 * np.eye(3), n_worker//3)

    elif scenario == "hetero":
    
        A[:n_task//3, :] = np.random.multivariate_normal(alpha_0, 2 * np.eye(3), n_task//3)
        A[n_task//3: 2*n_task//3, :] = np.random.multivariate_normal(alpha_1, 2 * np.eye(3), n_task//3)
        A[2*n_task//3:, :] = np.random.multivariate_normal(alpha_2, 2 * np.eye(3), n_task//3)

        B1[:n_worker//3, :] = np.random.multivariate_normal(beta11, np.eye(3), n_worker//3)
        B1[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta12, np.eye(3), n_worker//3)
        B1[2*n_worker//3:, :] = np.random.multivariate_normal(beta13, np.eye(3), n_worker//3)

        B2[:n_worker//3, :] = np.random.multivariate_normal(beta21, np.eye(3), n_worker//3)
        B2[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta22, np.eye(3), n_worker//3)
        B2[2*n_worker//3:, :] = np.random.multivariate_normal(beta23, np.eye(3), n_worker//3)

        B3[:n_worker//3, :] = np.random.multivariate_normal(beta31, np.eye(3), n_worker//3)
        B3[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta32, np.eye(3), n_worker//3)
        B3[2*n_worker//3:, :] = np.random.multivariate_normal(beta33, np.eye(3), n_worker//3)

    R_tsr = np.zeros((n_task, n_worker, 3))
    R_tsr[:, :, 0] = A.dot(B1.T)
    R_tsr[:, :, 1] = A.dot(B2.T)
    R_tsr[:, :, 2] = A.dot(B3.T)
    R = np.argmax(R_tsr, axis=2)

    list = []

    for i in range(n_task):

        sub_n_worker = int(n_worker * obs_prob)
        sub_worker = np.sort(np.random.choice(np.arange(n_worker), size=sub_n_worker, replace=False))
        tmp = np.zeros((sub_n_worker, 3))
        tmp[:, 0] = i
        tmp[:, 1] = sub_worker
        tmp[:, 2] = R[i, sub_worker]
        list.append(tmp)

    R_long = np.concatenate(list, axis=0)
    
    true_label = np.array([0] * (n_task // 3) + [1] * (n_task // 3) + [2] * (n_task // 3))

    return R_long, true_label

def getdata2(ph=0.2, scenario="homo", seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    n_task, n_worker = 100, 100

    alpha_1 = np.array([5, 5])
    alpha_0 = np.array([-5, -5])

    beta_1 = np.array([5, 3])
    beta_2 = np.array([5, -5])

    A = np.zeros((n_task, 2))
    B = np.zeros((n_worker, 2))

    if scenario == "homo":

        A[:n_task//2, :] = np.random.multivariate_normal(alpha_1, 0.5 * np.eye(2), n_task//2)
        A[n_task//2:, :] = np.random.multivariate_normal(alpha_0, 0.5 * np.eye(2), n_task//2)

        B[:int(n_worker * ph), :] = np.random.multivariate_normal(beta_1, np.eye(2), int(n_worker * ph))
        B[int(n_worker * ph):, :] = np.random.multivariate_normal(beta_2, np.eye(2), int(n_worker * (1 - ph)))

    elif scenario == "hetero":

        A[:n_task//2, :] = np.random.multivariate_normal(alpha_1, 2.5 * np.eye(2), n_task//2)
        A[n_task//2:, :] = np.random.multivariate_normal(alpha_0, 2.5 * np.eye(2), n_task//2)

        B[:int(n_worker * ph), :] = np.random.multivariate_normal(beta_1, np.eye(2), int(n_worker * ph))
        B[int(n_worker * ph):, :] = np.random.multivariate_normal(beta_2, np.eye(2), int(n_worker * (1 - ph)))


    R = sigmoid(A.dot(B.T))
    R[R > 1/2] = 1
    R[R <= 1/2] = 0

    list = []

    for i in range(n_task):

        sub_n_worker = int(n_worker * 0.3)
        sub_worker = np.sort(np.random.choice(np.arange(n_worker), size=sub_n_worker, replace=False))
        tmp = np.zeros((sub_n_worker, 3))
        tmp[:, 0] = i
        tmp[:, 1] = sub_worker
        tmp[:, 2] = R[i, sub_worker]
        list.append(tmp)

    R_long = np.concatenate(list, axis=0)
    
    true_label = np.zeros((n_task, 2))
    true_label[:, 0] = np.arange(n_task)
    true_label[:, 1] = np.array([1] * (n_task//2) + [0] * (n_task //2))

    return R_long, true_label

def getdata3(case=1, seed=None):

    if seed is not None:
        np.random.seed(seed)

    if case == 1:

        n_task, n_worker = 100, 300

        alpha_1 = np.array([2, 2])
        alpha_0 = np.array([-2, -2])

        beta_1 = np.array([2, 2])
        beta_2 = np.array([2, -1])

        A = np.zeros((n_task, 2))
        B = np.zeros((n_worker, 2))

        A[:n_task//2, :] = np.random.multivariate_normal(alpha_1, 2 * np.eye(2), n_task//2)
        A[n_task//2:, :] = np.random.multivariate_normal(alpha_0, 2 * np.eye(2), n_task//2)

        B[:n_worker//2, :] = np.random.multivariate_normal(beta_1, np.eye(2), n_worker//2)
        B[n_worker//2:, :] = np.random.multivariate_normal(beta_2, np.eye(2), n_worker//2)

    elif case == 2:

        n_task, n_worker = 100, 300

        alpha_1 = np.array([2, 2])
        alpha_0 = np.array([-2, -2])

        beta_1 = np.array([2, 2])
        beta_2 = np.array([1, -2])

        A = np.zeros((n_task, 2))
        B = np.zeros((n_worker, 2))

        A[:n_task//2, :] = np.random.multivariate_normal(alpha_1, 2 * np.eye(2), n_task//2)
        A[n_task//2:, :] = np.random.multivariate_normal(alpha_0, 2 * np.eye(2), n_task//2)

        B[:n_worker//2, :] = np.random.multivariate_normal(beta_1, np.eye(2), n_worker//2)
        B[n_worker//2:, :] = np.random.multivariate_normal(beta_2, np.eye(2), n_worker//2)

    elif case == 3:
    
        n_task, n_worker = 100, 300

        alpha_1 = np.array([2, 2])
        alpha_0 = np.array([-2, -2])

        beta_1 = np.array([2, 2])
        beta_2 = np.array([2, -1])
        beta_3 = np.array([1, -2])


        A = np.zeros((n_task, 2))
        B = np.zeros((n_worker, 2))

        A[:n_task//2, :] = np.random.multivariate_normal(alpha_1, 2 * np.eye(2), n_task//2)
        A[n_task//2:, :] = np.random.multivariate_normal(alpha_0, 2 * np.eye(2), n_task//2)

        B[:n_worker//3, :] = np.random.multivariate_normal(beta_1, 0.5 * np.eye(2), n_worker//3)
        B[n_worker//3: 2 * n_worker//3, :] = np.random.multivariate_normal(beta_2, 0.5 * np.eye(2), n_worker//3)
        B[2 * n_worker//3:, :] = np.random.multivariate_normal(beta_3, 0.5 * np.eye(2), n_worker//3)

    R = sigmoid(A.dot(B.T))
    R[R > 1/2] = 1
    R[R <= 1/2] = 0

    list = []

    for i in range(n_task):

        sub_n_worker = int(n_worker * 0.3)
        sub_worker = np.sort(np.random.choice(np.arange(n_worker), size=sub_n_worker, replace=False))
        tmp = np.zeros((sub_n_worker, 3))
        tmp[:, 0] = i
        tmp[:, 1] = sub_worker
        tmp[:, 2] = R[i, sub_worker]
        list.append(tmp)

    R_long = np.concatenate(list, axis=0)
    
    true_label = np.zeros((n_task, 2))
    true_label[:, 0] = np.arange(n_task)
    true_label[:, 1] = np.array([1] * (n_task//2) + [0] * (n_task //2))

    return R_long, true_label

def getdata4(case=1, seed=None):

    if seed is not None:
        np.random.seed(seed)

    n_task, n_worker = 150, 300

    A = np.zeros((n_task, 3))
    B1 = np.zeros((n_worker, 3))
    B2 = np.zeros((n_worker, 3))
    B3 = np.zeros((n_worker, 3))


    if case == 1:

        alpha_0 = np.array([2, 0, 0])
        alpha_1 = np.array([0, 2, 0])
        alpha_2 = np.array([0, 0, 2])

        beta_11 = np.array([2, 0, 0])
        beta_12 = np.array([2, 0, 0])

        beta_21 = np.array([0, 2, 0])
        beta_22 = np.array([0, 2, 1])

        beta_31 = np.array([0, 0, 2])
        beta_32 = np.array([0, 0, 2])

        A[:n_task//3, :] = np.random.multivariate_normal(alpha_0, 0.5 * np.eye(3), n_task//3)
        A[n_task//3: 2*n_task//3, :] = np.random.multivariate_normal(alpha_1, 0.5 * np.eye(3), n_task//3)
        A[2*n_task//3:, :] = np.random.multivariate_normal(alpha_2, 0.5 * np.eye(3), n_task//3)

        B1[:n_worker//2, :] = np.random.multivariate_normal(beta_11, 0.5 * np.eye(3), n_worker//2)
        B1[n_worker//2:, :] = np.random.multivariate_normal(beta_12, 0.5 * np.eye(3), n_worker//2)

        B2[:n_worker//2, :] = np.random.multivariate_normal(beta_21, 0.5 * np.eye(3), n_worker//2)
        B2[n_worker//2:, :] = np.random.multivariate_normal(beta_22, 0.5 * np.eye(3), n_worker//2)

        B3[:n_worker//2, :] = np.random.multivariate_normal(beta_31, 0.5 * np.eye(3), n_worker//2)
        B3[n_worker//2:, :] = np.random.multivariate_normal(beta_32, 0.5 * np.eye(3), n_worker//2)

    elif case == 2:

        alpha_0 = np.array([2, 0, 0])
        alpha_1 = np.array([0, 2, 0])
        alpha_2 = np.array([0, 0, 2])

        beta_11 = np.array([2, 0, 0])
        beta_12 = np.array([1, 2, 0])

        beta_21 = np.array([0, 2, 0])
        beta_22 = np.array([0, 2, 1])

        beta_31 = np.array([0, 0, 2])
        beta_32 = np.array([1, 1, 1])

        A[:n_task//3, :] = np.random.multivariate_normal(alpha_0, 0.5 * np.eye(3), n_task//3)
        A[n_task//3: 2*n_task//3, :] = np.random.multivariate_normal(alpha_1, 0.5 * np.eye(3), n_task//3)
        A[2*n_task//3:, :] = np.random.multivariate_normal(alpha_2, 0.5 * np.eye(3), n_task//3)

        B1[:n_worker//2, :] = np.random.multivariate_normal(beta_11, 0.5 * np.eye(3), n_worker//2)
        B1[n_worker//2:, :] = np.random.multivariate_normal(beta_12, 0.5 * np.eye(3), n_worker//2)

        B2[:n_worker//2, :] = np.random.multivariate_normal(beta_21, 0.5 * np.eye(3), n_worker//2)
        B2[n_worker//2:, :] = np.random.multivariate_normal(beta_22, 0.5 * np.eye(3), n_worker//2)

        B3[:n_worker//2, :] = np.random.multivariate_normal(beta_31, 0.5 * np.eye(3), n_worker//2)
        B3[n_worker//2:, :] = np.random.multivariate_normal(beta_32, 0.5 * np.eye(3), n_worker//2)

    elif case == 3:

        alpha_0 = np.array([2, 0, 0])
        alpha_1 = np.array([0, 2, 0])
        alpha_2 = np.array([0, 0, 2])

        beta_11 = np.array([2, 0, 0])
        beta_12 = np.array([2, 0, 0])
        beta_13 = np.array([1, 2, 0])

        beta_21 = np.array([0, 2, 0])
        beta_22 = np.array([0, 2, 1])
        beta_23 = np.array([0, 2, 1])

        beta_31 = np.array([0, 0, 2])
        beta_32 = np.array([0, 0, 2])
        beta_33 = np.array([1, 1, 1])

        A[:n_task//3, :] = np.random.multivariate_normal(alpha_0, 0.5 * np.eye(3), n_task//3)
        A[n_task//3: 2*n_task//3, :] = np.random.multivariate_normal(alpha_1, 0.5 * np.eye(3), n_task//3)
        A[2*n_task//3:, :] = np.random.multivariate_normal(alpha_2, 0.5 * np.eye(3), n_task//3)

        B1[:n_worker//3, :] = np.random.multivariate_normal(beta_11, 0.5 * np.eye(3), n_worker//3)
        B1[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta_12, 0.5 * np.eye(3), n_worker//3)
        B1[2*n_worker//3:, :] = np.random.multivariate_normal(beta_13, 0.5 * np.eye(3), n_worker//3)

        B2[:n_worker//3, :] = np.random.multivariate_normal(beta_21, 0.5 * np.eye(3), n_worker//3)
        B2[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta_22, 0.5 * np.eye(3), n_worker//3)
        B2[2*n_worker//3:, :] = np.random.multivariate_normal(beta_23, 0.5 * np.eye(3), n_worker//3)

        B3[:n_worker//3, :] = np.random.multivariate_normal(beta_31, 0.5 * np.eye(3), n_worker//3)
        B3[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta_32, 0.5 * np.eye(3), n_worker//3)
        B3[2*n_worker//3:, :] = np.random.multivariate_normal(beta_33, 0.5 * np.eye(3), n_worker//3)

    elif case == 4:

        alpha_0 = np.array([2, 0, 0])
        alpha_1 = np.array([0, 2, 0])
        alpha_2 = np.array([0, 0, 2])

        beta_11 = np.array([2, 0, 0])
        beta_12 = np.array([2, 0, 0])
        beta_13 = np.array([1, 2, 0])

        beta_21 = np.array([0, 2, 0])
        beta_22 = np.array([1, 0, 1])
        beta_23 = np.array([0, 2, 1])

        beta_31 = np.array([0, 0, 2])
        beta_32 = np.array([0, 2, 0])
        beta_33 = np.array([1, 1, 1])

        A[:n_task//3, :] = np.random.multivariate_normal(alpha_0, 0.5 * np.eye(3), n_task//3)
        A[n_task//3: 2*n_task//3, :] = np.random.multivariate_normal(alpha_1, 0.5 * np.eye(3), n_task//3)
        A[2*n_task//3:, :] = np.random.multivariate_normal(alpha_2, 0.5 * np.eye(3), n_task//3)

        B1[:n_worker//3, :] = np.random.multivariate_normal(beta_11, 0.5 * np.eye(3), n_worker//3)
        B1[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta_12, 0.5 * np.eye(3), n_worker//3)
        B1[2*n_worker//3:, :] = np.random.multivariate_normal(beta_13, 0.5 * np.eye(3), n_worker//3)

        B2[:n_worker//3, :] = np.random.multivariate_normal(beta_21, 0.5 * np.eye(3), n_worker//3)
        B2[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta_22, 0.5 * np.eye(3), n_worker//3)
        B2[2*n_worker//3:, :] = np.random.multivariate_normal(beta_23, 0.5 * np.eye(3), n_worker//3)

        B3[:n_worker//3, :] = np.random.multivariate_normal(beta_31, 0.5 * np.eye(3), n_worker//3)
        B3[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta_32, 0.5 * np.eye(3), n_worker//3)
        B3[2*n_worker//3:, :] = np.random.multivariate_normal(beta_33, 0.5 * np.eye(3), n_worker//3)

    R_tsr = np.zeros((n_task, n_worker, 3))
    R_tsr[:, :, 0] = A.dot(B1.T)
    R_tsr[:, :, 1] = A.dot(B2.T)
    R_tsr[:, :, 2] = A.dot(B3.T)
    R = np.argmax(R_tsr, axis=2)

    list = []

    for i in range(n_task):

        sub_n_worker = int(n_worker * 0.3)
        sub_worker = np.sort(np.random.choice(np.arange(n_worker), size=sub_n_worker, replace=False))
        tmp = np.zeros((sub_n_worker, 3))
        tmp[:, 0] = i
        tmp[:, 1] = sub_worker
        tmp[:, 2] = R[i, sub_worker]
        list.append(tmp)

    R_long = np.concatenate(list, axis=0)
    
    true_label = np.array([0] * (n_task // 3) + [1] * (n_task // 3) + [2] * (n_task // 3))

    return R_long, true_label

def logistic_reg(X, y, lambda1, centroid):
    
    # logistic regression for updating latent factors in binary crowdsourcing
    # penalty is to shrink beta to a given centroid
    if X.ndim < 2:
        X = np.expand_dims(X, axis=0)

    n, p = X.shape
    beta = np.zeros((p, ))
    y_hat = sigmoid(X.dot(beta)).squeeze()
    y_hat[y_hat < 1e-8] = 1e-8
    y_hat[y_hat > 1 - 1e-8] = 1 - 1e-8
    grad = X.T.dot(y_hat - y) + 2 * lambda1 * (beta - centroid)
    hessian = X.T.dot(np.diagflat(np.multiply(y_hat, 1 - y_hat))).dot(X) + 2 * lambda1 * np.eye(p)
    iter = 0
    while np.linalg.norm(grad) > 1e-5:
        beta = beta - np.linalg.inv(hessian).dot(grad)
        y_hat = sigmoid(X.dot(beta))
        y_hat[y_hat < 1e-8] = 1e-8
        y_hat[y_hat > 1 - 1e-8] = 1 - 1e-8
        grad = X.T.dot(y_hat - y) + 2 * lambda1 * (beta - centroid)
        hessian = X.T.dot(np.diagflat(np.multiply(y_hat, 1 - y_hat))).dot(X) + 2 * lambda1 * np.eye(p)
        iter = iter + 1
        if iter > 100:
            break
    return beta.T

def sigmoid(x):

    return np.divide(1, (1 + np.exp(-x)))


def label_swap(Grp_cur, Grp_prev):

    grp = np.unique(Grp_cur)

    perm_all = list(permutations(grp))

    rand_index_list = np.zeros((len(perm_all), ))

    for i in range(len(perm_all)):

        dic = {idx: perm_all[i][idx] for idx in grp}
        Grp_perm = [dic[Grp_cur[j]] for j in range(len(Grp_cur))]

        rand_index_list[i] = accuracy_score(Grp_prev, Grp_perm)

    ix = np.argmax(rand_index_list)
    dic = {idx: perm_all[ix][idx] for idx in grp}
    Grp_perm = [dic[Grp_cur[j]] for j in range(len(Grp_cur))]

    return np.array(Grp_perm)

def data_converter(data):

    n_task = len(np.unique(data[:, 0]))
    n_worker = len(np.unique(data[:, 1]))
    num_class = len(np.unique(data[:, 2]))

    data_tensor = np.zeros((n_task, n_worker, num_class))

    for row in data:

        data_tensor[int(row[0]), int(row[1]), int(row[2])] += 1

    return data_tensor


def multinomial_reg1(A_init, B, Y, O, lambd, centroid):

    N = B.shape[0]
    k = B.shape[1]
    C = O.shape[0]

    conc1 = np.zeros((N, C))
    conc2 = np.zeros((N, C, k))
    beta = A_init

    for n in range(N):

        for c in range(C):
            
            conc1[n, c] = np.dot(np.dot(beta.T, O[c, n, :, :]), B[n, :])
            conc2[n, c, :] = np.dot(O[c, n, :, :], B[n, :])

    conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]

    grad = np.sum(- conc2[np.arange(N), Y.astype('int'), :] + \
            np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
            2 * lambd * (beta - centroid)
    
    iter = 0
    while np.linalg.norm(grad) > 1e-5:

        beta = beta - 0.001 * grad
        for n in range(N):
    
            for c in range(C):
                
                conc1[n, c] = np.dot(np.dot(beta.T, O[c, n, :, :]), B[n, :])
                conc2[n, c, :] = np.dot(O[c, n, :, :], B[n, :])

        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
        grad = np.sum(- conc2[np.arange(N), Y.astype('int'), :] + \
                np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                2 * lambd * (beta - centroid)
        
        iter += 1

        if iter > 10:
            break

    return beta

def multinomial_reg2(B_init, A, Y, O, lambd, centroid):
    
    M = A.shape[0]
    k = A.shape[1]
    C = O.shape[0]

    conc1 = np.zeros((M, C))
    conc2 = np.zeros((M, C, k))
    beta = B_init

    for m in range(M):

        for c in range(C):
            
            conc1[m, c] = np.dot(np.dot(A[m, :], O[c, :, :]), beta)
            conc2[m, c, :] = np.dot(O[c, :, :].T, A[m, :])

    conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
    
    grad = np.sum(- conc2[np.arange(M), Y.astype('int'), :] + \
            np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
            2 * lambd * (beta - centroid)
    
    iter = 0
    while np.linalg.norm(grad) > 1e-5:

        beta = beta - 0.001 * grad
        
        for m in range(M):
    
            for c in range(C):
                
                conc1[m, c] = np.dot(np.dot(A[m, :], O[c, :, :]), beta)
                conc2[m, c, :] = np.dot(O[c, :, :].T, A[m, :])

        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
    
        grad = np.sum(- conc2[np.arange(M), Y.astype('int'), :] + \
                np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                2 * lambd * (beta - centroid)
        
        iter += 1
        if iter > 10:
            break

    return beta

def cayley_transform(A, B, Y, O):

    G = np.zeros(O.shape)
    S = np.zeros(O.shape)
    C = G.shape[0]
    prob = np.zeros(Y.shape)

    for c in range(C):

        prob[:, c] = np.exp(np.sum(A * (B.dot(O[c, :, :].T)), axis=1))
    
    prob = prob / np.sum(prob, axis=1)[:, np.newaxis]
    coef = prob - Y

    iter = 0
    err = 1

    while err > 1e-2:
    
        for c in range(1, C): # the first set of rotation matrices are fixed as identity matrix for reference group

            G[c, :, :] = B.T.dot(np.diagflat(coef[:, c])).dot(A)
            S[c, :, :] = G[c, :, :].dot(O[c, :, :].T) - O[c, :, :].dot(G[c, :, :].T)
            O[c, :, :] = np.linalg.inv(np.eye(O.shape[1]) + 0.00005 * S[c, :, :]).dot(np.eye(O.shape[1]) - 0.00005 * S[c, :, :]).dot(O[c, :, :])

        for c in range(C):
    
            prob[:, c] = np.sum(np.exp(A.dot(O[c, :, :]) * B), axis=1)

        prob = prob / np.sum(prob, axis=1)[:, np.newaxis]
        coef = prob - Y

        err = np.max([np.linalg.norm(G[c, :, :]) for c in range(C)])
        iter += 1

        if iter > 10:
            break

    return O

