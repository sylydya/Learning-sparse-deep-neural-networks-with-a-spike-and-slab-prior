import numpy as np
import pandas as pd
import os
import errno

def my_relu(x):
    return x*(x>0)


a = 1
b = 1


#-----------------Regression Data------------------------_#


for my_seed in range(1,11):
    np.random.seed(my_seed)
    TotalP = 2000
    print('p = ', TotalP)
    NTrain = 10000
    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    sigma = 1.0
    for i in range(NTrain):
        if i%1000 == 0:
            print("x_train generate = ", i)
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            x_train[i, j] = (a*ee + b*np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a*a+b*b)
        x0 = x_train[i, 0]
        x1 = x_train[i, 1]
        x2 = x_train[i, 2]
        x3 = x_train[i, 3]
        x4 = x_train[i, 4]

        y_train[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

    Nval = 1000
    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    sigma = 1.0
    for i in range(Nval):
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            x_val[i, j] = (a*ee + b*np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a*a+b*b)
        x0 = x_val[i, 0]
        x1 = x_val[i, 1]
        x2 = x_val[i, 2]
        x3 = x_val[i, 3]
        x4 = x_val[i, 4]

        y_val[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

    NTest = 1000
    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    for i in range(NTest):
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            x_test[i, j] = (a*ee + b*np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a*a+b*b)
        x0 = x_test[i, 0]
        x1 = x_test[i, 1]
        x2 = x_test[i, 2]
        x3 = x_test[i, 3]
        x4 = x_test[i, 4]

        y_test[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train)

    x_val_df = pd.DataFrame(x_val)
    y_val_df = pd.DataFrame(y_val)

    x_test_df = pd.DataFrame(x_test)
    y_test_df = pd.DataFrame(y_test)

    PATH = './data/regression/' + str(my_seed) + "/"
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    print("write train")

    x_train_df.to_csv(PATH + "x_train.csv")
    y_train_df.to_csv(PATH + "y_train.csv")
    print("write val")

    x_val_df.to_csv(PATH + "x_val.csv")
    y_val_df.to_csv(PATH + "y_val.csv")

    print('write test')

    x_test_df.to_csv(PATH + "x_test.csv")
    y_test_df.to_csv(PATH + "y_test.csv")



#--------------------Structure Selection Data-----------------------------#

for my_seed in range(1,11):
    np.random.seed(my_seed)
    TotalP = 1000
    print('p = ', TotalP)
    NTrain = 10000
    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    sigma = 0.5
    for i in range(NTrain):
        if i%1000 == 0:
            print("x_train generate = ", i)
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            x_train[i, j] = (a*ee + b*np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a*a+b*b)
        x0 = x_train[i, 0]
        x1 = x_train[i, 1]
        x2 = x_train[i, 2]
        x3 = x_train[i, 3]
        x4 = x_train[i, 4]


        y_train[i, 0] = np.tanh(2 * np.tanh(2 * x0 - x1)) + 2*np.tanh(
            2* np.tanh(x2 - 2 * x3 - 2*x4 )) + np.random.normal(0, 1)



    Nval = 1000
    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    sigma = 1.0
    for i in range(Nval):
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            x_val[i, j] = (a*ee + b*np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a*a+b*b)
        x0 = x_val[i, 0]
        x1 = x_val[i, 1]
        x2 = x_val[i, 2]
        x3 = x_val[i, 3]
        x4 = x_val[i, 4]


        y_val[i, 0] = np.tanh(2 * np.tanh(2 * x0 - x1)) + 2*np.tanh(
            2* np.tanh(x2 - 2 * x3 - 2*x4 )) + np.random.normal(0, 1)


    NTest = 1000
    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    for i in range(NTest):
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            x_test[i, j] = (a*ee + b*np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a*a+b*b)
        x0 = x_test[i, 0]
        x1 = x_test[i, 1]
        x2 = x_test[i, 2]
        x3 = x_test[i, 3]
        x4 = x_test[i, 4]


        y_test[i, 0] = np.tanh(2 * np.tanh(2 * x0 - x1)) + 2*np.tanh(
            2* np.tanh(x2 - 2 * x3 - 2*x4 )) + np.random.normal(0, 1)




    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train)

    x_val_df = pd.DataFrame(x_val)
    y_val_df = pd.DataFrame(y_val)

    x_test_df = pd.DataFrame(x_test)
    y_test_df = pd.DataFrame(y_test)


    PATH = './data/structure/' + str(my_seed) + "/"
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    print("write train")

    x_train_df.to_csv(PATH +  "/x_train.csv")
    y_train_df.to_csv(PATH + "/y_train.csv")

    print("write val")

    x_val_df.to_csv(PATH + "/x_val.csv")
    y_val_df.to_csv(PATH + "/y_val.csv")

    print('write test')

    x_test_df.to_csv(PATH + "/x_test.csv")
    y_test_df.to_csv(PATH + "/y_test.csv")
