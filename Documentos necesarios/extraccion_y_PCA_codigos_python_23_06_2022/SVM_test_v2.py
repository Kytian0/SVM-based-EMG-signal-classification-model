import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs, make_circles
from matplotlib.colors import ListedColormap
from SVM import SVM
from SVM_aprox import SVM_aprox
from SVM_mod_M import SVM_mod_M
from scipy.optimize import minimize
from SVM_RNS_v2 import SVM_RNS_v2
from export_data_files import export_data_files
import pandas as pd
import time


start_time = time.time()

# Original features for training
X, y = make_circles(25, factor=.1, noise=.1)
X_max = np.amax(np.absolute(X))
X = ((X / X_max) + 1) / 2

print("original X", X)
print("original X shape", X.shape)
print("original y", y)
print("original y shape", y.shape)

def read_pickle_data():
    x_train = read_output_data("Xtrain.pkl")
    y_train = read_output_data("ytrain.pkl")
    x_test = read_output_data("Xtest.pkl")
    y_test = read_output_data("ytest.pkl")

    return x_train, y_train, x_test, y_test
def read_output_data(name):
    return pd.read_pickle(name)

def test_plot(X, y, svm_model, axes, title):
    plt.axes(axes)
    xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    ylim = [np.min(X[:, 1]), np.max(X[:, 1])]
    xx, yy = np.meshgrid(np.linspace(*xlim, num=700), np.linspace(*ylim, num=700))
    rgb = np.array([[210, 0, 0], [0, 0, 150]]) / 255.0

    svm_model.fit(X, y)
    z_model = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    print("valores del hiperplano: ", z_model, "\n")
    print("maximo valor del hiperplano", np.amax(z_model), "mínimo valor del hiperplano",  np.amin(z_model), "\n")

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
    plt.title(title)


def test_plot_aprox(X, y, svm_model_in, axes, axes_2, axes_3, title):
    #Normalizacion de los datos
    #Todo: realizar la normalización antes del entrenamiento de la SVM
    # X_max = np.amax(np.absolute(X_raw))
    # X = ((X_raw/X_max) + 1)/2
    #Definición del meshgrid
    plt.axes(axes)
    xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    ylim = [np.min(X[:, 1]), np.max(X[:, 1])]
    # xx, yy = np.meshgrid(np.linspace(*xlim, num=700), np.linspace(*ylim, num=700))
    xx, yy = np.meshgrid(np.linspace(*xlim, num=350), np.linspace(*ylim, num=350))
    rgb = np.array([[210, 0, 0], [0, 0, 150]]) / 255.0
    #RBF using Taylor Series
    svm_model_in.fit(X, y)
    aprox_model = SVM_aprox(svm_model_in, inner_product_type='rbf_inner', aprox_order=8)
    z_axis_model = aprox_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    print("vectors to classify: ", np.c_[xx.ravel(), yy.ravel()],"\n", "... and its shape",(np.c_[xx.ravel(), yy.ravel()]).shape)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.contour(xx, yy, z_axis_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_axis_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb),
                 zorder=1)
    plt.title(title)

    # RBF using scaled SVM in mod M
    plt.axes(axes_2)
    plt.title("aprox SVM with integers modulo M")
    k = X.shape[1]
    lambdas = svm_model_in.lambdas
    labels = svm_model_in.y
    ##buscar muestras aleatorias del plano de decision para 0<x<1 y 0<y<1 y encontrar un minimo forzosamente
    random_precision = 1000
    number_of_random_samples_x = 30*(xx.shape[0])   #orig

    random_X = (np.random.randint(low=1, high=random_precision, size= (number_of_random_samples_x,k)))/random_precision
    # plane_samples = svm_model_in.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plane_samples = svm_model_in.decision_function(random_X)
    bias = svm_model_in.b

    # gain_plane = 60    # Original
    # gain_lambdas = 8   # Original
    beta = 5
    alpha = 4
    mod_set = np.array([2 ** beta - 1, 2 ** alpha, 2 ** beta + 1])
    M = np.prod(mod_set)
    print("Dynamic range:", M, "\n")
    gain_plane = np.floor(0.8*((M-1)/(2*k))) # Aprox. "amplitude" of hyperplane -> 80% of positive dynamic range for k features
    gain_lambdas = np.floor(0.01*gain_plane) # Amplitude of lambdas -> 1% of plane's gain
    svm_mod_M = SVM_mod_M(svm_ap=svm_model_in,alpha=alpha, beta=beta,gamma=1, feature_dimensions=k,lambdas=lambdas,
                          labels=labels,plane_samples=plane_samples,bias=bias,gain_plane=gain_plane,gain_lambdas=gain_lambdas)
    svm_mod_M.verify_gains()
    print("meshgrid [xx,yy] para el plano de decision mod M:", np.c_[xx.ravel(), yy.ravel()],"\n")
    z_axis_M = svm_mod_M.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    print("valores de la exponencial original:", svm_model_in.kernel((np.c_[xx.ravel(), yy.ravel()]),svm_model_in.X))
    print("valores del hiperplano mod M:", z_axis_M, "\n")
    print("mod M plane", z_axis_M, "max and min:", np.amax(z_axis_M),
          np.amin(z_axis_M))


    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.contour(xx, yy, z_axis_M, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_axis_M.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb),
                 zorder=1)


    #ToDo: utilizar la SVM con rns (version 2) basandose en lo utilizado previamente en SVM_test
    # RBF represented with RNS using LUT
    exp_lut_rns = svm_mod_M.exp_rns
    svm_rns = SVM_RNS_v2(exp_rns_lut=exp_lut_rns,svm_model_in=svm_model_in,lambdas_mod_M=svm_mod_M.normal_lambdas,
                         bias_mod_M=svm_mod_M.normal_bias,labels_mod_M=svm_mod_M.normal_labels,alpha=alpha,
                         beta=beta,feature_dimensions=k)
    plt.axes(axes_3)
    plt.title("aprox SVM with RNS")
    clasification = svm_rns.get_hyperplane_rns(np.c_[xx.ravel(), yy.ravel()])
    plt.contourf(xx, yy, clasification.reshape(xx.shape), alpha=0.3, levels=2, cmap=ListedColormap(rgb),
                 zorder=1)

    # Export data
    data_out = export_data_files(svm_rns,"kernel_lut.data", "params_lut.data", "bias_lut.data")

X_tr, y_tr, x_te,y_te = read_pickle_data()
print("labels", y_tr)
X_tr = X_tr.to_numpy()
y_tr = y_tr.to_numpy()
print("labels (to numpy)", y_tr)
print("labels (to numpy) shape", y_tr.shape)
print("X (to numpy)", X_tr)
print("X (to numpy) shape", X_tr.shape)
X, x_te_red, y, y_te_red = train_test_split(X_tr, y_tr, train_size=0.03, test_size=0.03)
y = y.reshape(y.shape[0])
print("labels (to numpy) reduced", y)
print("labels (to numpy) reduced shape", y.shape)
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 4))
C = 10
gamma = 1
test_plot(X, y, SVM(kernel='rbf', C=C, max_iter=60, degree=3, gamma=gamma), axs[0,0], 'RBF using SMO lite')
test_plot(X, y, SVC(kernel='rbf', C=C, degree=3, gamma=gamma), axs[0,1], 'sklearn.svm.SVC')
test_plot_aprox(X, y, SVM(kernel='rbf', C=C, max_iter=60, degree=3, gamma=gamma), axs[0,2], axs[0,3],axs[1,0], 'Aprox. model of RBF')

print("--- Elapsed time: %s seconds" % (time.time() - start_time))
plt.show()
