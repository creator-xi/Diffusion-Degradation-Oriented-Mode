import numpy as np
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import os
import torch
import pdb
from sklearn import metrics


feature_save_path = 'feature_vision/'
if not os.path.exists(feature_save_path):
    os.makedirs(feature_save_path)


exp_model = 'sr4_inpainting_csbb10'
iteration = 't500'
layer = 'adap_rb10'

test_model = '{}_{}_{}'.format(iteration, layer, exp_model)
print(test_model)



# a, sr4
target_directory = "feature_vision/adap_t500_rb10/sr4/"
file_paths = []
for root, dirs, files in os.walk(target_directory):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)

a_file_names = file_paths
arrays = []
for file_name in a_file_names:
    loaded_array = np.load(file_name)
    arrays.append(loaded_array)
a = np.concatenate(arrays, axis=0)

# b, inpainting
target_directory = "feature_vision/adap_t500_rb10/inpainting/"
file_paths = []
for root, dirs, files in os.walk(target_directory):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
        
b_file_names = file_paths
arrays = []
for file_name in b_file_names:
    loaded_array = np.load(file_name)
    arrays.append(loaded_array)
b = np.concatenate(arrays, axis=0)

# c, csbb10
target_directory = "feature_vision/adap_t500_rb10/csbb10/"
file_paths = []
for root, dirs, files in os.walk(target_directory):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
        
c_file_names = file_paths
arrays = []
for file_name in c_file_names:
    loaded_array = np.load(file_name)
    arrays.append(loaded_array)
c = np.concatenate(arrays, axis=0)


a_name = 'SR4'
b_name = 'Inpainting'
c_name = 'CS10'


X = np.concatenate([a,b,c],axis=0)
X = X.reshape(24,8,-1)
X = X.reshape(192,-1)


batch = X.shape[0]
b1 = batch // 3
b2 = b1 * 2
b3 = batch


new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

if not os.path.exists(os.path.join(feature_save_path, 'X-tSNE-'+test_model+'.npy')):
    print('Prepare t-SNE.')
    pca = PCA(n_components=50, random_state=1)
    X_tSNE = pca.fit_transform(X)
    # use test environment, scikit-learn 1.4.2

    tsne = TSNE(n_components=2, random_state=1)

    print('Learning t-SNE......')
    start_time = time.time()
    X_tSNE = tsne.fit_transform(X_tSNE)
    end_time = time.time()
    print('Finish t-SNE learning. Total duration time: %.4f s'%(end_time-start_time))
    
    print('Saving npy.')
    np.save(os.path.join(feature_save_path, 'X-tSNE-'+test_model+'.npy'), X_tSNE)

else:
    print('Exist! Loading {} ...'.format(os.path.join(feature_save_path, 'X-tSNE-'+test_model+'.npy')))   
    X_tSNE = np.load(os.path.join(feature_save_path, 'X-tSNE-'+test_model+'.npy'))
    
fig, ax = plt.subplots()
#plt.axis('off')

plt.scatter(X_tSNE[0:b1,0], X_tSNE[0:b1,1], label=a_name, color=new_colors[0], alpha=0.7, s=50)
plt.scatter(X_tSNE[b1:b2,0], X_tSNE[b1:b2,1], label=b_name, color=new_colors[1], alpha=0.7, s=50)
plt.scatter(X_tSNE[b2:b3,0], X_tSNE[b2:b3,1], label=c_name, color=new_colors[2], alpha=0.7, s=50)

plt.legend(fontsize='xx-large', loc='best')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0.0, right = 1, left = 0.0, hspace = 0.0, wspace = 0.0)
plt.margins(0.07,0.07)
plt.savefig(os.path.join(feature_save_path, 'X-tSNE-'+test_model+'.pdf'))

plt.show()


X_tSNE_class0 = X_tSNE[0:b1, :]
X_tSNE_class1 = X_tSNE[b1:b2, :]
X_tSNE_class2 = X_tSNE[b2:b3, :]



X_tSNE_class_list = []
X_tSNE_class_list.extend((X_tSNE_class0,X_tSNE_class1,X_tSNE_class2))

X_tSNE_class_W = 0
X_tSNE_class_B = 0

# Within-cluster dispersion
for i in range(len(X_tSNE_class_list)):
    W = np.trace(np.dot((X_tSNE_class_list[i] - X_tSNE_class_list[i].mean(0)).T, (X_tSNE_class_list[i] - X_tSNE_class_list[i].mean(0))))
    X_tSNE_class_W += W

print('Within-cluster dispersion: {:.2f}'.format(X_tSNE_class_W))


# Between-cluster dispersion
X_tSNE_class_center_list = []
for i in range(len(X_tSNE_class_list)):
    X_tSNE_class_center_list.append(X_tSNE_class_list[i].mean(0))

X_tSNE_class_center_all = X_tSNE.mean(0)
for i in range(len(X_tSNE_class_list)):
    X_tSNE_class_B += X_tSNE_class_list[i].shape[0]*np.sum((X_tSNE_class_center_list[i]-X_tSNE_class_center_all)**2)
    
print('Between-cluster dispersion: {:.2f}'.format(X_tSNE_class_B))


# Calinski-Harabaz Index
CHI = (X_tSNE_class_B*(X_tSNE.shape[0]-len(X_tSNE_class_list))) / (X_tSNE_class_W*(len(X_tSNE_class_list)-1))
print('Calinski-Harabaz Index: {:.2f}'.format(CHI))


labels = np.zeros([X_tSNE.shape[0]])
labels[0:b1] = 0
labels[b1:b2] = 1
labels[b2:b3] = 2



CHI = metrics.calinski_harabasz_score(X_tSNE, labels)  
print('Calinski-Harabaz Index: {:.2f}'.format(CHI))
