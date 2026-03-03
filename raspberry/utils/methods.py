from sklearn.decomposition import PCA
import numpy as np

def PcaApplication(signal, n_componentes):

    pca = PCA(n_components=n_componentes)  
    pca.fit(signal)
    transformed_feature = pca.transform(signal)
    
    return transformed_feature

def MeanApplication(signal):
    
    transformed_feature = np.mean(signal, axis=1)
    transformed_feature = np.array(transformed_feature).reshape(-1, 1)
    
    return transformed_feature