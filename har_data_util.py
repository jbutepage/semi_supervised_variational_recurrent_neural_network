import numpy as np


def cover_y(y, perc):
    y = np.copy(y)
    B,T,C = y.shape
    for b in range(B):
        p = np.random.permutation(T)[:int(T*perc)]
        y[b,p] = 1
    return y

def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]




def read_data(data_path, data):
    
    f = open(data_path +  data + "/X_" + data + ".txt", "r")
    x_lines = list(f)
    f.close()
    f = open(data_path +  data + "/y_" + data + ".txt", "r")
    y_lines = list(f)
    f.close()
    f = open(data_path +  data + "/subject_" + data + ".txt", "r")
    s_lines = list(f)
    f.close()
    x_data = []
    y_data = []
    s_data = []
    for l in range(len(x_lines)):
        x_data.append( list(map(float,x_lines[l].split())))
        y_data.append( list(map(int,y_lines[l].split())))
        s_data.append( list(map(int,s_lines[l].split())))
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    s_data = np.array(s_data)
    
    N_labels = len(np.unique(y_data))
    
    min_rec_len = 100000
    data_out = {}
    for s in np.unique(s_data):
        times = (s_data == s)[:,0]
        data_out['x_' + data + '_{0}'.format(s)] = x_data[times]
        data_out['y_' + data + '_{0}'.format(s)] = indices_to_one_hot(y_data[times] - 1, N_labels)
        if min_rec_len > x_data[times].shape[0]:
            min_rec_len = x_data[times].shape[0]
    x_data = []
    y_data = []
    for s in np.unique(s_data):
        data_out['x_' + data + '_{0}'.format(s)] = data_out['x_' + data + '_{0}'.format(s)][:min_rec_len,:]
        data_out['y_' + data + '_{0}'.format(s)] = data_out['y_' + data + '_{0}'.format(s)][:min_rec_len,:]
        x_data.append(data_out['x_' + data + '_{0}'.format(s)])
        y_data.append(data_out['y_' + data + '_{0}'.format(s)])
        
        
    return np.array(x_data), np.array(y_data), data_out

