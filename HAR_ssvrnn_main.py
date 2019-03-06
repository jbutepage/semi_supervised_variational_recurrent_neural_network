"""
     Script to train SSVRNN 
     Contact: Judith Butepage
"""
from __future__ import division
import numpy as np
import tensorflow as tf
from util_weights import create_ssvrnn
from har_data_util import read_data, cover_y
from class_ssvrnn import SSVRNN
tf.reset_default_graph()


 
def get_data(data_path):
 
    x_train, y_train_orig, train_data = read_data(data_path, 'train')
    x_test, y_test, test_data         = read_data(data_path, 'test')
    
    # make test size as large as training size 
    x_test      = np.concatenate((x_test,x_test,x_test),0)[:21]
    y_test      = np.concatenate((y_test,y_test,y_test),0)[:21]
    
    
    return x_train, y_train_orig, x_test, y_test
    
def model_setup(flags):
    
    tf.reset_default_graph()
    
    # create weights
    w, b = create_ssvrnn(flags) 
    flags['w'] = w
    flags['b'] = b
    
    # create session
    sess = tf.Session()
    flags['sess'] = sess  
    
    model = SSVRNN(flags)
    
    return model
    

def main(flags, data_path):
    
    x_train, y_train_orig, x_test, y_test = get_data(data_path)
    print('loaded data')
    
    model  = model_setup(flags)
    
    loss = 0
    print('start training')
    for i in range(0, flags['ne']):
        y_train   = cover_y(y_train_orig, flags['unobs_%'])
        x_train_t = x_train + np.random.normal(0,0.05, x_train.shape)
        loss_list = model.train(x_train_t, y_train, keep_prob=flags['drop_%'], n_epochs=1, tau = flags['tau'])
        loss += loss_list[-1]
        
        if np.mod(i,flags['print_epoch']) == 0 and i!=0:
            print("Mean loss, epoch {0}   ".format(i) + "-log p(x) <= {:0.3f}".format(loss ))
            loss = 0.
        if np.mod(i,flags['save_epoch']) == 0 and i!=0:
            print("saving model in epoch {0}".format(i))
            model.save_model(model_path + 'model_' +  str(i) + '.ckpt')
    model.save_model(model_path + 'model_' +  str(i) + '.ckpt')
    return model, x_train, y_train_orig, x_test, y_test 

if __name__ == "__main__":
    model_path = "/models/"
    data_path  = "/UCIHAR/"
            
    path    = "./data4/"
    flags = {} 
    # data size
    flags['input_sz']  = 561    
    flags['dc']        = 6     # number of classes
    flags['split']     = 561   # what of prediction is input
    # training
    flags['bs']        = 21    # batchsize
    flags['tbl']       = 10    # truncated backprop 
    flags['lookahead'] = 5     # after how many steps should prediction be input (aka burn in)
    flags['lr']        = 0.001 # learning rate
    flags['ne']        = 200    # number of epochs
    flags['save_epoch']= 50
    flags['print_epoch']= 50
    flags['mgn']       = 2.    # gradient magnitude cut
    flags['alpha']     = flags['input_sz']     # SSVAE weight
    flags['unobs_%']   = 0.1   # how many labels are unobserved in each epoch
    flags['tau']       = 0.1   # Gumbel parameter
    flags['drop_%']    = 0.8
    flags['data_scale']= 100.
    # model size
    flags['rnn_sz']    = 200   # size of rnn hidden state  
    flags['hidden']    = 100    # size of latent variable z
    flags['embedding'] = 250   # size of input embedding
    flags['lstmlayers']= 3     # how many lstm layers

    # uncomment when you want to load somethin
    #flags['name']      = model_path_name  # only when loading
    model, x_train, y_train_orig, x_test, y_test  = main(flags, data_path)
            
    y_test_covered = cover_y(y_test,1.0)
    y_hat, o_hat, avg_loss, y_hat2 = model.evaluate( x_train, y_test_covered)
    y_real = np.argmax(y_train_orig[:,:y_hat.shape[1],:],2)
    y_pred = np.argmax(y_hat,2)
    acc_train = np.sum(y_real == y_pred) / np.float(y_pred.shape[0]*y_pred.shape[1])
    
    y_hat, o_hat, avg_loss, y_hat2 = model.evaluate( x_test, y_test_covered)
    y_real = np.argmax(y_test[:9,:y_hat.shape[1],:],2)
    y_pred = np.argmax(y_hat[:9],2)
    acc_test = np.sum(y_real == y_pred) / np.float(y_pred.shape[0]*y_pred.shape[1])
    
    print('Classification accuracy: train: {0}, test: {1}'.format(acc_train, acc_test) )


    model.save_model(model_path + 'model.ckpt')
    
