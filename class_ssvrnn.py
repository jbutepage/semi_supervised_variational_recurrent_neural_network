import numpy as np
import tensorflow as tf
import os.path
from util_ssvrnn import run_rnn  
 
 
class SSVRNN:
    
    
    def __init__(self, flags ):
        
        print('Creating model')
        
        self.batch_size = flags['bs']
        self.tr_ba_le   = flags['tbl']
        self.input_size = flags['input_sz']
        self.dc         = flags['dc']
        self.state_size = flags['rnn_sz']
        self.hidden     = flags['hidden']
        self.alpha      = flags['alpha']
        self.w          = flags['w']
        self.b          = flags['b']
        self.num_epochs = flags['ne']
        self.sess       = flags['sess']
        self.lr         = flags['lr']
        self.mgn        = flags['mgn']
        self.lookahead  = flags['lookahead']
        self.n_layers   = flags['lstmlayers']
             
        # dropout variable
        self.keep_prob = tf.placeholder(tf.float32)
        # Gumbel parameter
        self.tau       = tf.placeholder(tf.float32)
        
        
        self.batchX_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.tr_ba_le , self.input_size])
        self.batchY_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.tr_ba_le, self.dc])
        self.batchI_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.tr_ba_le, self.dc])
        
        
        # encoder RNNs continuous and discrete
        self.ecs_h   = tf.placeholder(tf.float32, [self.n_layers, 2, self.batch_size, self.state_size])
        self.ehs_h = tf.unstack(self.ecs_h, axis=0)
        self.eis_h = tuple( [tf.nn.rnn_cell.LSTMStateTuple(self.ehs_h[idx][0], self.ehs_h[idx][1]) for idx in range(self.n_layers)]) 
        self.einc_h  = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, activation=tf.nn.tanh)
        self.esc_h   = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, activation=tf.nn.tanh)
        cells = [self.einc_h]
        for n in range(self.n_layers-1):
            cells.append(self.esc_h )
        self.ec_h    = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        self.ec_h    = tf.nn.rnn_cell.DropoutWrapper(self.ec_h, output_keep_prob=self.keep_prob)
        self.enc_rnn = [self.eis_h, self.ec_h]
        
        
        
        # Unpack inputs
        self.inputs_series = tf.split(self.batchX_placeholder, self.tr_ba_le , 1)
        self.labels_series = tf.unstack(self.batchY_placeholder, axis=1)
        self.bool_series   = tf.unstack(self.batchI_placeholder, axis=1)
        
        # run model
        self.loss, self.outputs,  self.eis_h, self.y_out, self.y_out_sample = run_rnn(self.inputs_series, \
            self.labels_series, self.bool_series, self.enc_rnn,  self.w, self.b,  self.dc, \
            self.keep_prob, self.tau ,  lookahead=self.lookahead, alpha=self.alpha, batch_size=self.batch_size, laten_dim=self.hidden)
        
        # optimizer
        self.optimizer = tf.train.AdagradOptimizer(self.lr) 
        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs =  [(tf.clip_by_value(grad, -self.mgn, self.mgn), var) for grad, var in gvs]
        self.train_step = self.optimizer.apply_gradients(capped_gvs)
        
        # Init
        self.sess.run(tf.initialize_all_variables()) 
        print('Created model')
        
        if 'name' in flags.keys():
            if os.path.exists(flags['name']+'.index'):
                self.load_model(flags['name'])
                print('Loaded model ' + flags['name'])
            else:
                print('File name does not exist! ' + flags['name'])
        
        
    def save_model(self, name):
        """Save model"""
        saver = tf.train.Saver()
        saver.save(self.sess, name)

    def load_model(self, name):
        """Load model"""
        saver = tf.train.Saver()
        saver.restore(self.sess, name)


        
    def train(self, x, y, n_epochs = -1, keep_prob = 1.0, tau = 0.1):
        """ Train model
            The tricky part is how to feed in the observed labels and the 
            inferred labels, when it is unobserved.
            We solve this by creating a second input that contains only zeros
            at a time point at which the label is not observed and only ones
            when it is observed.
            
            We can use this to sample from y and to compute the loss, e.g.:
                
                
            y = 1(y observed) * y_real + (1-1(y observed)) * y_inferred
            
        
        """
                
        if n_epochs == -1:
            n_epochs = self.num_epochs
        
        loss_list = []
        for epoch_idx in range(n_epochs):
             
            avg_loss = 0
            num_batches = x.shape[1] // self.tr_ba_le
            _current_ecs_h = np.zeros((self.n_layers, 2, self.batch_size, self.state_size))
    
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.tr_ba_le
                end_idx   = start_idx + self.tr_ba_le
         
                batchX = x[:,start_idx:end_idx,:]
                batchY = y[:,start_idx:end_idx,:]
                batchI = np.copy(batchY)
                batchI[:,np.sum(batchI,(2))[0] == self.dc] = 0
                batchI[:,np.sum(batchI,(2))[0] == 1]       = 1
                 
                feed = { self.batchX_placeholder: batchX,
                        self.batchY_placeholder: batchY,
                        self.batchI_placeholder: batchI,
                        self.ecs_h: _current_ecs_h,
                        self.keep_prob: keep_prob,
                        self.tau: tau}
                   
                _loss, _train_step, _ecs_h, _outputs, _y_prob = self.sess.run(
                    [self.loss,self. train_step, self.eis_h, self.outputs, self.y_out], feed_dict=feed)
                
                _loss = _loss                  
                _current_ecs_h   = _ecs_h
 
                avg_loss += _loss
            avg_loss /= self.batch_size
        
            avg_loss /= (self.tr_ba_le * num_batches)
            loss_list.append(avg_loss)
            

        return loss_list
            
    
    def evaluate(self, x, y, keep_prob = 1.0, tau = 0.1):
        """ Generate output:
            
            y_hat: inferred current classes py q
            o_hat: inferred continuous states
            avg_loss:loss
            y_hat_prior: inferred current classes py p
        
            The tricky part is how to feed in the observed labels and the 
            inferred labels, when it is unobserved.
            We solve this by creating a second input that contains only zeros
            at a time point at which the label is not observed and only ones
            when it is observed.
            
            We can use this to sample from y and to compute the loss, e.g.:
                
                
            y = 1(y observed) * y_real + (1-1(y observed)) * y_inferred
            
        
        """
        
        loss_list = []
        for epoch_idx in range(1):
             
            avg_loss = 0
            num_batches = x.shape[1]// self.tr_ba_le
            _current_ecs_h = np.zeros((self.n_layers, 2, self.batch_size, self.state_size))
        
        for batch_idx in range(num_batches):
                start_idx = batch_idx * self.tr_ba_le
                end_idx   = start_idx + self.tr_ba_le
         
                batchX = x[:,start_idx:end_idx,:]
                batchY = y[:,start_idx:end_idx,:]
                batchI = np.copy(batchY)
                batchI[:,np.sum(batchI,(2))[0] == self.dc] = 0
                batchI[:,np.sum(batchI,(2))[0] == 1]  = 1
                 
                feed = { self.batchX_placeholder: batchX,
                        self.batchY_placeholder: batchY,
                        self.batchI_placeholder: batchI,
                        self.ecs_h: _current_ecs_h,
                        self.keep_prob: keep_prob,
                        self.tau: tau}
                  
                _loss, _ecs_h, outputs_pred, y_prob, y_prob_prior  = self.sess.run(
                    [self.loss, self.eis_h, self.outputs, self.y_out, self.y_out_sample], feed_dict=feed)              
                _current_ecs_h  = _ecs_h
             
                if batch_idx == 0:
                    y_hat       = y_prob
                    y_hat_prior = y_prob_prior
                    o_hat       = outputs_pred
                else:
                    y_hat       = np.concatenate((y_hat,y_prob),0)  
                    y_hat_prior = np.concatenate((y_hat_prior,y_prob_prior),0)  
                    o_hat       = np.concatenate((o_hat,outputs_pred),0)  
                    
                avg_loss += _loss
        avg_loss /= self.batch_size
        avg_loss /= (self.tr_ba_le * num_batches)
        loss_list.append(avg_loss)
              
 
        # reshape correctly 
        y_hat       = np.array(y_hat) 
        y_hat_prior = np.array(y_hat_prior) 
        o_hat       = np.array(o_hat ) 
        y_hat       = y_hat.reshape((-1, y_hat.shape[-1]),order='F')
        y_hat       = y_hat.reshape((self.batch_size,  -1, y_hat.shape[-1]),order='C')
        y_hat_prior = y_hat_prior.reshape((-1, y_hat_prior.shape[-1]),order='F')
        y_hat_prior = y_hat_prior.reshape((self.batch_size,  -1, y_hat_prior.shape[-1]),order='C')
        o_hat       = o_hat.reshape((-1, o_hat.shape[-1]),order='F')
        o_hat       = o_hat.reshape((self.batch_size,  -1, o_hat.shape[-1]),order='C')
        
        return y_hat, o_hat, avg_loss, y_hat_prior
    
    
    