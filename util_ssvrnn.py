import tensorflow as tf
from util_functions_ssvrnn import encode_rnn, encode, comp_loss 
from util_functions_ssvrnn import normal_params, sample_normal,  make_class_pred    
from util_functions_ssvrnn import decode_nn 


def run_rnn(x_in,  y_in, i_y, enc_rnn, w, b,  n_class, keep_prob, tau,   
            lookahead=100000, alpha=1., batch_size=1, laten_dim=10):    
    """
        inputs: 
            x_in, y_in = continuous input and labels
            i_y        = indices at which time steps the label was observed
            enc_rnn    = rnn part
            w, b       = weights of [x encoder (generating mu and sigma), 
                               x decoder (generating prediction), 
                               y decoder (classifier),
                               x prior (generating prior mu and sigma)
                               y prior (generating prior for classifier)]
            n_class     = number of label classes
            keep_prob   = dropout percentage parameter
            tau         = Gumbel parameter
            lookahead   = if time step is larger than lookahead, we use our predictions to predict
            alpha       = how much to wait labeled classifications
            batch_size  = how big is the batch
            latent_dim  = dimension of z
            
    
    """
    
    if lookahead < 1:
        raise RuntimeError('The model needs at least one time step fed as input.')
        
    # unpack
    we, wd, wy, wp, wyp = w 
    be, bd, by, bp, byp = b 
    del w
    del b
    eis_h, ec_h = enc_rnn # rnn states

    outputs      = []  # continuous predictions
    y_prob       = []  # discrete classifications of current time step
    y_prob_prior = []  # discrete classifications of current time step

    loss = 0
   
     
    for time_st in range(len(x_in) ):
        
        # if time step is larger than lookahead, we use our predictions to predict.                
        if   time_st<=lookahead:
            print('o')
            x_curr_all = x_in[time_st][:,0,:]
            weight = 1.
        elif time_st>lookahead:
            x_curr_all = outputs[-1] 
            weight = 5.
            print('n')        
         
        
        #x_curr_all = (x_curr_all - mean) / std

        xt  =  encode(x_curr_all , we, be, keep_prob)
                
        # old hidden state
        ht = eis_h[-1][1] 
        
        # make classification q(y|X, h)
        y_sample,  y_logits , y_logits_with_sm, log_q_y = make_class_pred(tf.concat([ht, xt], 1), \
                            y_in[time_st], i_y[time_st], wy, by, batch_size, tau, n_class)
        y_prob.append(y_logits_with_sm)
        # make classification with prior p(y|h)
        y_sample_prior, y_logits_prior, y_logits_with_sm_prior, log_p_y_prior = make_class_pred(tf.concat([ht], 1), \
                            y_in[time_st], i_y[time_st], wyp, byp, batch_size, tau, n_class)
        y_prob_prior.append(y_logits_with_sm_prior)
        
        # compute approx. posterior q(z|X, h) ~ N(m,v)
        m , v  = normal_params(tf.concat([xt, ht, y_sample], axis = 1), we[1:] , be[1:] , keep_prob)    
        # compute prior p(z|h) ~ N(mp,vp)
        mp, vp = normal_params(tf.concat([ht, y_sample_prior], axis = 1), wp, bp, keep_prob)
        # sample from approx. posterior q(z|X, h) ~ N(m,v)
        s      = sample_normal(m,v,D=batch_size, laten_dim=laten_dim)
        s      = tf.reshape(s,(batch_size,-1))
        # decode
        sht = tf.concat([s, ht, y_sample] , axis = 1)
        decoded =  decode_nn(sht, wd, bd, keep_prob) 
        # we make use of the residual prediction proposed by 
        # On Human Motion Prediction Using Recurrent Neural Networks - Martinez et al
        decoded = decoded + x_curr_all 
        decoded = decoded 
        outputs.append(decoded )
        
        x = x_in[time_st] 
         
        
        # compute loss
        y_true = tf.cast(y_in[time_st],tf.float32)
        loss += weight*comp_loss(x , decoded, y_logits_with_sm, y_logits_with_sm_prior,\
                                  log_q_y, log_p_y_prior, y_true, i_y[time_st], m, v, mp, vp, alpha, n_class)
        
        # update RNN
        xtc = tf.concat([x_curr_all, sht, y_sample], axis = 1)
        ess_h, eis_h      = encode_rnn([xtc],  eis_h, ec_h, we, be, keep_prob, lstm='lstm_1_0')
    
     
 
    return  loss, outputs, eis_h, y_prob, y_prob_prior

