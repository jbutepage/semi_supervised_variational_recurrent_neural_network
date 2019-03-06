import numpy as np, tensorflow as tf

eps = 1e-08



"""
Training helpers
"""
def generator(array, batch_size):
    """Generate batch with respect to array's first axis."""
    start = 0
    while True:
        stop = start + batch_size
        diff = stop - array.shape[0]
        if diff <= 0:
            batch = array[start:stop]
            start += batch_size
        else:
            batch = np.concatenate((array[start:], array[:diff]))
            start = diff
        batch = batch.astype(np.float32)
        yield batch


"""
Loss helpers
"""

def kl_comp(loc, scale):
    """KL divergence between N(loc,scale) and N(0,1)"""
    kl = -0.5 * tf.reduce_sum(tf.square(loc) + tf.square(scale) - tf.log(eps + tf.square(scale)) - 1)
    return kl

def kl_comp_nosum(loc, scale):
    """KL divergence between N(loc,scale) and N(0,1), sum only ofer latent dim"""
    kl = -0.5 * tf.reduce_sum(tf.square(loc) + tf.square(scale) - tf.log(eps + tf.square(scale)) - 1, axis=1)
    return kl

def kl_comp2_nosum(loc, scale, loc2, scale2):
    """KL divergence between N(loc,scale) and N(loc2,scale2), sum only ofer latent dim"""
    kl = (tf.square(loc - loc2) + tf.square(scale)) / (2.0 * tf.square(scale2))
    kl = kl - tf.log(eps + tf.square(scale)) + tf.log(eps + tf.square(scale2))
    kl = -0.5 * tf.reduce_sum(kl - 1, axis=1)
    return kl

def log_lik(x, x_pred, ltype='mse'):
    """log likelihood, binomial or mse (fake gaussian)"""
    if ltype is not 'mse':
        log_like = -tf.reduce_sum(x * tf.log(x_pred + eps) + (1.0 - x) * tf.log(1.0 - x_pred + eps) )
    else:
        log_like = tf.reduce_sum(tf.squared_difference(x_pred,x)  )
    return log_like

def log_lik_nosum(x, x_pred, ltype='mse'):
    if ltype is not 'mse':
        log_like = -tf.reduce_sum((x * tf.log(x_pred + eps) + (1.0 - x) * tf.log(1.0 - x_pred + eps)), axis=1)
    else:
        log_like =  tf.reduce_sum(tf.squared_difference(x_pred, x), axis=1)
    return log_like

def comp_loss(xcurr, decoded, y_logits_with_sm_, y_logits_with_sm_prior, log_q_y, log_p_y_prior, y_true, i_y, m, v, mp, vp, alpha, n_class):
    """computes log likelihood - kl + class loss"""
    kl_z = tf.reshape(kl_comp2_nosum(m, v, mp, vp), [-1, 1])
    ll   = log_lik_nosum(xcurr, decoded)
    loss = tf.reduce_mean(ll - kl_z)
    loss += comp_class_loss(y_logits_with_sm_, y_logits_with_sm_prior, log_q_y, log_p_y_prior, y_true, i_y, alpha, n_class)
    return loss

def comp_class_loss(y_logits_with_sm_, y_logits_with_sm_prior, log_q_y, log_p_y_prior, y_true, i_y, alpha, n_class):
    """computes  class loss"""
    # crossentropy between q(y) and prior p(y)
    loss = tf.reduce_mean(tf.reduce_sum(tf.reshape((1.0 - tf.cast(i_y, tf.float32)) * (y_logits_with_sm_ * (log_q_y - log_p_y_prior)), [-1, n_class]), 1))
    # loss between observed and inferred classes of prior
    class_loss = y_true * tf.log(eps + y_logits_with_sm_prior)
    class_loss = -tf.reduce_mean(tf.reduce_sum(tf.cast(i_y, tf.float32) * class_loss, 1))
    loss = loss + alpha * class_loss
    # loss between observed and inferred classes of q    
    class_loss = y_true * tf.log(eps + y_logits_with_sm_)
    class_loss = -tf.reduce_mean(tf.reduce_sum(tf.cast(i_y, tf.float32) * class_loss, 1))
    loss = loss + alpha * class_loss
    return loss

"""
Classifier helpers
"""

def eval_class(x_in, w, b):
    """Classifies a single data point x_in"""
    hidden = tf.nn.relu(tf.matmul(x_in, w['wy1']) + b['by1'])
    y = tf.nn.softmax(tf.matmul(hidden, w['wy1a']) + b['by1a'])
    return y

def eval_class_list(x_in, w, b):
    """Classifies a several data points x_in"""
    y = []
    for i in range(x_in.shape[1].value):
        hidden = tf.nn.relu(tf.matmul(x_in[:, i, :], w['wy1']) + b['by1'])
        y.append(tf.nn.softmax(tf.matmul(hidden, w['wy1a']) + b['by1a']))
    return y

def encode_rnn_class(x_in, enc_init_state, enc_cell, w, b, keep_prob, lstm='lstm1'):
    """Encoder RNN for classifier"""
    with tf.variable_scope(lstm):
        enc_states_series, enc_current_state = tf.contrib.rnn.static_rnn(enc_cell, x_in, initial_state=enc_init_state)
    return (
     enc_states_series, enc_current_state)

"""
Encoder helpers
"""

def encode(x_in, w, b, keep_prob):
    """Encoder / Embedding for continuous states"""
    encoded = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w[0]) + b[0]), keep_prob)
    return encoded

def encode_rnn(x_in, enc_init_state, enc_cell, w, b, keep_prob, lstm='lstm1'):
    """Encoder RNN for continuous states"""    
    with tf.variable_scope(lstm):
        enc_states_series, enc_current_state = tf.contrib.rnn.static_rnn(enc_cell, x_in, enc_init_state)
    
    return (enc_states_series, enc_current_state)

def normal_params(x, w, b, keep_prob):
    """Encodes input with MLP into normal parameters mean and sigmoid variance"""
    encoded = tf.nn.dropout(tf.nn.relu(tf.matmul(x, w[0]) + b[0]), keep_prob)
    for i in range(1, len(w) - 2):
        encoded = tf.nn.dropout(tf.nn.relu(tf.matmul(encoded, w[i]) + b[i]), keep_prob)

    m = tf.matmul(encoded, w[-2]) + b[-2]
    v = 1.0 / (1.0 + tf.exp(-tf.matmul(encoded, w[-1]) + b[-1]))
    return (m, v)

"""
Decoder helpers
"""

def decode_rnn(z, dec_init_state, dec_cell, lstm='lstm1'):
    """Recoder RNN for continuous states"""
    with tf.variable_scope(lstm):
        dec_states_series, dec_current_state = tf.contrib.rnn.static_rnn(dec_cell, z, dec_init_state)
    return (dec_states_series, dec_current_state)


def decode_nn(z, w, b, keep_prob):
    """Recoder MLP for continuous states"""
    decoded = tf.nn.dropout(tf.nn.relu(tf.matmul(z, w[0]) + b[0]), keep_prob)
    for i in range(1, len(w) - 1):
        decoded = tf.nn.dropout(tf.nn.relu(tf.matmul(decoded, w[i]) + b[i]), keep_prob)

    decoded = tf.matmul(decoded, w[-1]) + b[-1]
    return decoded

"""
Distribution helpers
"""

def sample_normal(loc, scale, D=1, laten_dim=1):
    """Sample from a normal"""
    epsilon = tf.random_normal([D, laten_dim])
    sample = loc + scale * epsilon
    return sample


def make_class_pred(cst, y_in, i_y, w, b, batch_size, tau, n_class, normalize=False):
    """Uses Gumble to infer distribution over classes
       See https://github.com/vithursant/VAE-Gumbel-Softmax/ 
    """
    if len(w) > 1:
        cst = tf.nn.relu(tf.matmul(cst, w[-2]) + b[-2])
    y_logits = tf.matmul(cst, w[-1]) + b[-1]
    y_sample = tf.reshape(gumbel_softmax(y_logits, tau, hard=False), [batch_size, n_class])
    
    y_in = tf.cast(y_in, tf.float32) 
    i_y  = tf.cast(i_y, tf.float32) 
    y_sample = i_y*y_in + (1-i_y)*y_sample
     
    y_logits_with_sm  = tf.nn.softmax(y_logits)
    log_q_y = tf.log(y_logits_with_sm + 1e-20)
    return (y_sample, y_logits, y_logits_with_sm, log_q_y)


def sample_gumbel(shape, eps=1e-20):
    """Sample from a Gumbel"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample and apply stable softmax"""
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y




