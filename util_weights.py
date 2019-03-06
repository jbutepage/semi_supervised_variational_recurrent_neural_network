import tensorflow as tf

def create_ssvrnn(flags):
    
    # classfier - human
    wy_1 = tf.get_variable('wy_1', [flags['embedding']  + flags['rnn_sz'], flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    by_1 = tf.get_variable('by_1', [flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wy_2 = tf.get_variable('wy_2', [flags['embedding'], flags['dc']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    by_2 = tf.get_variable('by_2', [flags['dc']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wy = [wy_1, wy_2]
    by = [by_1, by_2]
    
    # classfier prior - human
    wy_p_1 = tf.get_variable('wy_p_1', [flags['rnn_sz'], flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    by_p_1 = tf.get_variable('by_p_1', [flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wy_p_2 = tf.get_variable('wy_p_2', [flags['embedding'], flags['dc']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    by_p_2 = tf.get_variable('by_p_2', [flags['dc']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wyp = [wy_p_1, wy_p_2]
    byp = [by_p_1, by_p_2]
    
    # encoder (embedding)
    wx_enc_emb = tf.get_variable('wx_enc_emb', [flags['input_sz'] , flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_enc_emb = tf.get_variable('bx_enc_emb', [flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    # encoder (latent variable z)    
    wx_enc_1 = tf.get_variable('wx_enc_1', [flags['embedding'] + flags['dc'] + flags['rnn_sz'], flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_enc_1 = tf.get_variable('bx_enc_1', [flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wx_enc_2 = tf.get_variable('wx_enc_2', [flags['embedding'], flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_enc_2 = tf.get_variable('bx_enc_2', [flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wx_enc_loc = tf.get_variable('wx_enc_loc', [flags['embedding'], flags['hidden']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_enc_loc = tf.get_variable('bx_enc_loc', [flags['hidden']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wx_enc_scale = tf.get_variable('wx_enc_scale', [flags['embedding'], flags['hidden']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_enc_scale = tf.get_variable('bx_enc_scale', [flags['hidden']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    we = [wx_enc_emb, wx_enc_1, wx_enc_2, wx_enc_loc, wx_enc_scale]
    be = [bx_enc_emb, bx_enc_1, bx_enc_2, bx_enc_loc, bx_enc_scale]

    # prior
    wx_enc_p_1 = tf.get_variable('wx_enc_p_1', [flags['dc'] + flags['rnn_sz'], flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_enc_p_1 = tf.get_variable('bx_enc_p_1', [flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wx_enc_p_loc = tf.get_variable('wx_enc_p_loc', [flags['embedding'], flags['hidden']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_enc_p_loc = tf.get_variable('bx_enc_p_loc', [flags['hidden']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wx_enc_p_scale = tf.get_variable('wx_enc_p_scale', [flags['embedding'], flags['hidden']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_enc_p_scale = tf.get_variable('bx_enc_p_scale', [flags['hidden']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wep = [wx_enc_p_1, wx_enc_p_loc, wx_enc_p_scale]
    bep = [bx_enc_p_1, bx_enc_p_loc, bx_enc_p_scale]
    
    
    # decoder
    wx_dec_1 = tf.get_variable('wx_dec_1', [flags['dc'] + flags['hidden'] + flags['rnn_sz'], flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_dec_1 = tf.get_variable('bx_dec_1', [flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wx_dec_2 = tf.get_variable('wx_dec_2', [flags['embedding'], flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_dec_2 = tf.get_variable('bx_dec_2', [flags['embedding']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wx_dec_3 = tf.get_variable('wx_dec_3', [flags['embedding'], flags['input_sz']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    bx_dec_3 = tf.get_variable('bx_dec_3', [flags['input_sz']], initializer=tf.initializers.truncated_normal(0.0,0.01))
    wd = [wx_dec_1, wx_dec_2, wx_dec_3]
    bd = [bx_dec_1, bx_dec_2, bx_dec_3]
    
    # all weights 
    w = [we, wd, wy, wep, wyp]
    b = [be, bd, by, bep, byp]
    
    return w, b
