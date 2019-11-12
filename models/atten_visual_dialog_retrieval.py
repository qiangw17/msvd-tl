import tensorflow as tf 
import numpy as np 


class VisualDialogRetrieval(object):

    def __init__(self,  vocab_size=1000,
                        hidden_dim=256,
                        max_video_enc_steps=50,
                        max_context_enc_steps=50,
                        max_response_enc_steps=20,
                        emb_dim=128,
                        num_layers=2,
                        img_dim = 1536,
                        rand_unif_init_mag=0.08,
                        trunc_norm_init_std=1e-4,
                        cell_type='lstm',
                        optimizer_type = 'adam',
                        learning_rate = 0.001,
                        max_grad_clip_norm = 10,
                        beam_size = 1,
                        wemb = None,
                        loss_function = 'cross-entropy',
                        enable_video_context=True,
                        enable_chat_context=True,
                        enable_dropout=True,
                        is_training=True):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_video_enc_steps = max_video_enc_steps
        self.max_context_enc_steps = max_context_enc_steps
        self.max_response_enc_steps = max_response_enc_steps
        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.rand_unif_init_mag = rand_unif_init_mag
        self.trunc_norm_init_std = trunc_norm_init_std
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.max_grad_clip_norm = max_grad_clip_norm
        self.beam_size = beam_size
        self.enable_dropout = enable_dropout
        self.is_training = is_training
        self.enable_video_context = enable_video_context
        self.enable_chat_context = enable_chat_context
        self.loss_function = loss_function
        



        # create a debugger variable to use it to debug some other variable
        self.debugger = []
        


        if self.enable_dropout:
            self.keep_prob = tf.placeholder(tf.float32)
        # creating placeholders
        self.video_enc_batch = tf.placeholder(tf.float32, [None, None, self.img_dim])
        self.video_enc_mask_batch = tf.placeholder(tf.float32, [None, None])
        self.context_enc_batch = tf.placeholder(tf.int32, [None, self.max_context_enc_steps])
        self.context_enc_mask_batch = tf.placeholder(tf.float32, [None, self.max_context_enc_steps])
        self.response_enc_batch = tf.placeholder(tf.int32, [None, self.max_response_enc_steps])
        self.response_enc_mask_batch = tf.placeholder(tf.float32, [None, self.max_response_enc_steps])
        self.target_label_batch = tf.placeholder(tf.int32,[None])
        self.timestamp_offset_batch = tf.placeholder(tf.float32, [None, None])
        self.batch_size = tf.shape(self.video_enc_batch)[0]
        #self.max_video_enc_steps = tf.shape(self.video_enc_batch)[1]



        # word embedding look up
        if wemb is None:
            self.wemb = tf.Variable(tf.random_uniform([self.vocab_size,self.emb_dim], -self.rand_unif_init_mag,self.rand_unif_init_mag), name='Wemb')
        else:
            self.wemb = tf.Variable(wemb,name='Wemb')




        self.rand_unif_init = tf.random_uniform_initializer(-self.rand_unif_init_mag, self.rand_unif_init_mag)
        self.trunc_normal_init = tf.truncated_normal_initializer(stddev=self.trunc_norm_init_std)
        


        # creating rnn cells
        if self.cell_type == 'lstm':
            self.rnn_cell = tf.contrib.rnn.LSTMCell
        elif self.cell_type == 'gru':
            self.rnn_cell = tf.contrib.rnn.GRUCell
        # multi-layer cell setup 
        if self.num_layers > 1:
            self.video_enc_rnn_cell_fw = []
            self.video_enc_rnn_cell_bw = []
            self.context_enc_rnn_cell_fw = []
            self.context_enc_rnn_cell_bw = []
            self.response_enc_rnn_cell_fw = []
            self.response_enc_rnn_cell_bw = []
            for _ in range(self.num_layers):
                self.video_enc_rnn_cell_fw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.video_enc_rnn_cell_bw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.context_enc_rnn_cell_fw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.context_enc_rnn_cell_bw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.response_enc_rnn_cell_fw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.response_enc_rnn_cell_bw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
          

            self.video_enc_rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(self.video_enc_rnn_cell_fw,state_is_tuple=True)
            self.video_enc_rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(self.video_enc_rnn_cell_bw,state_is_tuple=True)
            self.context_enc_rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(self.context_enc_rnn_cell_fw,state_is_tuple=True)
            self.context_enc_rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(self.context_enc_rnn_cell_bw,state_is_tuple=True)
            self.response_enc_rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(self.response_enc_rnn_cell_fw,state_is_tuple=True)
            self.response_enc_rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(self.response_enc_rnn_cell_bw,state_is_tuple=True)
        else:
            self.video_enc_rnn_cell_fw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.video_enc_rnn_cell_bw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.context_enc_rnn_cell_fw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.context_enc_rnn_cell_bw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.response_enc_rnn_cell_fw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.response_enc_rnn_cell_bw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)

        ## add dropout to lstm
        if self.enable_dropout:
            self.rnn_cell = tf.contrib.rnn.DropoutWrapper(self.rnn_cell,output_keep_prob=self.keep_prob)
        
        
        # with tf.variable_scope('video_encoder_reduce_encoder_final_state',reuse=False):
        #     w_reduce = tf.get_variable('w_reduce_c', [self.img_dim, self.hidden_dim*2], dtype=tf.float32, initializer=self.trunc_normal_init)
        #     bias_reduce = tf.get_variable('bias_reduce_c',[self.hidden_dim*2],dtype=tf.float32, initializer=self.trunc_normal_init)
        # self.video_encoder_hidden_states = tf.nn.relu(tf.matmul(tf.reshape(self.video_enc_batch, (-1, self.img_dim)), w_reduce) + bias_reduce)
        # self.video_encoder_hidden_states = tf.reshape(self.video_encoder_hidden_states, (-1, self.max_video_enc_steps, self.hidden_dim*2))


        self.video_encoder_output,self.video_encoder_hidden_states = self._encoder(self.video_enc_rnn_cell_fw,self.video_enc_rnn_cell_bw,self.video_enc_batch,self.video_enc_mask_batch,scope='video_encoder')
        print ('video_encoder_output: ', self.video_encoder_output, self.video_encoder_output.shape, 'video_encoder_hidden_states: ', self.video_encoder_hidden_states, self.video_encoder_hidden_states.shape)
        self.context_encoder_output,self.context_encoder_hidden_states = self._encoder(self.context_enc_rnn_cell_fw,self.context_enc_rnn_cell_bw,self.context_enc_batch,self.context_enc_mask_batch,scope='context_encoder')
        print ('context_encoder_output: ', self.context_encoder_output, self.context_encoder_output.shape, 'context_encoder_hidden_states: ', self.context_encoder_hidden_states, self.context_encoder_hidden_states.shape)
        self.response_encoder_output,self.response_encoder_hidden_states = self._encoder(self.response_enc_rnn_cell_fw,self.response_enc_rnn_cell_bw,self.response_enc_batch,self.response_enc_mask_batch,scope='response_encoder')
        print ('response_encoder_output: ', self.response_encoder_output, self.response_encoder_output.shape, 'response_encoder_hidden_states: ', self.response_encoder_hidden_states, self.response_encoder_hidden_states.shape)


        self.video_feats = self.video_encoder_hidden_states
        print ('video_feats: ', self.video_feats, self.video_feats.shape)
        
        self.context_feats = self._context_cross_attention(self.context_encoder_hidden_states, self.video_encoder_hidden_states, self.context_enc_mask_batch, self.video_enc_mask_batch, self.max_video_enc_steps, self.max_context_enc_steps, scope = 'context_cross_attention')
        print ('context_feats: ', self.context_feats, self.context_feats.shape)

        self.response_feats = self._response_cross_attention(self.response_encoder_hidden_states, self.video_encoder_hidden_states, self.context_encoder_hidden_states,
                                                            self.response_enc_mask_batch, self.video_enc_mask_batch, self.context_enc_mask_batch, 
                                                            self.max_response_enc_steps, self.max_video_enc_steps, self.max_context_enc_steps, scope = 'response_cross_attention')
        print ('response_feats: ', self.response_feats, self.response_feats.shape)



        self.video_encoder_final_state = self._self_attention(self.video_feats, self.video_enc_mask_batch, self.max_video_enc_steps, scope = 'video_self_attention')
        print ('video_encoder_final_state: ', self.video_encoder_final_state, self.video_encoder_final_state.shape)
        self.context_encoder_final_state = self._self_attention(self.context_feats, self.context_enc_mask_batch, self.max_context_enc_steps, scope = 'context_self_attention')
        print ('context_encoder_final_state: ', self.context_encoder_final_state, self.context_encoder_final_state.shape)
        self.response_encoder_final_state = self._self_attention(self.response_feats, self.response_enc_mask_batch, self.max_response_enc_steps, scope = 'response_self_attention')
        print ('response_encoder_final_state: ', self.response_encoder_final_state, self.response_encoder_final_state.shape)




        if self.is_training:
            self.loss, self.response_loss, self.frame_loss, self.response_probs, self.frame_probs = self._calculate_loss(self.video_encoder_final_state,self.context_encoder_final_state,self.response_encoder_final_state,self.video_encoder_hidden_states,self.target_label_batch,self.timestamp_offset_batch)
            
            # calculate gradients
            if self.optimizer_type == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer_type == 'sgd':
                    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer_type == 'adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif self.optimizer_type == 'adadelta':
                    self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            else:
                    raise Exception('optimizer type:{} not supported'.format(self.optimizer_type))
    
            self.tvars = tf.trainable_variables()
            self.grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,self.tvars),self.max_grad_clip_norm)
            self.train_op=self.optimizer.apply_gradients(zip(self.grads,self.tvars))
        else:
            response_logits, frame_logits = self._projection_layer(self.video_encoder_final_state,self.context_encoder_final_state,self.response_encoder_final_state,self.video_encoder_hidden_states)
            self.response_probs = tf.sigmoid(response_logits)
            self.frame_probs = tf.sigmoid(response_logits)



    def _video_embed(self,video_batch):
        """ takes video batch of size batch_size,encoder_steps,features and project it 
            down to lower space: batch_size,encoder_steps,embedding_feature space
        """
        video_batch = tf.reshape(video_batch,[self.batch_size*self.max_video_enc_steps,self.img_dim])

        with tf.variable_scope('video_embeddings',reuse=False):
            weight = tf.get_variable('weights',[self.img_dim,256])
            emb_video_batch = tf.matmul(video_batch,weight)

        emb_video_batch = tf.reshape(emb_video_batch,[self.batch_size,self.max_video_enc_steps,256])

        return emb_video_batch


    def _reduce_encoder_state(self,state_fw,state_bw,scope):
        with tf.variable_scope(scope+'_reduce_encoder_final_state',reuse=False):
            w_reduce_c = tf.get_variable('w_reduce_c', [self.hidden_dim*2, self.hidden_dim], dtype=tf.float32, initializer=self.trunc_normal_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [self.hidden_dim*2, self.hidden_dim], dtype=tf.float32, initializer=self.trunc_normal_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c',[self.hidden_dim],dtype=tf.float32, initializer=self.trunc_normal_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h',[self.hidden_dim],dtype=tf.float32, initializer=self.trunc_normal_init)

        if self.num_layers <=1:
            old_c = tf.concat([state_fw.c, state_bw.c],axis=1) # Concatenation of fw and bw cell
            old_h = tf.concat([state_fw.h, state_bw.h],axis=1) # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
            # return tf.contrib.rnn.LSTMStateTuple(new_c,new_h)
            return tf.concat([new_c,new_h], axis = -1)
        else:
            final_state = []
            for i in range(self.num_layers):
                old_c = tf.concat([state_fw[i].c, state_bw[i].c],axis=1) # Concatenation of fw and bw cell
                old_h = tf.concat([state_fw[i].h, state_bw[i].h],axis=1) # Concatenation of fw and bw state
                new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
                new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
                # final_state.append(tf.contrib.rnn.LSTMStateTuple(new_c,new_h))
                final_state.append(tf.concat([new_c,new_h], axis = -1))
            return final_state



    def _encoder(self,cell_fw,cell_bw,inputs,inputs_mask,scope):
        # ENCODER PART
        if 'video' in scope:
            current_embed = self._video_embed(inputs)
        else:
            current_embed = tf.nn.embedding_lookup(self.wemb,inputs)

        # add dropout
        if self.enable_dropout:
            current_embed = tf.nn.dropout(current_embed,self.keep_prob)

        # find the seqence lengths from the mask placeholders
        seq_len = tf.reduce_sum(inputs_mask,1)
        seq_len = tf.cast(seq_len,dtype=tf.int32)

        ((output_fw, output_bw),(state_fw,state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,current_embed,sequence_length=seq_len,dtype=tf.float32,swap_memory=True,scope=scope)
        outputs = tf.concat((output_fw, output_bw),axis=2) # hidden fw/bw are in tuple form, concat them
        state = self._reduce_encoder_state(state_fw,state_bw,scope)
        
        return state,outputs




    def _context_cross_attention(self,context_state, video_state, context_mask, video_mask, video_step, context_step, scope):
        with tf.variable_scope(scope, reuse = False):
            context_weight = tf.get_variable('Wc', [2*self.hidden_dim, 2*self.hidden_dim])
            video_weight = tf.get_variable('Wv', [2*self.hidden_dim, 2*self.hidden_dim])
            weight = tf.get_variable('W', [4*self.hidden_dim, 2*self.hidden_dim])
            
            compressed_context = tf.reshape(context_state, [self.batch_size*context_step, 2*self.hidden_dim])
            compressed_video = tf.reshape(video_state, [self.batch_size*video_step, 2*self.hidden_dim])

            context_proj = tf.nn.relu(tf.matmul(compressed_context, context_weight))
            video_proj = tf.nn.relu(tf.matmul(compressed_video, video_weight))

            uncompressed_context = tf.reshape(context_proj, [self.batch_size, context_step, 2*self.hidden_dim])
            uncompressed_video = tf.reshape(video_proj, [self.batch_size, video_step, 2*self.hidden_dim])


            attn_score = tf.matmul(uncompressed_context, tf.transpose(uncompressed_video, [0, 2, 1]))
            attn_score = tf.nn.softmax(attn_score)
            attn_score = attn_score * tf.tile(tf.expand_dims(video_mask, 1), [1, context_step, 1])
            attn_sums = tf.reduce_sum(attn_score, -1, keep_dims = True)
            attn_weight = attn_score / attn_sums
            attended_video_on_context = tf.matmul(attn_weight, video_state)

            compressed_updataed_context = tf.reshape(tf.concat([context_state, attended_video_on_context], axis = -1), [self.batch_size*context_step, 4*self.hidden_dim])
            updataed_context_state = tf.tanh(tf.matmul(compressed_updataed_context, weight))
            unompressed_updataed_context = tf.reshape(updataed_context_state, [self.batch_size, context_step, 2*self.hidden_dim])

            return unompressed_updataed_context
    def _response_cross_attention(self, response_state, video_state, context_state, response_mask, video_mask, context_mask, response_step, video_step, context_step, scope):
        with tf.variable_scope(scope, reuse = False):
            response_weight = tf.get_variable('Wr', [2*self.hidden_dim, 2*self.hidden_dim])
            context_weight = tf.get_variable('Wc', [2*self.hidden_dim, 2*self.hidden_dim])
            video_weight = tf.get_variable('Wv', [2*self.hidden_dim, 2*self.hidden_dim])
            weight = tf.get_variable('W', [6*self.hidden_dim, 2*self.hidden_dim])
            
            compressed_response = tf.reshape(response_state, [self.batch_size*response_step, 2*self.hidden_dim])
            compressed_context = tf.reshape(context_state, [self.batch_size*context_step, 2*self.hidden_dim])
            compressed_video = tf.reshape(video_state, [self.batch_size*video_step, 2*self.hidden_dim])

            response_proj = tf.nn.relu(tf.matmul(compressed_response, response_weight))
            context_proj = tf.nn.relu(tf.matmul(compressed_context, context_weight))
            video_proj = tf.nn.relu(tf.matmul(compressed_video, video_weight))

            uncompressed_response = tf.reshape(response_proj, [self.batch_size, response_step, 2*self.hidden_dim])
            uncompressed_context = tf.reshape(context_proj, [self.batch_size, context_step, 2*self.hidden_dim])
            uncompressed_video = tf.reshape(video_proj, [self.batch_size, video_step, 2*self.hidden_dim])
            

            c2r_attn_score = tf.matmul(uncompressed_response, tf.transpose(uncompressed_context, [0, 2, 1]))
            c2r_attn_score = tf.nn.softmax(c2r_attn_score)
            c2r_attn_score = c2r_attn_score * tf.tile(tf.expand_dims(context_mask, 1), [1, response_step, 1])
            c2r_attn_sums = tf.reduce_sum(c2r_attn_score, -1, keep_dims = True)
            c2r_attn_weight = c2r_attn_score / c2r_attn_sums
            attended_context_on_response = tf.matmul(c2r_attn_weight, context_state)


            v2r_attn_score = tf.matmul(uncompressed_response, tf.transpose(uncompressed_video, [0, 2, 1]))
            v2r_attn_score = tf.nn.softmax(v2r_attn_score)
            v2r_attn_score = v2r_attn_score * tf.tile(tf.expand_dims(video_mask, 1), [1, response_step, 1])
            v2r_attn_sums = tf.reduce_sum(v2r_attn_score, -1, keep_dims = True)
            v2r_attn_weight = v2r_attn_score / v2r_attn_sums
            attended_video_on_response = tf.matmul(v2r_attn_weight, video_state)

            compressed_updataed_response = tf.reshape(tf.concat([response_state, attended_context_on_response, attended_video_on_response], axis = -1), [self.batch_size*response_step, 6*self.hidden_dim])
            updataed_response_state = tf.tanh(tf.matmul(compressed_updataed_response, weight))
            uncompressed_updataed_response = tf.reshape(updataed_response_state, [self.batch_size, response_step, 2*self.hidden_dim])

            return uncompressed_updataed_response
    def _self_attention(self,hidden_states,mask,step_size,scope):
        compressed_hidden_states = tf.reshape(hidden_states,[self.batch_size*step_size,2*self.hidden_dim])

        with tf.variable_scope(scope,reuse=False):
            Wa = tf.get_variable('Wa',[2*self.hidden_dim,2*self.hidden_dim])
            ba = tf.get_variable('ba',[2*self.hidden_dim],initializer=tf.constant_initializer(0.0))
            output = tf.tanh(tf.matmul(compressed_hidden_states,Wa)+tf.tile(tf.expand_dims(ba,0),[self.batch_size*step_size,1]))




            V = tf.get_variable('V',[2*self.hidden_dim,1])
            attn_dist = tf.matmul(output,V)
            attn_dist = tf.reshape(attn_dist,[self.batch_size,step_size])
            attn_dist = tf.nn.softmax(attn_dist)
            attn_dist *= mask
            masked_sums = tf.reduce_sum(attn_dist,1)
            attn_dist = attn_dist/tf.reshape(masked_sums,[-1,1])


        context_vector= tf.reduce_sum(tf.reshape(attn_dist,[self.batch_size,-1,1,1]) * tf.expand_dims(hidden_states,axis=2), [1,2]) # shape: batch_size,2*dim_hidden


        return context_vector





    def _projection_layer(self,video_state,context_state,response_state,original_video_state,reuse=False):
        # consider only input hidden state information 

        # For response prediction
        pred_response_proj = tf.expand_dims(response_state,[2])

        if self.enable_video_context:
            # calculate projections
            with tf.variable_scope('pred_video_projection',reuse=False):
                W_pv = tf.get_variable('weights',[2*self.hidden_dim,2*self.hidden_dim])
            pred_video_proj = tf.matmul(video_state,W_pv)
            pred_video_proj = tf.expand_dims(pred_video_proj,[2])

        if self.enable_chat_context:
            ''' have to correct the name space later '''
            with tf.variable_scope('pred_context_projection',reuse=False):
                W_pc = tf.get_variable('weights',[2*self.hidden_dim,2*self.hidden_dim])
            pred_context_proj = tf.matmul(context_state,W_pc)
            pred_context_proj = tf.expand_dims(pred_context_proj,[2])

        with tf.variable_scope('pred_projection_layer',False):
            pb = tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))
        
        if self.enable_video_context and self.enable_chat_context:
            response_logits = tf.add(tf.matmul(pred_video_proj,pred_response_proj,True),tf.matmul(pred_context_proj,pred_response_proj,True))+pb
        elif self.enable_video_context and not self.enable_chat_context:
            response_logits = tf.matmul(pred_video_proj,pred_response_proj,True)+pb
        elif not self.enable_video_context and self.enable_chat_context:
            response_logits = tf.matmul(pred_context_proj,pred_response_proj,True)+pb
        else:
            raise Exception('At least one context must be present !!')
        response_logits = tf.squeeze(response_logits, [1,2])


        

        # For localization
        compressed_video_states = tf.reshape(original_video_state, [-1, 2*self.hidden_dim])
        print (original_video_state, compressed_video_states)
        with tf.variable_scope('loc_video_projection',reuse=False):
            W_lv = tf.get_variable('weights',[2*self.hidden_dim,2*self.hidden_dim])
        loc_video_proj = tf.matmul(compressed_video_states,W_lv)
        loc_video_proj = tf.reshape(loc_video_proj, [self.batch_size, self.max_video_enc_steps, 2*self.hidden_dim])

        # calculate projections
        with tf.variable_scope('loc_response_projection',reuse=False):
            W_lr = tf.get_variable('weights',[2*self.hidden_dim,2*self.hidden_dim])
        loc_response_proj = tf.matmul(response_state,W_lr)
        loc_response_proj = tf.expand_dims(loc_response_proj,[2])

        if self.enable_chat_context:
            ''' have to correct the name space later '''
            with tf.variable_scope('loc_context_projection',reuse=False):
                W_lc = tf.get_variable('weights',[2*self.hidden_dim,2*self.hidden_dim])
            loc_context_proj = tf.matmul(context_state,W_lc)
            loc_context_proj = tf.expand_dims(loc_context_proj,[2])

        with tf.variable_scope('loc_projection_layer',False):
            lb = tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))

        if self.enable_chat_context:
            frame_logits = tf.add(tf.matmul(loc_video_proj, loc_context_proj), tf.matmul(loc_video_proj, loc_response_proj)) + lb
        else:
            frame_logits = tf.matmul(loc_video_proj, loc_response_proj) + lb

        frame_logits = tf.squeeze(frame_logits, [2])



        return response_logits, frame_logits


    def _calculate_loss(self,video_state,context_state,response_state,original_video_state,target_labels,target_time_labels):
        print ('self._calculate_loss--original_video_state: ', original_video_state)
        response_logits, frame_logits = self._projection_layer(video_state,context_state,response_state, original_video_state)
        response_probs = tf.sigmoid(response_logits)
        frame_probs = tf.sigmoid(frame_logits)
        self.debugger = response_logits

        if self.loss_function == '3triplet':
            response_log_prob = tf.log(response_probs)
            response_log_prob = tf.split(response_log_prob,4)
            response_loss = tf.maximum(0.0,0.1+response_log_prob[1]-response_log_prob[0]) 
            if self.enable_video_context: # with video negative example
                response_loss += tf.maximum(0.0,0.1+response_log_prob[2]-response_log_prob[0])
            if self.enable_chat_context: # with chat negative eample
                response_loss += tf.maximum(0.0,0.1+response_log_prob[3]-response_log_prob[0])
            response_loss = tf.reduce_mean(response_loss)

            frame_selected_logits = tf.reshape(frame_logits, [self.batch_size/4, 4, self.max_video_enc_steps])[:, 0, :]
            frame_probs = tf.reshape(frame_probs, [self.batch_size/4, 4, self.max_video_enc_steps])[:, 0, :]
            frame_labels = tf.reshape(target_time_labels, [self.batch_size/4, 4, self.max_video_enc_steps])[:, 0, :]
            frame_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=frame_selected_logits, labels=tf.to_float(frame_labels))
            frame_loss = tf.reduce_mean(frame_cross_entropy)

            
            loss = response_loss + frame_loss
            

        else:
            raise Exception('Unknown loss function:{}'.format(self.loss_function)) 
        

        return loss, response_loss, frame_loss, response_probs, frame_probs


