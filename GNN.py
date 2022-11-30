#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np

from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LinearRegressionWeights
from federatedml.optim.initialize import Initializer
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.param.linear_regression_param import LinearParam
from federatedml.protobuf.generated import linr_model_param_pb2, linr_model_meta_pb2
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import LOGGER
from federatedml.util.fate_operator import vec_dot


class GNNLayer(BaseModel):
    def __init__(self, in_dim, out_dim,
                 attn_heads = 1,
                 attn_heads_reduction = 'concat',  # {'concat', 'average'}
                 dropout_rate = 0.1,
                 activation = None,
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 attn_kernel_initializer = 'glorot_uniform',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 attn_kernel_regularizer = None,
                 activity_regularizer = None,
                 **kwargs):
        
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.attn_kernel_initializer = attn_kernel_initializer

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.attn_kernel_regularizer = attn_kernel_regularizer
        self.activity_regularizer = activity_regularizer

        self.kernels = []
        self.biases = []
        self.atten_kernels = []

        super(MultiHeadGATLayer, self).__init__(**kwargs)


    def _init_model(self, params):
        super()._init_model(params)
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param

    def compute_wx(self, data_instances, coef_, intercept_=0):
        return data_instances.mapValues(
            lambda v: vec_dot(v.features, coef_) + intercept_)

    def _get_meta(self):
        meta_protobuf_obj = linr_model_meta_pb2.LinRModelMeta(penalty=self.model_param.penalty,
                                                              tol=self.model_param.tol,
                                                              alpha=self.alpha,
                                                              optimizer=self.model_param.optimizer,
                                                              batch_size=self.batch_size,
                                                              learning_rate=self.model_param.learning_rate,
                                                              max_iter=self.max_iter,
                                                              early_stop=self.model_param.early_stop,
                                                              fit_intercept=self.fit_intercept)
        return meta_protobuf_obj

    def _get_param(self):
        header = self.header
        # LOGGER.debug("In get_param, header: {}".format(header))
        weight_dict, intercept_ = {}, None
        if header is not None:
            weight_dict, intercept_ = self.get_weight_intercept_dict(header)

        best_iteration = -1 if self.validation_strategy is None else self.validation_strategy.best_iteration
        param_protobuf_obj = linr_model_param_pb2.LinRModelParam(iters=self.n_iter_,
                                                                 loss_history=self.loss_history,
                                                                 is_converged=self.is_converged,
                                                                 weight=weight_dict,
                                                                 intercept=intercept_,
                                                                 header=header,
                                                                 best_iteration=best_iteration)
        return param_protobuf_obj
    

    def build(self, input_shape):
        assert len(input_shape) >= 2

        for head in range(self.attn_heads):
            kernel = self.add_weight(shape=(self.in_dim, self.out_dim),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            if self.use_bias:
                bias = self.add_weight(shape=(self.out_dim, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)


            atten_kernel = self.add_weight(shape=(2 * self.out_dim, 1),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     name='kernel_{}'.format(head))
            self.atten_kernels.append(atten_kernel)

        self.built = True

    def call(self, inputs, training):
        X = inputs[0]
        A = inputs[1]

        N = X.shape[0]

        outputs = []
        for head in range(self.attn_heads):

            kernel = self.kernels[head]

            features = tf.matmul(X, kernel)

            concat_features = tf.concat(\
                [tf.reshape(tf.tile(features, [1, N]), [N * N, -1]),\
                tf.tile(features, [N, 1])], axis = 1)

            concat_features = tf.reshape(concat_features, [N, -1, 2 * self.out_dim])

            atten_kernel = self.atten_kernels[head]
            
            dense = tf.matmul(concat_features, atten_kernel)

            dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)

            dense = tf.reshape(dense, [N, -1])

            zero_vec = -9e15 * tf.ones_like(dense)
            attention = tf.where(A > 0, dense, zero_vec)

            dense = tf.keras.activations.softmax(attention, axis = -1)

            dropout_attn = tf.keras.layers.Dropout(self.dropout_rate)(dense, training = training)
            dropout_feat = tf.keras.layers.Dropout(self.dropout_rate)(features, training = training)

            node_features = tf.matmul(dropout_attn, dropout_feat)

            if self.use_bias:
                node_features = tf.add(node_features, self.biases[head])

            outputs.append(node_features)

        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis = -1)
        else:
            output = tf.reduce_mean(tf.stack(outputs), axis=-1)

        if self.activation is not None:
            output = self.activation(output)

        return output


class GraphAttentionModel(tf.keras.Model):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GraphAttentionModel, self).__init__()

        self.attention_layer1 = MultiHeadGATLayer(in_dim, hidden_dim, attn_heads = num_heads, activation = tf.keras.activations.elu)

        self.attention_layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, attn_heads = 1)
 
    def call(self, x, training = False):
        adj = x[1]

        x = self.attention_layer1(x, training)

        output = self.attention_layer2([x, adj], training)

        return output