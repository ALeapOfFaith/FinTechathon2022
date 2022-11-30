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

from .components import ComponentMeta

graph_analysis = ComponentMeta("GraphParam")


@graph_analysis.bind_param
def graph_nn_param():
    from federatedml.param import graph_param

    return GraphParam


@graph_analysis.bind_runner.on_guest
def graph_nn_runner_guest():
    from federatedml.param.gnn import  (
            graph_nn_guest
        )

    return graph_nn_guest


@graph_analysis.bind_runner.on_host
def graph_nn_runner_host():
    from federatedml.param.gnn import  (
            graph_nn_guest
    )

    return graph_nn_host


@graph_analysis.bind_runner.on_arbiter
def graph_nn_runner_arbiter():
    from federatedml.param.gnn import  (
            graph_nn_runner_arbiter
    )

    return graph_nn_runner_arbiter
