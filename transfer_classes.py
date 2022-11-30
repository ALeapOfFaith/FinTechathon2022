from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


# noinspection PyAttributeOutsideInit
class GraphNNTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.batch_data_index = self._create_variable(name='batch_data_index', src=['guest'], dst=['host'])
        self.batch_info = self._create_variable(name='batch_info', src=['guest'], dst=['host', 'arbiter'])
        self.converge_flag = self._create_variable(name='converge_flag', src=['arbiter'], dst=['host', 'guest'])
        self.fore_gradient = self._create_variable(name='fore_gradient', src=['guest'], dst=['host'])
        self.forward_hess = self._create_variable(name='forward_hess', src=['guest'], dst=['host'])
        self.guest_gradient = self._create_variable(name='guest_gradient', src=['guest'], dst=['arbiter'])
        self.guest_hess_vector = self._create_variable(name='guest_hess_vector', src=['guest'], dst=['arbiter'])
        self.guest_optim_gradient = self._create_variable(name='guest_optim_gradient', src=['arbiter'], dst=['guest'])
        self.host_forward_dict = self._create_variable(name='host_forward_dict', src=['host'], dst=['guest'])
        self.host_gradient = self._create_variable(name='host_gradient', src=['host'], dst=['arbiter'])
        self.host_hess_vector = self._create_variable(name='host_hess_vector', src=['host'], dst=['arbiter'])
        self.host_loss_regular = self._create_variable(name='host_loss_regular', src=['host'], dst=['guest'])
        self.host_optim_gradient = self._create_variable(name='host_optim_gradient', src=['arbiter'], dst=['host'])
        self.host_prob = self._create_variable(name='host_prob', src=['host'], dst=['guest'])
        self.host_sqn_forwards = self._create_variable(name='host_sqn_forwards', src=['host'], dst=['guest'])
        self.loss = self._create_variable(name='loss', src=['guest'], dst=['arbiter'])
        self.loss_intermediate = self._create_variable(name='loss_intermediate', src=['host'], dst=['guest'])
        self.paillier_pubkey = self._create_variable(name='paillier_pubkey', src=['arbiter'], dst=['host', 'guest'])
        self.sqn_sample_index = self._create_variable(name='sqn_sample_index', src=['guest'], dst=['host'])
        self.use_async = self._create_variable(name='use_async', src=['guest'], dst=['host'])
