# Reproduced with edits from 
# https://github.com/CoinCheung/fixmatch-pytorch

class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name]
                + (1 - decay) * state[name]
            )
        self.step += 1

    def update_buffer(self):
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name].copy_(state[name])

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }