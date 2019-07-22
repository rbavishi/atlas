class Network:
    def __init__(self):
        self.sess = None
        self.graph = None
        self.tf_config = None


class NetworkComponent:
    def __init__(self):
        self.placeholders = {}
        self.weights = {}
        self.ops = {}
