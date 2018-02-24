import pickle

datapath = '../cache'


class dataset(object):
    def __init__(self, **parts):
        self.parts = parts

    def __getitem__(self, key):
        return self.parts[key]

    def save(self, name):
        for part_name in self.parts:
            file_name = '{}/{}-{}.pickle'.format(datapath, part_name, name)
            with open(file_name, 'wb') as f:
                pickle.dump(self.parts[part_name], f)

    @classmethod
    def load(cls, part_name, name):
        file_name = '{}/{}-{}.pickle'.format(datapath, part_name, name)
        with open(file_name, 'rb') as f:
            return pickle.load(f)
