from collections import namedtuple
from itertools import product


class RunBuilder():
    '''Class used to generate runs during hyper parameter search'''
    @staticmethod
    def get_runs(params):
        # creates Run class to encapsulate data
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
