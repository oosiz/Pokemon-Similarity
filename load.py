import os

class DataLoad:
    def __init__(self):
        pass


    def _load(self):
        cur_dir = os.getcwd()
        data_list = os.listdir(cur_dir + "\images")
        
        return data_list