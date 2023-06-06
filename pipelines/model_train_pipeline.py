import gin
from base import BasePipeline



@gin.configurable
class ModelTrainPipeline(BasePipeline):


    def __init__(self) -> None:
        super().__init__()


    def run(self):
        print("training....")