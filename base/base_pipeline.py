from abc import ABC, abstractmethod


class BasePipeline(ABC):

    @abstractmethod
    def run(self):
        raise NotImplementedError("This method is not implemented")
