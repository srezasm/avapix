from avapix.common.versioned_strategies.base_strategy import BaseStrategy


class StrategyContext:
    def set_strategy(self, strategy: BaseStrategy):
        self.strategy = strategy

    def embed(self, text):
        # TODO: add version parameter
        return self.strategy.embed(text)

    def extract(self, image_file):
        # TODO: extract version from image_file
        return self.strategy.extract(image_file)
