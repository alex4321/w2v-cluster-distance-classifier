class DatasetItem:
    def __init__(self, text, labels):
        """
        Dataset item
        :param text: item text
        :type text: str
        :param labels: class names
        :type labels: list[str]
        """
        self.text = text
        self.labels = labels