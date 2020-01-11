class Scoring:
    def __init__(self, classifier, modality, precision = 0, recall = 0, f1 = 0):
        self.classifier = classifier
        self.modality = modality
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def __str__(self):
        return "Classifier: {0} | Modality: {1} | Precision: {2}, Recall: {3}, F1: {4}"\
            .format(self.classifier, self.modality, self.precision, self.recall, self.f1)

    def to_tuple(self):
        return (self.classifier, self.modality, self.precision, self.recall, self.f1)
