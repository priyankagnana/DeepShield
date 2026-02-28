class EarlyStopping:
    def __init__(self, patience=12, delta=0.0, monitor="max"):
        """
        patience: epochs to wait without improvement before stopping
        delta:    minimum change to count as an improvement
        monitor:  "max" to track accuracy (higher is better),
                  "min" to track loss (lower is better)
        """
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric: float):
        if self.best_score is None:
            self.best_score = metric
        elif self._is_improvement(metric):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, metric: float) -> bool:
        if self.monitor == "max":
            return metric > self.best_score + self.delta
        return metric < self.best_score - self.delta