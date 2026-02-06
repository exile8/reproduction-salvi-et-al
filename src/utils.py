class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.improved = False
        
    def __call__(self, score, epoch):
        self.improved = False
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.improved = True
            return False
        
        if self.mode == 'max':
            is_better = score >= self.best_score + self.min_delta
        else:
            is_better = score <= self.best_score - self.min_delta
        
        if is_better:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
        
        return self.counter >= self.patience

    def state_dict(self):
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
        }
    
    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.mode = state_dict['mode']
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']