
class AverageMeter(object):
    """Computes and stores current value and average"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MovingAverageMeter(object):
    """Computes and stores the current value and moving average"""
    def __init__(self, alpha=0.995):
        self.reset()
        self.alpha = alpha

    def reset(self):
        self.val = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        if self.avg == 0:
            self.avg = val
        else:
            self.avg = self.avg * self.alpha + val * (1 - self.alpha)

class GradientMeter(object):
    """Computes and stores the current value, moving average and current gradient"""
    def __init__(self, alpha=0.995, n=1000):
        self.reset()
        self.alpha = alpha
        self.n = n

    def reset(self):
        self.val = 0
        self.avg = 0
        self.avg_list = []
        self.gradient = None

    def update(self, val, n=1):
        self.val = val
        if self.avg == 0:
            self.avg = val
        else:
            self.avg = self.avg * self.alpha + val * (1 - self.alpha)
        self.avg_list.append(self.avg)
        if len(self.avg_list) >= self.n:
            self.gradient = (self.avg_list[-1] - self.avg_list[-self.n]) / self.n
        else:
            self.gradient = None
