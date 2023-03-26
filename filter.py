
class LPF:
    def __init__(self, alpha):
        self.alpha = alpha
        # 이전 예측값
        self.prev_x = 0

    def compute(self, x):
        # low pass filter
        x_lpf = self.alpha * self.prev_x + (1 - self.alpha) * x
        # 이전 스텝 값 갱신
        self.prev_x = x_lpf
        return x_lpf