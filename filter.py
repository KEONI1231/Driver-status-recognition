
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

class CascadedLPF:
    def __init__(self, alpha1, alpha2):
        self.lpf1 = LPF(alpha1)
        self.lpf2 = LPF(alpha2)

    def compute(self, x):
        x_lpf1 = self.lpf1.compute(x)
        x_lpf2 = self.lpf2.compute(x_lpf1)
        return x_lpf2