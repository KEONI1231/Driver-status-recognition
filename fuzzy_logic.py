import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

import matplotlib.pyplot as plt_module
# 가로축: 눈각도 범위, 세로축: 멤버십 함수 값
import warnings

warnings.filterwarnings('ignore')
x = np.arange(0, 1000, 1)

test_ms = 550
y1 = np.where(x < 200, 1, np.where(x < 400, (400 - x) / 200, 0))
y2 = np.where(x < 200, 0, np.where(x < 400, (x - 400) / 200 +1, np.where(x > 400, np.where(x<600,(600 - x)/200,0), 1)))
y3 = np.where(x < 400, 0, np.where(x < 600, (x - 600) / 200 +1, np.where(x > 600, np.where(x < 800,(800 - x)/200,0), 1)))
y4 = np.where(x < 600, 0, np.where(x < 800, (x - 800) / 200 + 1, 1))
#
#
hedge_mode1_y1 = np.where(x < 200, 1, np.where(x < 400, ((400 - x) / 200)**2, 0))
hedge_mode1_y2 = np.where(x < 200, 0, np.where(x < 400, ((x - 400) / 200 +1)**1.7, np.where(x > 400, np.where(x<600,((600 - x)/200)**1.7,0), 1)))
hedge_mode1_y3 = np.where(x < 400, 0, np.where(x < 600, ((x - 600) / 200 +1)**2, np.where(x > 600, np.where(x < 800,((800 - x)/200)**2,0), 1)))
hedge_mode1_y4 = np.where(x < 600, 0, np.where(x < 800, ((x - 800) / 200 + 1)**3, 1))


hedge_mode2_y1 = np.where(x <= 200, 1, np.where(x < 400, 2*(((400-x) / 200) ** 2), 0))
hedge_mode2_y2 = np.where(x < 200, 0, np.where(x < 400, ((x - 400) / 200 +1)**0.5, np.where(x > 400, np.where(x<600,((600 - x)/200)**0.5,0), 1)))
hedge_mode2_y3 = np.where(x < 400, 0, np.where(x < 600, ((x - 600) / 200 +1)**0.5, np.where(x > 600, np.where(x < 800,((800 - x)/200)**0.5,0), 1)))
hedge_mode2_y4 = np.where(x < 600, 0, np.where(x < 800, ((x - 800) / 200 + 1)**4, 1))
for i in x:
    if 0.5<=1-2*(1-((400-i)/200))**2<=1 and i >200:
        hedge_mode2_y1[i] = 1-2*(1-((400-i)/200))**2

def fuzzy_logic_graph(ms):
    def_hedge_mode2_y1 = np.where(x <= 200, 1, np.where(x < 400, 2 * (((400 - x) / 200) ** 2), 0))
    def_hedge_mode2_y2 = np.where(x < 200, 0, np.where(x < 400, ((x - 400) / 200 + 1) ** 0.5,
                                                   np.where(x > 400, np.where(x < 600, ((600 - x) / 200) ** 0.5, 0),
                                                            1)))
    def_hedge_mode2_y3 = np.where(x < 400, 0, np.where(x < 600, ((x - 600) / 200 + 1) ** 0.5,
                                                   np.where(x > 600, np.where(x < 800, ((800 - x) / 200) ** 0.5, 0),
                                                            1)))
    def_hedge_mode2_y4 = np.where(x < 600, 0, np.where(x < 800, ((x - 800) / 200 + 1) ** 4, 1))
    for i in x:
        if 0.5 <= 1 - 2 * (1 - ((400 - i) / 200)) ** 2 <= 1 and i > 200:
            def_hedge_mode2_y1[i] = 1 - 2 * (1 - ((400 - i) / 200)) ** 2

    def_y = np.fmax(np.fmax(np.fmax(def_hedge_mode2_y1,def_hedge_mode2_y2),def_hedge_mode2_y3),def_hedge_mode2_y4)

    return def_y

def_y = fuzzy_logic_graph(test_ms)

#기본 그래프 그리기
plt.plot(x, y1, label='Normal')
plt.plot(x, y2, label='Caution')
plt.plot(x, y3, label='Drowsy')
plt.plot(x, y4, label='Danger!!!')
# plt.legend()
# plt.show()
plt.legend()
plt.show()

#hedge_mode1 그래프 그리기
plt1.plot(x, hedge_mode1_y1, label='A little')
plt1.plot(x, hedge_mode1_y2, label='Slightly')
plt1.plot(x, hedge_mode1_y3, label='Very')
plt1.plot(x, hedge_mode1_y4, label='Extremely!!!')
plt1.legend()
plt1.show()

# #hedge_mode2 그래프 그리기
# plt2.plot(x, hedge_mode2_y1, label='Indeed')
# plt2.plot(x, hedge_mode2_y2, label='slightly')
# plt2.plot(x, hedge_mode2_y3, label='Very')
# plt2.plot(x, hedge_mode2_y4, label='Very very!!!')
# plt2.legend()
# plt2.show()
# #
# if(def_y[test_ms] == hedge_mode2_y1[test_ms]) :
#     print("Indeed")
# if(def_y[test_ms] == hedge_mode2_y2[test_ms]) :
#     print("Slightly")
# if(def_y[test_ms] == hedge_mode2_y3[test_ms]) :
#     print("Very")
# if(def_y[test_ms] == hedge_mode2_y4[test_ms]) :
#     print("Very Very")
#
# print(def_y[test_ms],hedge_mode2_y3[test_ms])
# plt_module.plot(x, def_y, label='test')
# plt_module.legend()
# plt_module.show()

