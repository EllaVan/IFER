# from matplotlib import pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(dpi=500)
# ax = Axes3D(fig, auto_add_to_figure=False)
# len = 8
# step = 0.4
#
#
# def build_layer(z_value):
#     x = np.arange(-len, len, step)
#     y = np.arange(-len, len, step)
#     z1 = np.full(x.size, z_value / 2)
#     z2 = np.full(x.size, z_value / 2)
#     z1, z2 = np.meshgrid(z1, z2)
#     z = z1 + z2
#
#     x, y = np.meshgrid(x, y)
#     return (x, y, z)
#
# def build_gaussian_layer(mean, standard_deviation):
#     x = np.arange(-len, len, step)
#     y = np.arange(-len, len, step)
#     x, y = np.meshgrid(x, y)
#     z = np.exp(-((y - mean) ** 2 + (x - mean) ** 2) / (2 * (standard_deviation ** 2)))
#     z = z / (np.sqrt(2 * np.pi) * standard_deviation)
#     return (x, y, z)
#
#
# # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
# # x1, y1, z1 = build_layer(0.2)
# # ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, color='green')
#
# # x5, y5, z5 = build_layer(0.15)
# # ax.plot_surface(x5, y5, z5, rstride=1, cstride=1, color='pink')
#
# # x2, y2, z2 = build_layer(-0.26);
# # ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, color='yellow')
# #
# # x6, y6, z6 = build_layer(-0.22);
# # ax.plot_surface(x6, y6, z6, rstride=1, cstride=1, color='pink')
#
# # x4, y4, z4 = build_layer(0);
# # ax.plot_surface(x4, y4, z4, rstride=1, cstride=1, color='purple')
#
# x3, y3, z3 = build_gaussian_layer(0, 3)
# ax.grid(False)
# ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='rainbow')
# # ax.tick_params('both', labelleft=False)
# # ax.tick_params('y', labelleft=False) # 取消left即可（top，bottom，right）
# # ax.tick_params('z', labelleft=False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
'''让下面的都在一个框框里面'''
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
'''绘制3D空间（坐标轴）'''
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
'''把x,y绘制对应到底面的面上去'''
R = np.exp(-((Y - 0) ** 2 + (X - 0) ** 2) / (2 * (1.6 ** 2)))
# R = np.sqrt(X ** 2 + Y ** 2 + np.exp(np.pi))
# Z = np.tanh(R)
Z = R
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('Oranges')) # BuGn YlGn
# ax.tick_params('both', labelleft=False)
# ax.tick_params('y', labelleft=False) # 取消left即可（top，bottom，right）
# ax.tick_params('z', labelleft=False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
'''绘制3D cmap又一种方法'''
'''
plt.contourf 与 plt.contour 区别：
f：filled，也即对等高线间的填充区域进行填充（使用不同的颜色）
contourf：将不会再绘制等高线（显然不同的颜色分界就表示等高线本身）
'''
ax.contour(X, Y, Z, zdir='x', offset=4, cmap='Wistia')
ax.contour(X, Y, Z, zdir='y', offset=-4, cmap='Purples')
# '''顺便在某平面画个等高线 zdir是决定从哪个方向压下去'''
plt.show()