import numpy as np
from matplotlib import pyplot as plt

from ga import GA, GAAnimation

ga_solver = GA(dna_size=15, population_size=100, crossover_rate=0.8, mutation_rate=0.003, n_generations=100)

# 可视化求解问题背景
xs = np.linspace(0, 30, 5000)
ys = ga_solver.fitness_func(xs)
plt.plot(xs, ys, label=ga_solver.fitness_func_latex)
plt.xlabel('x')
plt.ylabel('Target Function')
plt.legend(loc='lower center')
highest_x_index = ys.tolist().index(max(ys))
plt.scatter(xs[highest_x_index], ys[highest_x_index], c='black')
plt.annotate('({:.2f}, {:.2f})'.format(xs[highest_x_index], ys[highest_x_index]),
             xy=(xs[highest_x_index], ys[highest_x_index]), xytext=(xs[highest_x_index] - 2, ys[highest_x_index]),
             ha='right', va='center', arrowprops={'arrowstyle': '->'})
xs = xs[2500:]
ys = ys[2500:]
highest_x_index = ys.tolist().index(max(ys))
plt.scatter(xs[highest_x_index], ys[highest_x_index], c='black')
plt.annotate('({:.2f}, {:.2f})'.format(xs[highest_x_index], ys[highest_x_index]),
             xy=(xs[highest_x_index], ys[highest_x_index]), xytext=(xs[highest_x_index] + 2, ys[highest_x_index]),
             ha='left', va='center', arrowprops={'arrowstyle': '->'})
plt.savefig(ga_solver.pic_path + 'Problem Background.png', dpi=300)
plt.show()

# 初始化种群
population = ga_solver.initial_pop()
ga_solver.plot_population(population, 0)
plt.savefig(ga_solver.pic_path + 'Initial Population.png', dpi=300)
plt.show()

# 种群演化可视化 动画
population_records = ga_solver.revolution()
ga_animation = GAAnimation(pops=population_records)
ga_animation.plot()

# 种群演化可视化 箱线图
ga_solver.plot_evolution(population_records, [0, 71], fig_size=(8, 3))
plt.savefig(ga_solver.pic_path+'Revolution Boxplot.png', dpi=300)
plt.tight_layout()
plt.show()

# 关键帧分析
for n in [0, len(population_records)-1, 60, 25, 70]:
    ga_solver.plot_population(population_records[n], n)
    plt.savefig(ga_solver.pic_path+'Generation {}.png'.format(n), dpi=300)
    plt.show()

