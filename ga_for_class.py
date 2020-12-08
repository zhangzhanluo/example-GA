import time
from ga import GA
from matplotlib import pyplot as plt

# 使用说明
#   改动下方参数值，可以获得不同的图形和文字输出。但注意，由于过程随机性非常强，两次运行结果可能差异非常大，因此建议多跑几次以查看效果。
#       dna_size: 整数，控制编码精度
#       population_size: 整数，控制种群数量
#       crossover_rate: 小数，介于0-1之间，控制交叉概率
#       mutation_rate: 小数，介于0-1之间，一般建议远小于1，控制变异概率
#       n_generation: 整数，控制迭代次数
ga_solver = GA(dna_size=15, population_size=10, crossover_rate=0.8, mutation_rate=0.003, n_generations=100)

# 种群演化
start_time = time.time()
population_records = ga_solver.revolution()
end_time = time.time()

# 种群演化结果
ga_solver.plot_population(population_records[-1], len(population_records)-1)
plt.show()

# 种群演化可视化 箱线图
ga_solver.plot_evolution(population_records, [0, len(population_records)], fig_size=(12, 3))
plt.show()

best_results = ga_solver.get_best_result(population_records[-1])

print('********种群演化结果********\n')
print('参数设定：')
print('DNA_SIZE:{}'.format(ga_solver.dna_size))
print('POPULATION SIZE:{}'.format(ga_solver.pop_size))
print('CROSSOVER_RATE:{}'.format(ga_solver.crossover_rate))
print('MUTATION_RATE:{}'.format(ga_solver.mutation_rate))
print('N_GENERATION:{}'.format(ga_solver.n_generations))
print('\n运行结果：')
print('最佳解：{:.2f}\n最佳函数值:{:.2f}'.format(best_results[0], best_results[0]))
print('运行时间：{:.2f}s'.format(end_time-start_time))