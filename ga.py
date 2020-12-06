"""
    author: zhanluo zhang
    date: 20201204
    description: an example of Genetic Algorithm
    reference:
        [Genetic Algorithm] https://blog.csdn.net/ha_ha_ha233/article/details/91364937
        [Matplotlib Animation] https://blog.csdn.net/briblue/article/details/84940997
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class GA:
    """
    遗传算法求解工具，可以对染色体长度、种群数量，交叉概率，遗传概率进行设定。
    一个简单的使用例子为：

    from ga import GA

    ga_solver = GA()
    records = ga_solver.revolution()
    best_x, best_y = ga_solver.get_best_result(records[-1])
    """

    def __init__(self, dna_size=15, population_size=100, crossover_rate=0.8, mutation_rate=0.003, n_generations=100):
        self.dna_size = dna_size
        self.pop_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.x_range = (0, 30)
        self.fitness_func = lambda _x: 80 * np.sin(1.5 * _x) + 60 * np.cos(_x) - _x ** 2 + 30 * _x
        self.fitness_func_latex = '$80sin(x)+60cos(x)-x^2+30x$'
        self.pic_path = 'Pics/'
        self.data_path = 'Data/'
        for path in [self.pic_path, self.data_path]:
            if not os.path.exists(path):
                os.mkdir(path)

    def translate_dna(self, pop):
        """
        将编码的DNA转换为对应的x值。

        :param pop: 种群所有个体的DNA编码。np.array: (POP_SIZE, DNA_SIZE)
        :return: 种群所有个体对应的x值。np.array: (POP_SIZE, 1)
        """
        # (POP_SIZE, DNA_SIZE)*(DNA_SIZE, 1) --> (POP_SIZE, 1) 完成解码
        return pop.dot(2 ** np.arange(self.dna_size)[::-1]) / float(2 ** self.dna_size - 1) * (
                self.x_range[1] - self.x_range[0]) + self.x_range[0]

    def get_fitness(self, pop):
        """
        计算适应度。

        :param pop: 种群所有个体的DNA编码。np.array: (POP_SIZE, DNA_SIZE)
        :return: 种群中所有个体的适应度。np.array: (POP_SIZE, 1)
        """
        pred = self.fitness_func(self.translate_dna(pop))
        # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]
        # 最后在加上一个很小的数防止出现为0的适应度，因为遗传算法并不会绝对否定某一个体
        pred = pred - np.min(pred) + 1e-3
        return pred

    def initial_pop(self):
        """
        种群初始化。

        :return: 种群所有个体的DNA编码。np.array: (POP_SIZE, DNA_SIZE)
        """
        return np.random.randint(2, size=(self.pop_size, self.dna_size))

    def select(self, pop, fit):
        """
        根据适应度进行自然选择，注意这里的选择概率是当前种群的相对概率，淘汰种群中表现最差的部分。

        :param pop:
        :param fit:
        :return:
        """
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p=fit / fit.sum())
        return pop[idx]

    def mutation(self, child):
        """
        DNA变异。

        :param child: 一个子代的DNA。np.array: (DNA_SIZE, 1)
        :return: 在某一点上变异后的子代DNA。np.array: (DNA_SIZE, 1)
        """
        if np.random.rand() < self.mutation_rate:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, self.dna_size)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转
        return child

    def crossover_and_mutation(self, pop):
        """
        DNA交叉和变异。

        :param pop: 种群所有个体的DNA编码。np.array: (POP_SIZE, DNA_SIZE)
        :return: 经过交叉和变异的所有个体的DNA编码。np.array: (POP_SIZE, DNA_SIZE)
        """
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < self.crossover_rate:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[np.random.randint(self.pop_size)]  # 在种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=self.dna_size)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            child = self.mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)
        return np.array(new_pop)

    def revolution(self):
        """
        演化过程。

        :return: 演化过程中的所有种群。list<-np.array: (POP_SIZE, DNA_SIZE)
        """
        pop_records = []
        # 初始化种群
        pop = self.initial_pop()
        pop_records.append(pop)
        # 演化
        for _ in range(self.n_generations):
            # 评估群体中个体的适应度
            fit = self.get_fitness(pop)
            # 选择
            pop = self.select(pop, fit)
            # 交叉和变异
            pop = self.crossover_and_mutation(pop)
            pop_records.append(pop)
        return pop_records

    def get_best_result(self, pop):
        """
        获取种群中最佳的个体的值及最佳结果。

        :param pop: 种群所有个体的DNA编码。np.array: (POP_SIZE, DNA_SIZE)
        :return: 最佳个体所代表的解的值及最佳结果。
        """
        target_function_values = self.fitness_func(self.translate_dna(pop)).tolist()
        best_idx = target_function_values.index(max(target_function_values))
        return self.translate_dna(pop[best_idx]), target_function_values[best_idx]

    def plot_population(self, pop, n_generation):
        """
        对种群进行可视化。

        :param pop: 种群所有个体的DNA编码。np.array: (POP_SIZE, DNA_SIZE)
        :param n_generation: int。
        :return: matplotlib ax。可以进一步使用plt.savefig等函数对图片进行编辑和保存。
        """
        xs = self.translate_dna(pop)
        x_range = np.linspace(self.x_range[0], self.x_range[-1], 500)
        plt.plot(x_range, self.fitness_func(x_range), c='black', label=self.fitness_func_latex)
        plt.plot(xs, self.fitness_func(xs), 'ro', markersize=5, label='Current Population')
        plt.title('Genetic Algorithm')
        plt.xlabel('x')
        plt.ylabel('Target Function')
        plt.text(30, 300, 'Generation: {}'.format(n_generation), c='red', fontsize=11, ha='right')
        plt.text(30, -170, 'https://github.com/zhangzhanluo/example-GA', fontsize=6, ha='right')
        plt.text(-1, 283, 'DNA Size: {}\nPopulation Size: {}\nCrossover Rate: {}\nMutation Rate: {}'.format(
            self.dna_size, self.pop_size, self.crossover_rate, self.mutation_rate
        ), fontsize=9)
        plt.legend(fontsize=9)
        return plt.gca()

    def plot_evolution(self, pops, generation_range):
        """
        对演化过程使用箱线图进行分析。

        :param pops: 种群的演化记录。list<-np.array: (POP_SIZE, DNA_SIZE)
        :param generation_range: 需要可视化的范围，不包括右边界。[start, end]
        :return: matplotlib ax。可以进一步使用plt.savefig等函数对图片进行编辑和保存。
        """
        fitness_records = [self.fitness_func(self.translate_dna(x)) for x in
                           pops[generation_range[0]: generation_range[1]]]
        plt.boxplot(fitness_records, labels=range(generation_range[0], generation_range[1]))
        plt.xlabel('Generation')
        plt.ylabel('Target Function Values')
        plt.text(plt.gca().get_xlim()[-1], plt.gca().get_ylim()[0],
                 'DNA Size: {}\nPopulation Size: {}\nCrossover Rate: {}\nMutation Rate: {}'.format(
                     self.dna_size, self.pop_size, self.crossover_rate, self.mutation_rate
                 ), ha='right', va='bottom')
        return plt.gca()


class GAAnimation(GA):
    """
    对遗传算法的种群记录使用动图可视化。继承了GA类的属性和方法。
    """

    def __init__(self, pops, dna_size=15, population_size=100, crossover_rate=0.8, mutation_rate=0.003):
        GA.__init__(self, dna_size, population_size, crossover_rate, mutation_rate)
        self.fig, self.ax = plt.subplots()
        self.ln = None
        self.text = None
        self.frames = list(enumerate(pops))

    def init(self):
        """
        动图初始化。

        :return: 需要动图中进行更新的artists。
        """
        x_range = np.linspace(self.x_range[0], self.x_range[-1], 500)
        self.ax.plot(x_range, self.fitness_func(x_range), c='black', label=ga.fitness_func_latex)
        self.ln, = self.ax.plot([], [], 'ro', markersize=5, animated=True, label='Current Population')
        self.text = self.ax.text(23, 300, '', animated=True, c='red', fontsize=11)
        plt.xlabel('x')
        plt.ylabel('Target Function')
        plt.title('Genetic Algorithm')
        self.ax.text(30, -170, 'https://github.com/zhangzhanluo/example-GA', fontsize=6, ha='right')
        self.ax.text(-1, 283, 'DNA Size: {}\nPopulation Size: {}\nCrossover Rate: {}\nMutation Rate: {}'.format(
            self.dna_size, self.pop_size, self.crossover_rate, self.mutation_rate
        ), fontsize=9)
        return self.ln, self.text,

    def update(self, frame):
        """
        更新图区内容。

        :param frame: int，帧数。
        :return: 需要动图中进行更新的artists。
        """
        n_iteration, pop = self.frames[frame]
        xs = self.translate_dna(pop)
        self.ln.set_data(xs, self.fitness_func(xs))
        self.text.set_text('Generation: {}'.format(n_iteration))
        return self.ln, self.text,

    def plot(self):
        """
        完成动图并保存。使用本方法需要保证电脑上已安装ImageMagick。ImageMagick下载地址：https://imagemagick.org/script/download.php
        对于Linux用户：sudo apt-get install imagemagick

        :return: 无返回值。
        """
        anim = animation.FuncAnimation(self.fig, self.update, frames=len(self.frames), interval=500,
                                       init_func=self.init, blit=True)
        plt.legend()
        anim.save(self.pic_path + 'Generation Algorithm Illustration.gif', writer='imagemagick', dpi=300)
        plt.close(self.fig)


if __name__ == '__main__':
    # 定义参数
    num_generations = 80
    ga = GA(dna_size=15,
            population_size=100,
            crossover_rate=0.8,
            mutation_rate=0.003,
            n_generations=num_generations)
    population_records = []
    # 初始化种群
    population = ga.initial_pop()
    population_records.append(population)
    # 演化100代
    for _ in range(100):
        # 评估群体中个体的适应度
        fitness = ga.get_fitness(population)
        # 选择
        population = ga.select(population, fitness)
        # 交叉和变异
        population = ga.crossover_and_mutation(population)
        population_records.append(population)

    # 对初始种群进行可视化
    n = 0
    _ = ga.plot_population(population_records[n], n)
    plt.show()

    # 对任一种群进行可视化
    n = np.random.randint(0, num_generations, 1)[0]
    _ = ga.plot_population(population_records[n], n)
    plt.show()

    # 对最终种群进行可视化
    n = num_generations
    _ = ga.plot_population(population_records[n], n)
    plt.show()

    # 使用箱线图对演化进行可视化。
    plt.figure(figsize=(15, 4))
    _ = ga.plot_evolution(population_records, [0, num_generations+1])
    plt.xticks(range(0, num_generations+1, 5), range(0, num_generations+1, 5))
    plt.show()

    # 使用动图对演化进行可视化，这一步比较耗费时间，想要快速得到结果可以将anim.save函数内的dpi调低。
    ga_animation = GAAnimation(pops=population_records)
    ga_animation.plot()
