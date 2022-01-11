import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import numpy as np
import copy


def handle_result(file_name, epochs, curve_num, curve_name_list, accuracy_lists, loss_lists):
    # 根据系统类别判断是直接画出结果图像还是将结果保存至txt文件
    if platform.system().lower() == 'windows':
        file_path = "C:/Users/4444/PycharmProjects/Multi_server_federated_learning/results/" + file_name + ".txt"
        save_result_to_txt(file_path, epochs, curve_num, curve_name_list, accuracy_lists, loss_lists)
        draw_curves(epochs, curve_num, curve_name_list, accuracy_lists, loss_lists)
    elif platform.system().lower() == 'linux':
        file_path = "/home/zrz/codes/Multi_server_federated_learning/results/" + file_name + ".txt"
        save_result_to_txt(file_path, epochs, curve_num, curve_name_list, accuracy_lists, loss_lists)


def draw_curves(epochs, curve_num, curve_name_list, accuracy_lists, loss_lists):
    # epochs_number = range(1, epochs + 1)  # Get number of epochs
    epochs_number = range(200, epochs + 200)  # Get number of epochs

    # 画accuracy曲线
    if len(accuracy_lists) > 0:
        accuracy_curve_names = []
        for cna in range(curve_num):
            # if cna == 0:
            #     color = '#1f77b4'
            # else:
            #     color = '#d62728'
            # plt.plot(epochs_number, accuracy_lists[cna], color=color)
            plt.plot(epochs_number, accuracy_lists[cna])
            accuracy_curve_names.append(str(curve_name_list[cna]))

        # plt.title('Accuracy curves')
        plt.title(' ')
        # plt.xlabel("Epochs")
        plt.xlabel("Communication rounds")
        plt.ylabel("Test Accuracy")
        plt.legend(accuracy_curve_names)

        plt.figure()

    # 画loss曲线
    if len(loss_lists) > 0:
        loss_curve_names = []
        for cnl in range(curve_num):
            plt.plot(epochs_number, loss_lists[cnl])
            loss_curve_names.append(str(curve_name_list[cnl]))
        # plt.title('Loss curves')
        plt.title(' ')
        # plt.xlabel("Epochs")
        plt.xlabel("Communication rounds")
        plt.ylabel("Test Loss")
        plt.legend(loss_curve_names)

        plt.figure()

    plt.show()


def draw_bars(data_lists, fig_name):
    # plt.style.use('ggplot')
    np.transpose(data_lists)  # 矩阵转置
    t_d_l = list(np.transpose(data_lists).tolist())  # 矩阵转list
    for i in range(len(t_d_l)):
        for j in range(len(t_d_l[i])):
            t_d_l[i][j] = list(reversed(t_d_l[i][j]))
    # y1 = [np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]
    # y2 = [np.array([1.22, 0.14, 0.12]), np.array([1.23, 0.125, 0.1]), np.array([2.65, 0.35, 0.28])]
    # y3 = [np.array([2.35, 0.09, 0.38]), np.array([0.24, 0.011, 0.05]), np.array([0, 0, 0])]
    # y4 = [np.array([1.35, 0.59, 0.38]), np.array([1.2, 0.314, 0.29]), np.array([0, 0, 0])]
    y1 = [np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]
    y2 = [np.array(t_d_l[0][0]), np.array(t_d_l[0][1]), np.array(t_d_l[0][2])]
    y3 = [np.array(t_d_l[1][0]), np.array(t_d_l[1][1]), np.array(t_d_l[1][2])]
    y4 = [np.array(t_d_l[2][0]), np.array(t_d_l[2][1]), np.array(t_d_l[2][2])]

    # labels = ['MNIST', 'FashionMNIST', 'CIFAR-10']
    labels = ['CIFAR-10\n + VGG13', 'FashionMNIST\n + FC3', 'MNIST\n + FC3']
    p = np.arange(len(labels))
    width = 0.4
    move = width / 2

    x_t = range(len(labels))
    plt.yticks(x_t, labels)
    x_max = 20
    plt.xticks(range(1, x_max, 2))
    plt.xlim(1, x_max)

    colors = ['white', '#ff7f0e', '#1f77b4']

    b_height = 0.18

    l1_1 = plt.barh(p - move, y1[0], height=b_height, color=colors[0])
    l2_1 = plt.barh(p, y1[1], height=b_height, color=colors[0])
    l3_1 = plt.barh(p + move, y1[2], height=b_height, color=colors[0])

    l1_2 = plt.barh(p - move, y2[0], height=b_height, left=y1[0], color=colors[1], label=' 0 < Misclustering rate < 1')
    l2_2 = plt.barh(p, y2[1], height=b_height, left=y1[1], color=colors[1])
    l3_2 = plt.barh(p + move, y2[2], height=b_height, left=y1[2], color=colors[1])

    l1_3 = plt.barh(p - move, y3[0], height=b_height, left=y1[0] + y2[0], color=colors[2], label='Misclustering rate = 0')
    l2_3 = plt.barh(p, y3[1], height=b_height, left=y1[1] + y2[1], color=colors[2])
    l3_3 = plt.barh(p + move, y3[2], height=b_height, left=y1[2] + y2[2], color=colors[2])

    l1_4 = plt.barh(p - move, y4[0], height=b_height, left=y1[0] + y2[0] + y3[0], color=colors[1])
    l2_4 = plt.barh(p, y4[1], height=b_height, left=y1[1] + y2[1] + y3[1], color=colors[1])
    l3_4 = plt.barh(p + move, y4[2], height=b_height, left=y1[2] + y2[2] + y3[2], color=colors[1])

    # 给条形图添加数据标注
    for i in range(len(l1_4)):
        height_x = l1_1[i].get_width() + l1_2[i].get_width() + l1_3[i].get_width() + l1_4[i].get_width()
        plt.text(height_x, l1_4[i].get_y() + l1_4[i].get_height() / 2, ' ι=1', fontsize=11, va="center", ha="left")

    for i in range(len(l2_4)):
        height_x = l2_1[i].get_width() + l2_2[i].get_width() + l2_3[i].get_width() + l2_4[i].get_width()
        plt.text(height_x, l2_4[i].get_y() + l2_4[i].get_height() / 2, ' ι=2', fontsize=11, va="center", ha="left")

    for i in range(len(l3_4)):
        height_x = l3_1[i].get_width() + l3_2[i].get_width() + l3_3[i].get_width() + l3_4[i].get_width()
        plt.text(height_x, l3_4[i].get_y() + l3_4[i].get_height() / 2, ' ι=3', fontsize=11, va="center", ha="left")

    # plt.title("DPC")  # 图片标题
    plt.xlabel("Cluster center selection threshold κ")  # x轴标题
    # plt.legend(loc=[0, 0])  # 图例的显示位置设置
    plt.legend()  # 图例的显示位置设置
    file_path = "C:/Users/4444/PycharmProjects/Multi_server_federated_learning/results/" + fig_name + ".png"
    plt.savefig(file_path, bbox_inches='tight')  # 保存图片命令一定要放在plt.show()前面
    plt.show()


def draw_gradual_bars(class_lists, fig_name, x_max):
    # 设置 x 轴坐标
    plt.xticks(range(1, x_max, 2))
    plt.xlim(1, x_max)
    plt.xlabel("Cluster center selection threshold κ")  # x轴标题

    # 设置 y 轴坐标
    y_labels = ['CIFAR-10\n + VGG13', 'FashionMNIST\n + FC3', 'MNIST\n + FC3']
    y_t = range(len(y_labels))
    plt.yticks(y_t, y_labels)

    height = 0.2  # 每个 bar 的粗度
    width = 0.01  # 每个单元条的长度
    bar_space = 0.02  # 每个 bar 之间的间距

    # 每个类别
    for c_i in range(len(class_lists)):
        first_y = c_i - ((height + bar_space) * int(len(class_lists[c_i]) / 2))  # 本类别中第一个 bar 的 y 坐标
        # 每个 bar
        for b_i in range(len(class_lists[c_i])):
            b_n = len(class_lists[c_i][b_i])  # 一个渐变条中的单元条的个数
            y_position = first_y + b_i * (height + bar_space)  # 该 bar 的 y 坐标
            bar_label_y = 0  # 标注的 y 坐标
            # 每个单元条
            for i in range(len(class_lists[c_i][b_i])):
                alpha = class_lists[c_i][b_i][i]  # 单元条的透明度 31, 119, 180
                # b = plt.barh(y_position, width=width, height=height, left=(1 + i * width), color='#1f77b4', alpha=1)
                b = plt.barh(y_position, width=width, height=height, left=(1 + i * width), color='#4C89E1', alpha=alpha)
                bar_label_y += b[0].get_width()
            # 给条形图添加数据标注
            plt.text(bar_label_y + 1, y_position, ' ι=' + str(b_i + 1), fontsize=11, va="center", ha="left")

    # 颜色标注
    # fig = plt.figure()
    # ax = plt.axes()
    # colors = []
    # bn = []
    # for i in range(100):
    #     colors.append((180, 119, 31, i * 0.01))
    #     bn.append(i * 0.01)
    # cmp = mpl.colors.ListedColormap(colors)
    # norm = mpl.colors.BoundaryNorm(bn, cmp.N)
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), ax=ax)

    # plt.legend(loc=[0, 0])  # 图例的显示位置设置
    plt.xlabel("Cluster center selection threshold κ")  # x轴标题
    # plt.legend()  # 图例的显示位置设置
    if platform.system().lower() == 'windows':
        file_path = "C:/Users/4444/PycharmProjects/Multi_server_federated_learning/results/" + fig_name + ".png"
    elif platform.system().lower() == 'linux':
        file_path = "/home/zrz/codes/Multi_server_federated_learning/results/" + fig_name + ".png"
    else:
        file_path = "C:/Users/4444/PycharmProjects/Multi_server_federated_learning/results/" + fig_name + ".png"
    # plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')  # 保存图片命令一定要放在plt.show()前面
    plt.show()


def save_result_to_txt(file_path, epochs, curve_num, curve_name_list, accuracy_lists, loss_lists):
    # c_time = datetime.datetime.now()
    # sub_name = str(c_time.month) + "_" + str(c_time.day) + "_" + str(c_time.hour) + "_" + str(c_time.minute)
    # sub_name = file_name
    # file_name = "/home/zrz/codes/Multi_server_federated_learning/results/result" + sub_name + ".txt"
    # file_path = "/home/zrz/codes/Multi_server_federated_learning/results/" + file_name + ".txt"
    r_file = open(file_path, 'w+')

    # 写入epochs
    r_file.write(str(epochs) + "\n")
    r_file.write("\n")
    # 写入curve_num
    r_file.write(str(curve_num) + "\n")
    r_file.write("\n")
    # 写入grouping_cycle_list
    for cng in range(curve_num):
        r_file.write(str(curve_name_list[cng]) + "\n")
    r_file.write("\n")
    # 写入accuracy_lists
    for cna in range(curve_num):
        for epoch in range(epochs):
            r_file.write(str(accuracy_lists[cna][epoch]) + "\n")
        r_file.write("\n")
    # 写入loss_lists
    for cnl in range(curve_num):
        for epoch in range(epochs):
            r_file.write(str(float(loss_lists[cnl][epoch])) + "\n")
        r_file.write("\n")

    r_file.close()


def draw_curves_from_txt(file_name):
    epochs = -1
    curve_num = -1
    grouping_cycle_list = []
    accuracy_lists = []
    loss_lists = []
    # r_file = open('C:/Users/4444/PycharmProjects/Multi_server_federated_learning/results/' + str(file_name) + '.txt',
    #               'r')
    r_file = open('C:/Users/4444/PycharmProjects/Multi_server_federated_learning/results/' + str(file_name), 'r')
    # r_file = open('C:/Users/4444/PycharmProjects/Multi_server_federated_learning/results/paper_imgs/' + str(file_name), 'r')
    # 读取epochs
    for line in r_file:
        if line == '\n':
            break
        epochs = int(line)

    # 读取curve_num
    for line in r_file:
        if line == '\n':
            break
        curve_num = int(line)

    # 读取grouping_cycle_list
    for line in r_file:
        if line == '\n':
            break
        grouping_cycle_list.append(str(line))

    # 读取accuracy_lists
    for cna in range(curve_num):
        accuracy_list = []
        for line in r_file:
            if line == '\n':
                break
            accuracy_list.append(float(line))
        accuracy_lists.append(accuracy_list)

    # 读取accuracy_lists
    for cnl in range(curve_num):
        loss_list = []
        for line in r_file:
            if line == '\n':
                break
            loss_list.append(float(line))
        loss_lists.append(loss_list)

    # 画出图像
    draw_curves(epochs, curve_num, grouping_cycle_list, accuracy_lists, loss_lists)

    first_index_list = find_first_target_in_lists(0.65, accuracy_lists)
    print("first_index_list:", first_index_list)
    curve_variance_list = calculate_variance_of_curves(accuracy_lists)
    print("curve_variance_list:", curve_variance_list)

    r_file.close()


def find_first_target_in_lists(target, lists):
    print("target =", target)
    first_index_list = []
    for curve_list in lists:
        window_list = copy.deepcopy(curve_list[0: 10])
        first_index = -1
        for i in range(10, len(curve_list)):
            window_list.pop(0)
            window_list.append(curve_list[i])
            average_value = np.mean(window_list)
            if float(average_value) >= float(target):
                first_index = i
                break
        first_index_list.append(first_index)
    return first_index_list


def calculate_variance_of_curves(curve_lists):
    curve_variance_list = []
    for curve_list in curve_lists:
        variance_list = []
        window_list = copy.deepcopy(curve_list[0: 10])
        for i in range(10, len(curve_list)):
            window_list.pop(0)
            window_list.append(curve_list[i])
            window_variance = np.var(window_list)
            variance_list.append(window_variance)
        average_variance = np.mean(variance_list)
        curve_variance_list.append(average_variance)
    return curve_variance_list
