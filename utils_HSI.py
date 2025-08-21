# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import itertools
import spectral
import matplotlib.pyplot as plt
from scipy import io
import imageio
import os
import re
import torch
import numpy as np
def get_device(ordinal):  #获取计算的设备
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device  #获取jis

def seed_worker(seed):  #用于设置随机数种子以确保实验的可重复性
    torch.manual_seed(seed)  #设置PyTorch的随机数种子。
    if torch.cuda.is_available():   #果CUDA可用，使用 torch.cuda.manual_seed(seed) 设置CUDA的随机数种子。
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.  #如果正在使用多GPU，使用 torch.cuda.manual_seed_all(seed) 设置所有GPU的随机数种子。
    np.random.seed(seed)  # Numpy module.   #设置NumPy的随机数种子。
    random.seed(seed)  # Python random module.   #设置Python标准库的随机数种子。
    torch.backends.cudnn.benchmark = False   #以确保每次运行时都使用相同的算法以获得相同的结果。
    torch.backends.cudnn.deterministic = True   #以确保每次运行时使用相同的输入，输出将保持一致。

def open_file(dataset):   #打开文件
    _, ext = os.path.splitext(dataset)   #函数将文件名拆分为基本名称和扩展名部分，并将结果存储在 _ 和 ext 变量中。
    ext = ext.lower()   #使用 ext.lower() 将扩展名部分转换为小写格式。
    if ext == '.mat':
        # Load Matlab array
        # matlab v5.0 files using "io.loadmat"
        # return io.loadmat(dataset)
        # Solve bug: NotImplementedError: Please use HDF reader for matlab v7.3 files, e.g. h5py
        import h5py
        return h5py.File(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def convert_to_color_(arr_2d, palette=None):   #将标签数组转换为RGB颜色编码的图像。

    #接受一个包含标签的二维数组 arr_2d 和一个颜色调色板 palette，该调色板是一个字典，将标签号映射到RGB元组
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)    #创建一个与输入数组相同大小的三维数组 arr_3d，其中的每个元素都用RGB值表示。
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():   #遍历调色板中的每个标签值，并将与当前标签值相匹配的数组元素设置为对应的RGB值
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d   #返回RGB编码的图像数组 arr_3d。


def convert_from_color_(arr_3d, palette=None):   #将RGB编码的图像转换为灰度标签数组。
    #接受一个包含RGB编码的三维图像数组 arr_3d 和一个颜色调色板 palette，该调色板是一个字典，将RGB元组映射到标签号。
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)   #它创建一个与输入图像大小相同的二维数组 arr_2d，用于存储灰度标签。

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i   #遍历调色板中的每个RGB颜色值，并将与当前颜色值匹配的像素设置为对应的标签号

    return arr_2d   #返回灰度标签数组 arr_2d。


def display_predictions(pred, vis, gt=None, caption=""):   #在Visdom中显示预测结果和（可选）地面真实标签。
    #pred: 预测结果的图像数组。
    #vis: Visdom客户端对象。
    #gt（可选）：地面真实标签的图像数组。
    #caption：显示在图像上方的标题。

    if gt is None:   #如果没有提供地面真实标签 gt，则它仅显示预测结果
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})   #否则，它会以两列的形式显示预测结果和地面真实标签。

def display_dataset(img, gt, bands, labels, palette, vis):  #用于显示指定的数据集，主要用于显示RGB图像。
    #img：3D高光谱图像。
    #gt：2D数组标签。
    #bands：选择的RGB波段的元组。
    #labels：标签类别名称列表。
    #palette：颜色字典。
    #vis：Visdom客户端对象。
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))  #印图像的尺寸和通道数
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')  #根据提供的RGB波段选择生成RGB图像

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})  #将RGB图像传输到Visdom服务器以进行显示。

def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):   #用于计算和打印模型的性能指标，包括准确率、混淆矩阵、各类别的 F1 分数等。
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels  预测的标签列表。
        target: list of target labels  目标域的标签列表。
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef   #要忽略的标签列表  例如未定义的标签
        n_classes (optional): number of classes, max(target) by default    #类别的数量，默认为 target 中的最大值加1。
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)  #根据 ignored_labels 将要忽略的标签对应的样本排除出去
    for l in ignored_labels:#对于 ignored_labels 中的每个标签 l，
        ignored_mask[target == l] = True   #找到 target 中等于该标签的位置，  并将对应的 ignored_mask 中的位置设为 True。
    ignored_mask = ~ignored_mask   #将 ignored_mask 取反
    #target = target[ignored_mask] -1
    # target = target[ignored_mask]
    # prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes   #根据 target 中的最大值确定类别数 n_classes，如果 n_classes 为 None，则将其设为 target 中最大值加一。

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))  #利用 target 和 prediction 计算混淆矩阵 cm，其中使用 labels 参数指定了类别的范围，从 0 到 n_classes-1。

    results["Confusion_matrix"] = cm   #通过对混淆矩阵 cm 进行计算，

    FP = cm.sum(axis=0) - np.diag(cm)     #假阳性（False Positive，FP）
    FN = cm.sum(axis=1) - np.diag(cm)    #假阴性（False Negative，FN）
    TP = np.diag(cm)  #真阳性（True Positive，TP）
    TN = cm.sum() - (FP + FN + TP)   #真阴性（True Negative，TN）

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)   #将这些数量转换为浮点数类型
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)   #真阳性率（TPR）
    results["TPR"] = TPR   #将真阳性率存储在 results 字典中，使用键 "TPR"。
    # Compute global accuracy
    total = np.sum(cm)   #通过 np.sum(cm) 计算了混淆矩阵中所有元素的总和，即总样本数。
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)   #通过遍历混淆矩阵的对角线元素（即预测正确的样本数），计算了正确分类的样本数，并将其乘以 100 除以总样本数，得到准确率。

    results["Accuracy"] = accuracy  #将准确率存储在 results 字典中，使用键 "Accuracy"。

    # Compute F1 score
    F1scores = np.zeros(len(cm))  #创建了一个长度为类别数的零数组 F1scores，用于存储每个类别的 F1 分数。
    for i in range(len(cm)):  #通过循环遍历每个类别：
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))   #计算该类别的 F1 分数
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1   #将计算得到的 F1 分数存储在 F1scores 数组的相应位置。

    results["F1_scores"] = F1scores   #将所有类别的 F1 分数存储在 results 字典中，使用键 "F1_scores"。

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)   #pa 是分类器的准确率
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)   #pe 是分类器与随机分类器达成一致的概率，
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa#将计算得到的 Kappa 系数存储在 results 字典中，使用键 "Kappa"。

    results["prediction"] = prediction
    results["label"] = target   #将预测值和真实标签存储在 results 字典中，分别使用键 "prediction" 和 "label"。

    return results   #返回结果


def show_results(results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1_scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion_matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion_matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1_scores"]
        kappa = results["Kappa"]

    #label_values = label_values[1:]
    vis.heatmap(cm, opts={'title': "Confusion_matrix", 
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    text += "Confusion_matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1_scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print(text)


def sample_gt(gt, train_size, mode='random'):    #从标签数组中提取固定百分比的样本的功能。
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)   #使用 np.nonzero() 函数获取标签数组 gt 中非零元素的索引，得到一个包含两个数组的元组，分别表示非零元素的行索引和列索引。
    X = list(zip(*indices)) # x,y features   #将行列索引合并成一个列表，得到了特征 X，每个元素是一个包含行列索引的元组。
    y = gt[indices].ravel() # classes    #根据非零元素的索引，从标签数组 gt 中提取对应位置的标签值，使用 ravel() 函数将二维数组展平成一维数组，得到了类别 y，其中每个元素表示一个样本的类别。
    train_gt = np.zeros_like(gt)    #根据原始标签数组 gt 的形状，创建一个全零数组作为训练集的标签数组 train_gt，用于存储训练集的标签。
    test_gt = np.zeros_like(gt)   #根据原始标签数组 gt 的形状，创建一个全零数组作为测试集的标签数组 test_gt，用于存储测试集的标签。
    if train_size > 1:   #如果 train_size 大于 1，则将其转换为整数类型。这里的 train_size 表示训练集的大小或训练集所占比例。
       train_size = int(train_size)
    train_label = []
    test_label = []   #初始化两个空列表，用于存储训练集和测试集的类别标签。
    if mode == 'random':
        if train_size == 1:   #若训练集大小为 1，则随机打乱样本的顺序，并将所有样本作为训练集，测试集为空。
            random.shuffle(X)    #随机打乱样本的顺序，即随机重排序列表 X 中的元素。
            train_indices = [list(t) for t in zip(*X)]
            #将数据 X 的行和列进行解包，并将结果存储到 train_indices 中。zip(*X) 解包的是 X 的不同维度。
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            #通过 gt[tuple(train_indices)] 从标注数据 gt 中获取与训练集索引对应的标签，并将这些标签追加到 train_label 列表中。
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            #将训练集的索引和标签进行列堆叠，生成完整的训练集数据（包括索引和对应的标签）。
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            #根据训练集索引从 gt 中提取标签，存储在 train_gt 中。
            test_gt = []   #初始化测试集标签数组为空。
            test_set = []   # 初始化测试集特征和标签合并的数组为空。


        else:    #若训练集大小不为 1，则使用 sklearn 中的 train_test_split() 函数将样本按照给定的比例分割成训练集和测试集，同时保持类别分布的一致性。
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
            #使用 train_test_split() 函数将样本按照给定的比例分割成训练集和测试集，并保持类别分布的一致性。其中，train_size 参数指定了训练集的大小或比例，stratify=y 表示按照类别 y 进行分层采样，random_state=23 表示随机种子。
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]   #通过 zip(*train_indices) 和 zip(*test_indices) 将划分后的训练集和测试集索引进行解包，并分别存储为列表形式。
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]    #根据训练集和测试集的索引，从 gt 中提取相应的标签，分别存储在 train_gt 和 test_gt 中。

            [train_label.append(i) for i in gt[tuple(train_indices)]]  #将训练集和测试集的标签值添加到相应的列表 train_label 和 test_label 中。
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))  #将训练集和测试集的特征和标签合并成数组 train_set 和 test_set。
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0],test_indices[1],test_label))

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt, train_set, test_set   #返回训练集标签数组  测试集标签数组  训练集特征和标签合并数组 测试集特征和标签合并数组


def sample_gt_fixed(gt, train_size_list, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)

    train_label = []
    test_label = []
    print("Sampling {} with train size = {}".format(mode, train_size_list))
    train_indices, test_indices = [], []
    train_label = []
    test_label = []
    for c in np.unique(gt):
        if c == 0:
            continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices))  # x,y features

        train, test = sklearn.model_selection.train_test_split(
            X, train_size=train_size_list[c-1], random_state=23)
        train_indices += train
        test_indices += test
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]

    [train_label.append(i) for i in gt[train_indices]]
    train_set = np.column_stack(
        (train_indices[0], train_indices[1], train_label))
    [test_label.append(i) for i in gt[test_indices]]
    test_set = np.column_stack((test_indices[0], test_indices[1], test_label))

    return train_gt, test_gt, train_set, test_set

def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients 
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()




class Averagvalue(object):   #用于计算和存储当前值和平均值
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):  #重置
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):  #更新
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad