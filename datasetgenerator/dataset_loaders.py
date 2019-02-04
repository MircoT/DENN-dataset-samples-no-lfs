"""Module container of loader utilities.

The namedtuple SetObj and SetComponent are used to compose datasets.

SetObj is a container of SetComponent and SetComponent contains
data and labels.

    SetObj -> + train (SetComponent obj)
              + validation (SetComponent obj)
              + test (SetComponent obj)

Validation and test could be None. By defaults all values are None.

    SetComponent -> + data
                    + labels

"""
from __future__ import print_function

import csv
import gzip
import io
import struct
import sys
import zipfile
from collections import namedtuple
from os import devnull as DEVNULL
from os import path
from pathlib import PurePosixPath

import numpy as np
from tqdm import tqdm

from PIL import Image

__all__ = ['DatasetLoader', 'TestDataset', 'IrisLoader',
           'LetterLoader', 'MNISTLoader',
           'BankLoader', 'MagicLoader',
           'CoilLoader', 'CoilWithResizeLoader',
           'QsarLoader', 'MNISTLoader_01_10_Norm',
           "GeccoFirstDomain", "GasSensorArrayDrift",
           'SetObj', 'SetComponent', 'HeartDisease',
           'PimaIndiansDiabetes', 'WisconsinDiagnosticBreastCancer']


SetObj = namedtuple('SetObj', ['train', 'validation', 'test'])
##
# No better way to set default namedtuple values:
# https://mail.python.org/pipermail/python-ideas/2015-July/034637.html
SetObj.__new__.__defaults__ = (None, None, None)

SetComponent = namedtuple('SetComponent', ['data', 'labels'])


def dataset_loader_check(func):
    """Decorator to check the output of the loader class."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        assert isinstance(result, SetObj), "Function does not return a dataset"
        for idx, elm in enumerate(result):
            assert isinstance(
                elm, SetComponent) or\
                elm is None, "elm {} is not a proper dataset type".format(elm)
        return result
    return wrapper


class DatasetLoader(object):

    """Base class to construct a loader.

    User can simply inherit this class and implement the method load.

    Note:
        Method load have to return a SetObj instance, that is composed of
        SetComponent elements or None elements.

        For example:
            return SetObj(
                SetComponent(my_data, my_labels),  # train
                None,  # validation
                None   # test
            )

    """

    def __init__(self):
        self.classifier_labels = True

    def load(self):
        raise NotImplementedError("Please implement the load method!")

    @dataset_loader_check
    def real_load(self, *args, **kwargs):
        return self.load(*args, **kwargs)


def check_min_max(rows):
    print("++ Check min and max")
    with tqdm(total=len(rows)) as pbar:
        min_max = {}
        for row in rows:
            for key, value in row.items():
                try:
                    tmp = float(value)
                    if key not in min_max:
                        min_max[key] = [0, 0]
                    if tmp > min_max[key][1]:
                        min_max[key][1] = tmp
                    elif tmp < min_max[key][0]:
                        min_max[key][0] = tmp
                except ValueError:
                    pass
            pbar.update(1)

    return min_max


class TestDataset(DatasetLoader):

    def __init__(self):
        super(__class__, self).__init__()

    @staticmethod
    def gen_data(list_data, list_labels, size, n_features, n_classes):
        for idx in range(size):
            list_data.append([idx for _ in range(n_features)])
            list_labels.append([idx for _ in range(n_classes)])

    def load(self, base_path=None):
        train_data = []
        train_labels = []
        validation_data = []
        validation_labels = []
        test_data = []
        test_labels = []

        train_size = 1000
        validation_size = 200
        test_size = 400
        n_features = 3
        n_classes = 2

        self.gen_data(train_data, train_labels, train_size, n_features, n_classes)
        self.gen_data(validation_data, validation_labels, validation_size, n_features, n_classes)
        self.gen_data(test_data, test_labels, test_size, n_features, n_classes)

        return SetObj(
            SetComponent(np.array(train_data), np.array(train_labels)),
            SetComponent(np.array(validation_data),
                         np.array(validation_labels)),
            SetComponent(np.array(test_data), np.array(test_labels)),
        )


class GasSensorArrayDrift(DatasetLoader):

    def __init__(self):
        super(__class__, self).__init__()

    def load(self, base_path):
        data = []
        labels = []

        mins = np.zeros(128)
        maxs = np.zeros(128)

        print("+ Open zip file")
        with zipfile.ZipFile(path.join(base_path, 'GasSensorArrayDrift.zip'), 'r') as gas_zip:
            print("+ Extract Data")
            for batch_file in gas_zip.namelist():
                if batch_file.find(".dat") != -1:
                    cur_batch = gas_zip.read(batch_file).decode("utf-8")
                    for line in cur_batch.split("\n"):
                        tmp = line.strip().split(" ")
                        if len(tmp) > 1:
                            class_ = int(tmp[0]) - 1
                            cur_data = [float(feature.split(":")[1])
                                        for feature in tmp[1:]]
                            class_ = int(tmp[0]) - 1
                            cur_data = [float(feature.split(":")[1])
                                        for feature in tmp[1:]]
                            ##
                            # Min max
                            for idx, val in enumerate(cur_data):
                                if val < mins[idx]:
                                    mins[idx] = val
                                if val > maxs[idx]:
                                    maxs[idx] = val
                            if len(cur_data) != 128:
                                raise Exception(
                                    "Error on extract gas features")
                            data.append(np.array(cur_data))
                            labels.append(np.zeros(6))
                            labels[-1][class_] = 1
                            # print("{} -> {}".format(class_, labels[-1]))

        for record in data:
            for idx, val in enumerate(record):
                record[idx] = (val - mins[idx]) / (maxs[idx] - mins[idx])

        return SetObj(
            SetComponent(np.array(data), np.array(labels)),
            None,
            None
        )


class GeccoFirstDomain(DatasetLoader):

    """Gecco first domain loader.

    f(x) = (sin(5*x(3*y+1))+1) / 2

    """

    def __init__(self):
        super(__class__, self).__init__()
        self.classifier_labels = False

    @staticmethod
    def random_range(min_, max_):
        min_ = np.ceil(min_)
        max_ = np.floor(max_)
        return np.random.random() * (max_ - min_) + min_

    @staticmethod
    def __get_x_y_b(size, random=False):
        if not random:
            x_values = np.arange(-1, 1, 2. / np.sqrt(size))
            y_values = np.arange(-1, 1, 2. / np.sqrt(size))
        else:
            x_values = [
                GeccoFirstDomain.random_range(-1., 1.) for _ in range(int(np.sqrt(size)))]
            y_values = [
                GeccoFirstDomain.random_range(-1., 1.) for _ in range(int(np.sqrt(size)))]
        tuples = []
        for x in x_values:
            for y in y_values:
                tuples.append((x, y, 1.))

        return np.array(tuples)

    @staticmethod
    def __gen_labels(tuples):
        tmp = []
        for x, y, b in tuples:
            tmp.append((np.sin(5. * x * (3 * y + 1)) + 1.) / 2.)
        return tmp

    def load(self, base_path, duration=False):
        train_values = self.__get_x_y_b(800)
        train_labels = self.__gen_labels(train_values)

        # print(len(train_values))

        validation_values = self.__get_x_y_b(80)
        validation_labels = self.__gen_labels(validation_values)

        # print(len(validation_values))

        test_values = self.__get_x_y_b(160, True)
        test_labels = self.__gen_labels(test_values)

        # print(len(test_values))

        return SetObj(
            SetComponent(np.array(train_values), np.array(train_labels)),
            SetComponent(np.array(validation_values),
                         np.array(validation_labels)),
            SetComponent(np.array(test_values), np.array(test_labels)),
        )


class CoilLoader(DatasetLoader):

    """Coil loader."""

    def load(self, base_path, duration=False):
        data = []
        labels = []

        print("+ Open zip file")
        with zipfile.ZipFile(
                path.join(base_path, 'coil-20-proc.zip'), 'r') as coil_zip:
            for file_ in coil_zip.namelist():
                if path.splitext(file_)[-1] == ".png":
                    print("++ Insert file {}".format(file_), end="\r")
                    with coil_zip.open(file_) as cur_image_file:
                        image = Image.open(cur_image_file)
                    image = np.array(image, dtype=np.float64).ravel()
                    image /= 255.
                    data.append(image)
                    class_ = int(path.splitext(path.basename(file_))[
                                 0].split("__")[0].replace("obj", ""))
                    one_hot = np.full((20, ), 0.)
                    one_hot[class_ - 1] = 1
                    labels.append(one_hot)

        return SetObj(
            SetComponent(np.array(data), np.array(labels)),
            None,
            None
        )


class CoilWithResizeLoader(DatasetLoader):

    """Coil loader with resize to 28x28."""

    def load(self, base_path, duration=False):
        data = []
        labels = []

        print("+ Open zip file")
        with zipfile.ZipFile(
                path.join(base_path, 'coil-20-proc.zip'), 'r') as coil_zip:
            for file_ in coil_zip.namelist():
                if path.splitext(file_)[-1] == ".png":
                    print("++ Insert file {}".format(file_), end="\r")
                    with coil_zip.open(file_) as cur_image_file:
                        image = Image.open(cur_image_file)
                    image = image.resize((28, 28), Image.ANTIALIAS)
                    image = np.array(image, dtype=np.float64).ravel()
                    image /= 255.
                    data.append(image)
                    class_ = int(path.splitext(path.basename(file_))[
                                 0].split("__")[0].replace("obj", ""))
                    one_hot = np.full((20, ), 0.)
                    one_hot[class_ - 1] = 1
                    labels.append(one_hot)

        return SetObj(
            SetComponent(np.array(data), np.array(labels)),
            None,
            None
        )


class QsarLoader(DatasetLoader):

    """Qsar loader."""

    def load(self, base_path):

        data = []
        labels = []

        attributes = [
            'SpMax_L',
            'J_Dz',
            'nHM',
            'F01',
            'F04',
            'NssssC',
            'nCb',
            'C',
            'nCp',
            'nO',
            'F03',
            'SdssC',
            'from',
            'LOC',
            'SM6_L',
            'F03',
            'Me',
            'Mi',
            'nN',
            'nArNO2',
            'nCRX3',
            'SpPosA_B',
            'nCIR',
            'B01',
            'B03',
            'N',
            'SpMax_A',
            'Psi_i_1d',
            'B04',
            'SdO',
            'TI2_L',
            'nCrt',
            'C',
            'F02',
            'nHDon',
            'SpMax_B',
            'Psi_i_A',
            'nN',
            'SM6_B',
            'nArCOOR',
            'nX',
            'class'
        ]

        print("+ Open CSV")
        with open(path.join(base_path, "biodeg.csv"), "rb") as csvfile:
            file_wrapper = io.TextIOWrapper(csvfile, newline='')
            reader = csv.DictReader(file_wrapper,
                                    delimiter=';',
                                    fieldnames=attributes
                                    )
            rows = list(reader)

            min_max = check_min_max(rows)

            print("++ Read all data")
            with tqdm(total=len(rows)) as pbar:
                for row in rows:
                    new_data = []
                    for attr in attributes[:-1]:
                        tmp = float(row[attr])
                        tmp = (
                            tmp - min_max[attr][0]) / (
                            min_max[attr][1] - min_max[attr][0]
                        )
                        new_data.append(tmp)
                    data.append(np.array(new_data))
                    if row['class'] == 'RB':
                        labels.append(np.array([0, 1]))
                    else:
                        labels.append(np.array([1, 0]))
                    pbar.update(1)

        data = np.array(data)
        labels = np.array(labels)

        return SetObj(
            SetComponent(data, labels)
        )


class MagicLoader(DatasetLoader):

    """Magic loader."""

    def load(self, base_path):

        data = []
        labels = []

        attributes = [
            'fLength',
            'fWidth',
            'fSize',
            'fConc',
            'fConc1',
            'fAsym',
            'fM3Long',
            'fM3Trans',
            'fAlpha',
            'fDist',
            'class'
        ]

        print("+ Open CSV")
        with open(path.join(base_path, "magic04.data"), "rb") as csvfile:
            file_wrapper = io.TextIOWrapper(csvfile, newline='')
            reader = csv.DictReader(file_wrapper,
                                    delimiter=',',
                                    fieldnames=attributes
                                    )
            rows = list(reader)

            min_max = check_min_max(rows)

            print("++ Read all data")
            with tqdm(total=len(rows)) as pbar:
                for row in rows:
                    new_data = []
                    for attr in attributes[:-1]:
                        tmp = float(row[attr])
                        tmp = (
                            tmp - min_max[attr][0]) / (
                            min_max[attr][1] - min_max[attr][0]
                        )
                        new_data.append(tmp)
                    data.append(np.array(new_data))
                    if row['class'] == 'g':
                        labels.append(np.array([0, 1]))
                    else:
                        labels.append(np.array([1, 0]))
                    pbar.update(1)

        data = np.array(data)
        labels = np.array(labels)

        return SetObj(
            SetComponent(data, labels)
        )


class BankLoader(DatasetLoader):

    """Back loader."""

    def __init__(self):
        super(self.__class__, self).__init__()
        self.__attributes = [
            'age',
            'job',
            'marital',
            'education',
            'default',
            'housing',
            'loan',
            'contact',
            'month',
            'day_of_week',
            # 'duration',
            'campaign',
            'pdays',
            'previous',
            'poutcome',
            'emp.var.rate',
            'cons.price.idx',
            'cons.conf.idx',
            'euribor3m',
            'nr.employed'
        ]

    def load(self, base_path, duration=False):
        from . bank_tables import get_bank_map_attributes

        attr_ids, attr_names = get_bank_map_attributes()

        attributes = self.__attributes

        if duration:
            attributes.insert(9, 'duration')

        print("+ Open zip file")
        with zipfile.ZipFile(
                path.join(base_path, 'bank-additional.zip'), 'r') as bank_zip:
            print("+ Open CSV TRAIN file")
            with bank_zip.open(
                str(PurePosixPath(
                    'bank-additional', 'bank-additional-full.csv'
                )),
                'r'
            ) as csvfile:

                file_wrapper = io.TextIOWrapper(csvfile, newline='')
                reader = csv.DictReader(file_wrapper, delimiter=';')
                rows = list(reader)

                min_max = check_min_max(rows)

                train_data, train_labels = self.get_data(
                    rows, attributes, attr_names, min_max)

            print("+ Open CSV TEST file")
            with bank_zip.open(
                str(PurePosixPath(
                    'bank-additional', 'bank-additional.csv'
                )),
                'r'
            ) as csvfile:

                file_wrapper = io.TextIOWrapper(csvfile, newline='')
                reader = csv.DictReader(file_wrapper, delimiter=';')
                rows = list(reader)

                test_data, test_labels = self.get_data(
                    rows, attributes, attr_names, min_max)

        return SetObj(
            SetComponent(train_data, train_labels),
            None,
            SetComponent(test_data, test_labels)
        )

    @staticmethod
    def get_data(rows, attributes, attr_names, min_max):
        """Get data from csv."""
        data = []
        labels = []

        print("++ Read all data")
        with tqdm(total=len(rows)) as pbar:
            for row in rows:
                new_data = []
                for attr in attributes:
                    if attr in attr_names:
                        new_data.append(attr_names[attr][row[attr]])
                    elif attr in min_max:
                        tmp = float(row[attr])
                        tmp = (
                            tmp - min_max[attr][0]) / (
                            min_max[attr][1] - min_max[attr][0]
                        )
                        new_data.append(tmp)
                    else:
                        raise Exception(
                            "Not implemented {} attribute".format(
                                attr)
                        )
                data.append(np.array(new_data))
                if row['y'] == 'yes':
                    labels.append(np.array([0, 1]))
                else:
                    labels.append(np.array([1, 0]))
                pbar.update(1)
                # print(row)
                # print(new_data)

        data = np.array(data)
        labels = np.array(labels)

        return data, labels


class MNISTLoader(DatasetLoader):

    """MNIST loader for DENN."""

    def load(self, base_path, output=False):
        if not output:
            old_descriptor = sys.stdout
            sys.stdout = open(DEVNULL, 'w')

        print('+ Load data ...')
        mnist = SetObj(
            SetComponent(
                self.load_mnist_image(path.join(
                    base_path, 'train-images-idx3-ubyte.gz')),
                self.load_mnist_label(path.join(
                    base_path, 'train-labels-idx1-ubyte.gz'))
            ),
            None,
            SetComponent(
                self.load_mnist_image(path.join(
                    base_path, 't10k-images-idx3-ubyte.gz')),
                self.load_mnist_label(path.join(
                    base_path, 't10k-labels-idx1-ubyte.gz'))
            )
        )

        print('+ loading done!')

        if not output:
            sys.stdout = old_descriptor

        return mnist

    @staticmethod
    def load_mnist_image(file_name):
        with gzip.open(file_name, 'rb') as mnist_file:
            magic_num, num_images, rows, cols = struct.unpack(
                ">IIII", mnist_file.read(16))
            buf = mnist_file.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = np.divide(data, 255.)
            data = data.reshape(num_images, rows * cols)
            return data

    @staticmethod
    def load_mnist_label(file_name, num_classes=10):
        with gzip.open(file_name, 'rb') as mnist_file:
            magic_num, num_labels = struct.unpack(
                ">II", mnist_file.read(8))
            buf = mnist_file.read(num_labels)
            labels = np.frombuffer(buf, dtype=np.uint8)
            data = np.zeros((num_labels, num_classes))
            data[np.arange(labels.size), labels] = 1
            return data


class MNISTLoader_01_10_Norm(DatasetLoader):

    """MNIST loader for DENN with a different normalization."""

    def load(self, base_path, output=False):
        if not output:
            old_descriptor = sys.stdout
            sys.stdout = open(DEVNULL, 'w')

        print('+ Load data ...')
        mnist = SetObj(
            SetComponent(
                self.load_mnist_image(path.join(
                    base_path, 'train-images-idx3-ubyte.gz')),
                self.load_mnist_label(path.join(
                    base_path, 'train-labels-idx1-ubyte.gz'))
            ),
            None,
            SetComponent(
                self.load_mnist_image(path.join(
                    base_path, 't10k-images-idx3-ubyte.gz')),
                self.load_mnist_label(path.join(
                    base_path, 't10k-labels-idx1-ubyte.gz'))
            )
        )

        print('+ loading done!')

        if not output:
            sys.stdout = old_descriptor

        return mnist

    @staticmethod
    def load_mnist_image(file_name):
        with gzip.open(file_name, 'rb') as mnist_file:
            magic_num, num_images, rows, cols = struct.unpack(
                ">IIII", mnist_file.read(16))
            buf = mnist_file.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data / 255.
            data = data * 0.9
            data = data + 0.1
            data = data.reshape(num_images, rows * cols)
            return data

    @staticmethod
    def load_mnist_label(file_name, num_classes=10):
        with gzip.open(file_name, 'rb') as mnist_file:
            magic_num, num_labels = struct.unpack(
                ">II", mnist_file.read(8))
            buf = mnist_file.read(num_labels)
            labels = np.frombuffer(buf, dtype=np.uint8)
            data = np.zeros((num_labels, num_classes))
            data[np.arange(labels.size), labels] = 1
            return data


class LetterLoader(DatasetLoader):

    """Letters loader for DENN."""

    def load(self, path, output=False):
        if not output:
            old_descriptor = sys.stdout
            sys.stdout = open(DEVNULL, 'w')

        letter_data = []
        label_data = []

        labels = []
        features_max = []

        print('+ Load data ...')
        with open(path + '.data', 'r') as iris_file:
            for line in iris_file.readlines():
                cur_line = [elm.strip() for elm in line.split(',')]

                if len(cur_line) == 17:
                    cur_label = cur_line[0]
                    if cur_label not in labels:
                        labels.append(cur_label)

                    label_data.append(labels.index(cur_label))

                    features = [float(elm) for elm in cur_line[1:]]
                    if len(features_max) == 0:
                        features_max = [elm for elm in features]
                    else:
                        for idx, feature in enumerate(features):
                            if features_max[idx] < feature:
                                features_max[idx] = feature

                    letter_data.append(features)

        features_max = np.array(features_max, np.float64)
        letter_data = np.divide(
            np.array(letter_data, np.float64), features_max)
        ##
        # expand labels (one hot vector)
        tmp = np.zeros((len(label_data), len(labels)))
        tmp[np.arange(len(label_data)), label_data] = 1
        label_data = tmp

        print('+ letters: \n', letter_data)
        print('+ labels: \n', label_data)

        print('+ loading done!')

        if not output:
            sys.stdout = old_descriptor

        return SetObj(
            SetComponent(letter_data, label_data)
        )


class IrisLoader(DatasetLoader):

    """Iris loader for DENN."""

    def load(self, path, output=False):
        if not output:
            old_descriptor = sys.stdout
            sys.stdout = open(DEVNULL, 'w')

        flower_data = []
        label_data = []

        labels = []
        features_max = []

        print('+ Load data ...')
        with open(path + '.data', 'r') as iris_file:
            for line in iris_file.readlines():
                cur_line = [elm.strip() for elm in line.split(',')]

                if len(cur_line) == 5:
                    cur_label = cur_line[-1]
                    if cur_label not in labels:
                        labels.append(cur_label)

                    label_data.append(labels.index(cur_label))

                    features = [float(elm) for elm in cur_line[:-1]]
                    if len(features_max) == 0:
                        features_max = [elm for elm in features]
                    else:
                        for idx, feature in enumerate(features):
                            if features_max[idx] < feature:
                                features_max[idx] = feature

                    flower_data.append(features)

        features_max = np.array(features_max, np.float64)

        flower_data = np.divide(
            np.array(flower_data, np.float64), features_max)
        ##
        # expand labels (one hot vector)
        tmp = np.zeros((len(label_data), len(labels)))
        tmp[np.arange(len(label_data)), label_data] = 1
        label_data = tmp

        print('+ flowers: \n', flower_data)
        print('+ labels: \n', label_data)

        print('+ loading done!')

        if not output:
            sys.stdout = old_descriptor

        return SetObj(
            SetComponent(flower_data, label_data)
        )


class HeartDisease(DatasetLoader):

    """Heart Disease loader for DENN."""

    def load_dataset(self, filepath, separator=',', info_classes=(lambda x: int(x), 5)):
        f_data = []
        l_data = []

        with open(filepath, 'r') as curr_file:
            for line in curr_file.readlines():
                values = line.split(separator)
                values = [float(x.strip().replace('?', '-9')) for x in values]
                f_data.append(np.array(values[:-1]))
                labels = np.zeros((info_classes[1],))
                labels[info_classes[0](values[-1])] = 1
                l_data.append(labels)
        return (np.array(f_data), np.array(l_data))

    def load(self, dir_path, output=False):
        datasets = [
            # in the peper, they used only this
            ("processed.cleveland.data.txt", ',', (lambda x: int(x > 0), 2)),
            #    ("processed.hungarian.data.txt", ',', (lambda x: int(x>0),2)),
            #    ("processed.switzerland.data.txt", ',', (lambda x: int(x>0),2)),
            #    ("processed.va.data.txt", ',', (lambda x: int(x>0),2)),
        ]
        all_f_data = None
        all_l_data = None
        for dataset in datasets:
            filepath = path.join(dir_path, dataset[0])
            f_data, l_data = self.load_dataset(
                filepath, dataset[1], dataset[2])
            all_f_data = np.concatenate(
                (all_f_data, f_data)) if all_f_data is not None else f_data
            all_l_data = np.concatenate(
                (all_l_data, l_data)) if all_l_data is not None else l_data

        return SetObj(
            SetComponent(np.array(all_f_data), np.array(all_l_data)),
            None,
            None
        )


class PimaIndiansDiabetes(DatasetLoader):

    """Pima Indians Diabetes  loader for DENN."""

    def load(self, dir_path, output=False):
        f_data = []
        l_data = []

        filepath = path.join(dir_path, "pima-indians-diabetes.data.csv")
        with open(filepath, 'r') as curr_file:
            for line in curr_file.readlines():
                values = line.split(',')
                f_data.append(np.array([float(val) for val in values[:-1]]))
                label = np.array([0, 0])
                label[int(values[-1])] = 1
                l_data.append(label)

        return SetObj(
            SetComponent(np.array(f_data), np.array(l_data)),
            None,
            None
        )


class WisconsinDiagnosticBreastCancer(DatasetLoader):

    """Wisconsin Diagnostic Breast Cancer  loader for DENN."""

    def load(self, dir_path, output=False):
        f_data = []
        l_data = []

        filepath = path.join(dir_path, "wdbc.data.txt")
        with open(filepath, 'r') as curr_file:
            for line in curr_file.readlines():
                values = line.split(',')
                f_data.append(np.array([float(val) for val in values[2:]]))
                label = np.array(
                    [1, 0]) if values[1] == "M" else np.array([0, 1])
                l_data.append(label)

        return SetObj(
            SetComponent(np.array(f_data), np.array(l_data)),
            None,
            None
        )
