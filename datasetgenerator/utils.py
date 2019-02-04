import numpy as np
import struct
import gzip
import binascii
import math
from tqdm import tqdm
from os import path
from os import makedirs
from collections import namedtuple
from os import SEEK_CUR

__all__ = ['new', 'new_dataset', 'Dataset']

BASEDATASETPATH = './'
Batch = namedtuple('Batch', ['data', 'labels'])


class Dataset(object):

    def __init__(self, file_name):

        self.__file_name = file_name

        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            self.stats = Header(
                "<H5if3I",
                [
                    "version",
                    "n_batch",
                    "n_features",
                    "n_classes",
                    "type",
                    "seed",
                    "train_percentage",
                    "test_offset",
                    "validation_offset",
                    "train_offset"
                ],
                gz_file.read(38)
            )

        # print(self.stats)
        self.type = 'double' if self.stats.type == 2 else 'float'
        self.__dtype = np.float64 if self.stats.type == 2 else np.float32
        self.__elm_size = 8 if self.stats.type == 2 else 4
        self.__size_elm_data = self.stats.n_features * self.__elm_size
        self.__size_elm_label = self.stats.n_classes * self.__elm_size

    def __read_from(self, offset, type_):
        """Read data from offset.

        Args:
            offset: num of bytes to jump
            type: 0 if data, 1 if label
        """
        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            gz_file.seek(offset)
            num_elms = struct.unpack("<I", gz_file.read(4))[0]
            if type_ == 1:
                gz_file.seek(num_elms * self.__size_elm_data, SEEK_CUR)
                data = np.frombuffer(gz_file.read(
                    num_elms * self.__size_elm_label
                ), dtype=self.__dtype)
            else:
                data = np.frombuffer(gz_file.read(
                    num_elms * self.__size_elm_data
                ), dtype=self.__dtype)
            data = data.reshape([
                num_elms,
                self.stats.n_features if type_ == 0 else self.stats.n_classes
            ])
        return data

    @property
    def test_data(self):
        return self.__read_from(self.stats.test_offset, 0)

    @property
    def test_labels(self):
        return self.__read_from(self.stats.test_offset, 1)

    @property
    def validation_data(self):
        return self.__read_from(self.stats.validation_offset, 0)

    @property
    def validation_labels(self):
        return self.__read_from(self.stats.validation_offset, 1)

    @property
    def num_batches(self):
        return self.stats.n_batch

    def batches(self):
        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            gz_file.seek(self.stats.train_offset)

            if self.stats.version == 1:
                for idx in range(self.num_batches):
                    num_batch = struct.unpack("<I", gz_file.read(4))[0]
                    num_elms = struct.unpack("<I", gz_file.read(4))[0]

                    data = np.frombuffer(
                        gz_file.read(num_elms * self.__size_elm_data),
                        dtype=self.__dtype
                    )
                    data = data.reshape([num_elms, self.stats.n_features])

                    labels = np.frombuffer(
                        gz_file.read(num_elms * self.__size_elm_label),
                        dtype=self.__dtype
                    )
                    labels = labels.reshape([num_elms, self.stats.n_classes])

                    yield Batch(data, labels)
            elif self.stats.version == 2:
                for idx in range(self.num_batches):
                    data = [np.frombuffer(
                        gz_file.read(self.__size_elm_data),
                        dtype=self.__dtype
                    )]

                    labels = [np.frombuffer(
                        gz_file.read(self.__size_elm_label),
                        dtype=self.__dtype
                    )]

                    yield Batch(np.array(data), np.array(labels))
            else:
                raise Exception(
                    "Unknown version '{}'".format(self.stats.version))

    def __getitem__(self, index):
        if index > self.num_batches - 1:
            index %= self.num_batches

        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            gz_file.seek(self.stats.train_offset)

            if self.stats.version == 1:
                for idx in range(self.num_batches):
                    num_batch = struct.unpack("<I", gz_file.read(4))[0]
                    num_elms = struct.unpack("<I", gz_file.read(4))[0]

                    # print('Read item ->', num_batch, num_elms)

                    if num_batch == index:
                        break
                    else:
                        gz_file.seek(
                            num_elms * self.__size_elm_data +
                            num_elms * self.__size_elm_label, SEEK_CUR)

                # print('Read item ->', num_elms, self.__size_elm_data)
                data = np.frombuffer(
                    gz_file.read(num_elms * self.__size_elm_data),
                    dtype=self.__dtype
                )
                # print('Read item ->', data.shape)
                data = data.reshape([num_elms, self.stats.n_features])
                # print('Read item ->', data.shape)

                labels = np.frombuffer(
                    gz_file.read(num_elms * self.__size_elm_label),
                    dtype=self.__dtype
                )
                # print('Read item ->', labels.shape)
                labels = labels.reshape([num_elms, self.stats.n_classes])
                # print('Read item ->', labels.shape)

                # print(data[0])
                # print(labels[0])
            elif self.stats.version == 2:
                for idx in range(self.num_batches):
                    if idx == index:
                        break
                    else:
                        gz_file.seek(self.__size_elm_data +
                                     self.__size_elm_label, SEEK_CUR)

                # print('Read item ->', num_elms, self.__size_elm_data)
                data = np.array([np.frombuffer(
                    gz_file.read(self.__size_elm_data),
                    dtype=self.__dtype
                )])
                # print('Read item ->', data.shape)

                labels = np.array([np.frombuffer(
                    gz_file.read(self.__size_elm_label),
                    dtype=self.__dtype
                )])
                # print('Read item ->', labels.shape)

                # print(data[0])
                # print(labels[0])
            else:
                raise Exception(
                    "Unknown version '{}'".format(self.stats.version))

        return Batch(data, labels)


class Header(object):

    """Manage with a simple interface binary headers.

    Example:

        header = Header(
            "<7if",
            [
                # Dataset values
                ("n_batch", 3),
                ("n_features", 4),  # size individual
                ("n_classes", 3),  # size labels
                # Elems
                ('n_train_elms', 120),
                ('n_validation_elms', 15),
                ('n_test_elms', 15),
                # Stats
                ('seed', -1),
                ('train_percentage', 0.8)
            ]
        )

        # get header labels
        print(header.n_classes, header.n_features)
        # set header labels
        print(header.set_label("n_classes", 9))
        # repr of header
        print(header)

        new_header = Header(
            "<7if",
            [
                "n_batch",
                "n_features",
                "n_classes",
                "n_train_elms",
                "n_validation_elms",
                "n_test_elms",
                "seed",
                "train_percentage"
            ],
            header.binary
        )

        print(new_header)
    """

    def __init__(self, fmt, labels, data=None):

        self.__fmt = fmt
        self.__data = data
        self.__p_data = labels
        self.__out_size = 42

        if not data:
            self.__p_data = labels
            self.__data = struct.pack(
                self.__fmt,
                *[value for label, value in self.__p_data]
            )
        else:
            self.__data = data
            for idx, value in enumerate(struct.unpack(self.__fmt, self.__data)):
                self.__p_data[idx] = (labels[idx], value)

    def __len__(self):
        return struct.calcsize(self.__fmt)

    @property
    def binary(self):
        """Get binary of python data."""
        return self.__data

    def __getattr__(self, name):
        for label, value in self.__p_data:
            if label == name:
                return value
        raise AttributeError("'{}' is not a label of this header!".format(
            name
        ))

    def set_label(self, name, new_value):
        """Change an header label value."""
        for idx, _ in enumerate(self.__p_data):
            if self.__p_data[idx][0] == name:
                self.__p_data[idx] = (name, new_value)

                self.__data = struct.pack(
                    self.__fmt,
                    *[value for label, value in self.__p_data]
                )

                return self.__p_data[idx]

        raise Exception("'{}' is not a label of this header!".format(
            name
        ))

    def __repr__(self):
        byte_per_line = 8

        string = "+----------- HEADER ".ljust(self.__out_size, "-") + '+\n'
        format_ = "| format string: '{}'".format(self.__fmt)
        string += format_.ljust(self.__out_size, " ") + '|\n'
        string += "+".ljust(self.__out_size, "-") + '+\n'

        for label, value in self.__p_data:
            cur_data = "| {}: {}".format(label, value)
            string += cur_data.ljust(self.__out_size, " ") + '|\n'

        string += "+".ljust(self.__out_size, "-") + '+\n'

        data = binascii.b2a_hex(self.__data)
        counter = 0
        cur_data = ''

        for idx in range(0, len(data), 4):
            if counter == 0:
                cur_data += "| "
            elif counter == byte_per_line:
                counter = 0
                string += cur_data.ljust(self.__out_size, " ") + '|\n'
                cur_data = "| "

            cur_data += "{} ".format("".join([chr(data[idx + cur_i])
                                              for cur_i in range(4)]))
            counter += 2

        string += "+".ljust(self.__out_size, "-") + '+\n'

        return string


def to_bin(data, type_):
    """Convert a numpy array do binary.

    The conversion will lose the shape, the resulting
    array is flat.

    Params:
        data (numpy array): the array to convert
        type_ (string): type of the elements, could be
                        "double" or "float"
    """

    # print("flat", len(data.flat))
    return struct.pack("{}{}".format(
        data.size,
        "d" if type_ == "double" else "f"
    ), *data.flat)


def check_classes(labels, extracted_classes):
    """Controls the numer of classes remained.

    Params:
        labels (numpy array): the source labels
        extracted_classes (list): class id extracted

    Raises:
        Exception: if the numer of classes found in labels is different from
                   the number of classes in extracted_classes
    """
    classes = []

    if len(labels) > 0:
        # print("+++ Check class balancing")
        for elm in labels:
            class_ = np.argmax(elm)
            if class_ not in classes:
                classes.append(class_)

        if len(classes) != len(extracted_classes):
            print("!!! WARNING !!! Some classes are not represented anymore in original set. Classes: {}".format(
                set(extracted_classes) - set(classes)
            ))

    return True


def calc_ratios(labels):
    """Calculates the ratio of each class.

    Params:
        labels (numpy array): labels of the current dataset

    Returs:
        list: the ratio for each class
    """
    tmp = {}
    ratios = []

    for elm in labels:
        class_ = np.argmax(elm)
        if class_ not in tmp:
            tmp[class_] = 0
        tmp[class_] += 1

    tot_elms = sum(tmp.values())

    for class_, num_elms in tmp.items():
        ratios.append(num_elms / tot_elms)

    # print(tmp)
    # print(ratios)

    return ratios


def dataset_extraction(source_data, source_labels, num_elems, num_x_class, classifier_labels):
    """Extract a portion of a dataset to create another set.

    Params:
        source_data (numpy array): the data
        source_data (numpy array): data labels
        num_elems (int): number of elements to extract
        num_x_class (int): the minimum number of elements per class
        classifier_labels (bool): if True dataset is for classification

    Returns:
        (source_data, source_labels, new_data, new_labels): a tuple with the new
            source without the elements extracted and the new set of data

    Note:
        The number of elements is update after the calculus of the ratio or
        the creation of num_x_class array (integer argument is converted in
        a list of integer, one for each class):

            num_elems = sum(num_x_class)
    """

    # print(new_data.shape, new_labels.shape)

    if len(source_data) < num_elems:
        return [], [], np.array(source_data, dtype=np.float64), np.array(source_labels, dtype=np.float64)

    num_elems = sum(num_x_class)

    new_data = np.empty([num_elems, source_data.shape[-1]], dtype=np.float64)
    if not classifier_labels:
        new_labels = np.empty(
            [num_elems, 1], dtype=np.float64)
    else:
        new_labels = np.empty(
            [num_elems, source_labels.shape[-1]], dtype=np.float64)

    # print(num_x_class)

    counter = 0
    counter_classes = [0 for _ in range(source_labels.shape[-1])]
    found_classes = []
    index = 0
    index_to_delete = []

    for index in range(len(source_data)):
        # print("index", index, "len data", len(source_data), "del idx", len(index_to_delete), "counter", counter, "num elems", num_elems, "class", np.argmax(source_labels[index]), counter_classes, num_x_class)

        class_ = np.argmax(source_labels[index])

        if class_ not in found_classes:
            found_classes.append(class_)

        if counter_classes[class_] < num_x_class[class_]:
            new_data[counter] = source_data[index].copy()
            new_labels[counter] = source_labels[index].copy()
            index_to_delete.append(index)
            counter_classes[class_] += 1
            counter += 1

        # print(counter_classes, num_x_class)
        if counter == num_elems:
            break
    ##
    # Delete extracted elements
    source_data = np.delete(source_data, index_to_delete, 0)
    source_labels = np.delete(source_labels, index_to_delete, 0)
    index_to_delete = []

    ##
    # Shuffle source
    indexes = np.random.permutation(len(source_data))
    source_data = source_data[indexes]
    source_labels = source_labels[indexes]

    ##
    # Check num elems and fix the difference
    if counter < num_elems:
        for index in range(len(source_data)):
            class_ = np.argmax(source_labels[index])
            new_data[counter] = source_data[index].copy()
            new_labels[counter] = source_labels[index].copy()
            counter_classes[class_] += 1
            index_to_delete.append(index)
            counter += 1
            if counter == num_elems:
                break

    # print(counter)
    # print(counter_classes, num_x_class)
    # print(counter_classes, sum(counter_classes))

    ##
    # Check class balancing
    check_classes(source_labels, found_classes)

    ##
    # Delete extracted indexes
    source_data = np.delete(source_data, index_to_delete, 0)
    source_labels = np.delete(source_labels, index_to_delete, 0)

    # print(source_data.shape)

    return source_data, source_labels, new_data, new_labels


def simple_dataset_extraction(source_data, source_labels, num_elems, classifier_labels):
    """Extract a portion of a dataset to create another set.

    Params:
        source_data (numpy array): the data
        source_data (numpy array): data labels
        num_elems (int): number of elements to extract
        classifier_labels (bool): if True dataset is for classification

    Returns:
        (source_data, source_labels, new_data, new_labels): a tuple with the new
            source without the elements extracted and the new set of data
    """

    # print(new_data.shape, new_labels.shape)

    if len(source_data) < num_elems:
        return [], [], np.array(source_data, dtype=np.float64), np.array(source_labels, dtype=np.float64)

    new_data = np.empty([num_elems, source_data.shape[-1]], dtype=np.float64)
    if not classifier_labels:
        new_labels = np.empty(
            [num_elems, 1], dtype=np.float64)
    else:
        new_labels = np.empty(
            [num_elems, source_labels.shape[-1]], dtype=np.float64)

    # print(num_x_class)

    counter = 0
    counter_classes = [0 for _ in range(source_labels.shape[-1])]
    found_classes = []
    index = 0
    index_to_delete = []

    for index in range(num_elems):
        new_data[index] = source_data[index].copy()
        new_labels[index] = source_labels[index].copy()
        index_to_delete.append(index)

        counter += 1

        if counter == num_elems:
            break
    ##
    # Delete extracted elements
    source_data = np.delete(source_data, index_to_delete, 0)
    source_labels = np.delete(source_labels, index_to_delete, 0)
    index_to_delete = []

    ##
    # Shuffle source
    indexes = np.random.permutation(len(source_data))
    source_data = source_data[indexes]
    source_labels = source_labels[indexes]

    return source_data, source_labels, new_data, new_labels


def get_equal_extractions(data, labels, size, classes_x_batch, classifier_labels, random_extraction):
    """Creates batches extracting classes in equal mode.

    Params:
        data (numpy.ndarray): data
        labels (numpy.ndarray): data
        size (int): size of a batch
        classes_x_batch (int): num of minimum records for each class
        random_extraction (bool): if true extraction is random

    Returns:
        (numpy.ndarray, numpy.ndarray): the new data and labels
    """
    # print(data.shape, size, classes_x_batch)
    new_data = []
    new_labels = []

    # print("len ->", len(data))
    for _ in tqdm(range(int(len(data) / size)), desc="Batches"):
        if random_extraction:
            data, labels, tmp_data, tmp_labels = simple_dataset_extraction(
                data, labels, size, classifier_labels
            )
        else:
            data, labels, tmp_data, tmp_labels = dataset_extraction(
                data, labels, size, classes_x_batch, classifier_labels
            )
        # print("newlen ->", len(data), type(tmp_data))
        new_data.append(tmp_data)
        new_labels.append(tmp_labels)

    new_data = np.array(new_data, dtype=np.object)
    new_labels = np.array(new_labels, dtype=np.object)

    # print(new_data.shape)
    # print(new_labels.shape)

    return new_data, new_labels


def class_inspector(labels):
    classes = {}

    for label in labels:
        class_ = np.argmax(label)
        if class_ not in classes:
            classes[class_] = 0
        classes[class_] += 1

    return classes


def get_elm_x_class(samples_x_class, labels, size, respect_ratio):
    if respect_ratio:
        ratios = calc_ratios(labels)
        tmp = [math.ceil(ratios[idx] * size)
               for idx in range(len(ratios))]
        if sum(tmp) > size:
            for _ in range(sum(tmp) - size):
                tmp[np.random.randint(len(tmp))] -= 1
        return tmp
    else:
        return [samples_x_class for _ in range(labels.shape[-1])]


def new(base_path, loader, size, name, n_shuffle=1, batch_size=True,
        seed=None, train_percentage=0.8, validation_percentage=0.1,
        test_percentage=0.1, autobalance=False, type_="double",
        extraction_type='random', version=1,
        train_perc_items=1.0, test_perc_items=1.0, UI=False, k_i_fold={'k': 0, 'i': 0}):
    """Generate a dataset useful for DENN.

    Params:
        base_path (str): base_path of the dataset to convert (original)
        loader (DatasetLoader instance): an object to call for the loading of
                                         the data
        size (int): num. of elements in a batch or num. of bathes.
                    Set also 'batch_size' properly
        name (str): name of the output dataset file
        n_shuffle (integer, default=1): number of shuffle of the single dataset.
                                        Subsequent shuffles will be added at
                                        the end of the dataset
        batch_size (bool, default=True): indicate if size is the num. of elems.
                                         or the num. of batches
        seed (int, default=None): seed of random number generator
        train_percentage (float, default=0.8): size in percentage of the train
                                               set
        validation_percentage (float, default=0.1): size in percentage of the
                                                    validation set
        test_percentage (float, default=0.1): size in percentage of the
                                              test set
        debug (bool, default=False): indicate if the loader enables the debug
                                     output
        autobalance (bool): if True the odd number of records will be balanced
                            to be even
        type_ (str, default="double"): indicate the type of the elements.
                                       Values can be ["double", "float"]
        extraction_type (string): method with elements are extracted
        version (int): number of dataset type used

    Notes:
        After loaded the dataset the method will check if the test or the
        validation set are already present; if true they will be added as they
        are, without any modification.

        If one of them is not present (this is valid also if both are not
        available) it will be extracted from the train set according to the
        percentage indicated as parameter.

        Sizes are calculated before the extraction, so the percetage refers to
        the main train set size.

        Note also that the loader load the data in float64 format, so subsequent
        conversion are applied only after the creation of the dataset (random
        and shuffle part), before the split part.
    """
    makedirs(BASEDATASETPATH, exist_ok=True)

    assert train_percentage + validation_percentage + \
        test_percentage <= 1.0, "Wrong partitions, size [ {} | {} | {} ] > 1.0".format(
            train_percentage, validation_percentage, test_percentage)

    print("+-[ Load data of {}]-----+".format(base_path))

    if UI:
        yield 5

    dataset = loader.real_load(base_path)

    np.random.seed(seed)

    print("++ Prepare data of {}".format(base_path))

    if UI:
        yield 10

    classifier_labels = loader.classifier_labels

    train_data, train_labels = dataset.train

    if extraction_type == 'respect_ratio':
        respect_ratio = True
    else:  # extraction_type == 'equal_division'
        respect_ratio = False
    if extraction_type == 'random':
        random_extraction = True
    else:
        random_extraction = False

    ##
    # Cut train set if needed
    #   Note: cut is made with an extraction respecting
    #   the ratio if it is a dataset for classification
    if classifier_labels and train_perc_items != 1.0:
        tot_elm = int(len(train_data) * train_perc_items)
        print("+ Cut test data {} -> {}".format(len(train_data), tot_elm))
        samples_x_class = int(len(train_data) / train_labels.shape[-1])
        samples_x_class = get_elm_x_class(
            samples_x_class, train_labels, tot_elm, True)
        _data, _labels, train_data, train_labels = dataset_extraction(
            train_data, train_labels, tot_elm, samples_x_class, classifier_labels)
        print("+ Train cut done!")

    train_size = len(train_data)
    n_features = train_data.shape[-1]
    if len(train_labels.shape) > 1:
        n_classes = train_labels.shape[-1]
    else:
        n_classes = 1

    # Calculate other sizes
    validation_size = int(train_size * validation_percentage)
    test_size = int(train_size * test_percentage)

    ##
    # Initial shuffle
    indexes = np.random.permutation(len(train_data))
    train_data = train_data[indexes]
    train_labels = train_labels[indexes]

    if dataset.test is not None:
        print("++ Get Test set")
        test_data, test_labels = dataset.test
        test_size = len(test_data)
    elif k_i_fold['k'] != 0:
        test_size = int(float(train_size) / k_i_fold['k'])
        test_start_id = k_i_fold['i'] * test_size
        test_end_id = test_start_id + test_size
        print("test size:" + str(test_size))
        print("test start id:" + str(test_start_id))
        print("test end id:" + str(test_end_id))
        print("train data:" + str(train_data.shape))
        print("train labels:" + str(train_labels.shape))
        test_data = train_data[test_start_id:test_end_id]
        test_labels = train_labels[test_start_id:test_end_id]
        train_data = np.delete(train_data, range(
            test_start_id, test_end_id), axis=0)
        train_labels = np.delete(train_labels, range(
            test_start_id, test_end_id), axis=0)
        train_size = len(train_data)
        print("train data:" + str(train_data.shape))
        print("train labels:" + str(train_labels.shape))
    else:
        print("++ Extract Test set")
        samples_x_class = int(test_size / train_labels.shape[-1])
        samples_x_class = get_elm_x_class(
            samples_x_class, train_labels, test_size, True)

        train_data, train_labels, test_data, test_labels = dataset_extraction(
            train_data, train_labels, test_size, samples_x_class, classifier_labels)

        train_size = len(train_data)

    ##
    # Cut test set if needed
    #   Note: cut is made with an extraction respecting
    #   the ratio if it is a dataset for classification
    if classifier_labels and test_perc_items != 1.0:
        tot_elm = int(len(test_data) * test_perc_items)
        print("+ Cut test data {} -> {}".format(len(test_data), tot_elm))
        samples_x_class = int(len(test_data) / test_labels.shape[-1])
        samples_x_class = get_elm_x_class(
            samples_x_class, test_labels, tot_elm, True)
        _data, _labels, test_data, test_labels = dataset_extraction(
            test_data, test_labels, tot_elm, samples_x_class, classifier_labels)
        print("+ Test cut done!")
        test_size = len(test_data)

    if UI:
        yield 15

    if validation_percentage > 0.0:
        ##
        # Second shuffle
        indexes = np.random.permutation(len(train_data))
        train_data = train_data[indexes]
        train_labels = train_labels[indexes]

        if dataset.validation is not None:
            print("++ Get Validation set")
            validation_data, validation_labels = dataset.validation
            validation_size = len(validation_data)
        else:
            print("++ Extract Validation set")
            samples_x_class = int(validation_size / train_labels.shape[-1])
            samples_x_class = get_elm_x_class(
                samples_x_class, train_labels, validation_size, True)

            train_data, train_labels, validation_data, validation_labels = dataset_extraction(
                train_data, train_labels, validation_size, samples_x_class, classifier_labels)

            train_size = len(train_data)
    else:
        validation_data = np.array([])
        validation_labels = np.array([])

    if UI:
        yield 20

    original_data = train_data.copy()
    original_labels = train_labels.copy()

    # print(original_data.shape, len(original_data), original_data.size)

    # print(train_data.shape, train_labels.shape)

    # print('size', size)
    if not batch_size:
        print("++ Calculate num batches")
        num_elms = len(original_data) * n_shuffle
        elm_x_batch = size
        size = int(num_elms / size)
    # print('size', size)
    classes_x_batch = int(size / train_labels.shape[-1])

    if UI:
        yield 30

    if k_i_fold['k'] == 0:
        train_data = []
        train_labels = []
        for num in range(n_shuffle):
            print("++ Add shuffle n. {}".format(num + 1))
            indexes = np.random.permutation(len(original_data))
            cur_train_data = original_data[indexes].copy()
            cur_train_labels = original_labels[indexes].copy()

            # Convert to float if necessary
            if type_ == "float":
                cur_train_data = cur_train_data.astype(np.float32)
                cur_train_labels = cur_train_labels.astype(np.float32)

            # print(cur_train_data.shape)
            # print(cur_train_labels.shape)

            div_reminder = len(cur_train_data) % size

            # print(div_reminder)

            if autobalance and div_reminder != 0:
                idx_to_add = np.random.randint(
                    0, len(cur_train_data),
                    size=(size - div_reminder)
                )
                cur_train_data = np.append(
                    cur_train_data, cur_train_data[idx_to_add], axis=0)
                cur_train_labels = np.append(
                    cur_train_labels, cur_train_labels[idx_to_add], axis=0)

            # print(cur_train_data.shape, len(cur_train_data))
            # print(cur_train_labels.shape, len(cur_train_labels))

            cur_class_x_batch = get_elm_x_class(
                classes_x_batch, cur_train_labels, size, respect_ratio)

            # print(cur_class_x_batch)

            cur_train_data, cur_train_labels = get_equal_extractions(
                cur_train_data, cur_train_labels, size, cur_class_x_batch,
                classifier_labels, random_extraction
            )

            # print(cur_train_data.shape, len(cur_train_data))
            # print(cur_train_labels.shape, len(cur_train_labels))

            train_data.append(cur_train_data)
            train_labels.append(cur_train_labels)

            # print(len(train_data), len(train_labels))

            if UI:
                yield 40

        print("++ Concat {} shuffles".format(n_shuffle))
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
    else:
        train_data = np.array([train_data])
        train_labels = np.array([train_labels])

    if UI:
        yield 50

    # for label in train_labels:
    #     print(class_inspector(label))

    # print(train_data.shape, len(train_data))
    # print(train_labels.shape, len(train_labels))

    # print('train', train_data.shape, train_labels.shape)

    # print(len(train_data), len(train_data[0]), len(train_data[0][0]))
    # print(len(train_labels), len(train_labels[0]), len(train_labels[0][0]))

    print("+- Sizes: [ {} | {} | {} ]".format(train_size,
                                              validation_size, test_size))
    print('+- train {} | {}'.format(train_data.shape, train_labels.shape))
    print('+- validation {} | {}'.format(validation_data.shape, validation_labels.shape))
    print('+- test {} | {}'.format(test_data.shape, test_labels.shape))

    print("++ Prepare Header")

    if version == 1:
        n_batch = len(train_data)
    elif version == 2:
        n_batch = sum(len(elm) for elm in train_data)
    else:
        raise Exception("Unexpected version '{}'".format(version))

    header = Header(
        "<H5if3I",
        [
            ("version", version),
            # Dataset values
            ("n_batch", n_batch),
            ("n_features", n_features),  # size individual
            ("n_classes", n_classes),  # size labels
            ##
            # Type of values:
            #   1 -> float
            #   2 -> double
            ('type', 2 if type_ == "double" else 1),
            # Stats
            ('seed', seed if seed is not None else -1),
            ('train_percentage', train_percentage),
            # Offset
            ('test_offset', 0),
            ('validation_offset', 0),
            ('train_offset', 0)
        ]
    )

    if UI:
        yield 60

    # print(header)
    # print(len(header))

    is_batch = "-B" if batch_size else "xB"

    print("+++ Create gz file")
    if version == 1:
        file_name = path.join(
            BASEDATASETPATH, "{}_{}x{}_{}s.gz".format(name, n_batch, len(train_data[0]), n_shuffle))
    elif version == 2:
        file_name = path.join(
            BASEDATASETPATH, "{}_{}i_{}s.gz".format(name, n_batch, n_shuffle))
    else:
        raise Exception("Unexpected version '{}'".format(version))

    with gzip.GzipFile(file_name, mode='wb') as gz_file:

        print("+++ Calculate test size")
        test_size = struct.calcsize("{}{}".format(
            len(test_data) * header.n_features,
            "d" if type_ == "double" else "f")
        ) + struct.calcsize("{}{}".format(
            len(test_data) * header.n_classes,
            "d" if type_ == "double" else "f")
        )

        print("+++ Calculate validation size")
        validation_size = struct.calcsize("{}{}".format(
            len(validation_data) * header.n_features,
            "d" if type_ == "double" else "f")
        ) + struct.calcsize("{}{}".format(
            len(validation_data) * header.n_classes,
            "d" if type_ == "double" else "f")
        )

        # print('header', len(header))
        # print('validation', validation_size)
        # print('test', test_size)

        print("+++ Update offsets in header")
        header.set_label("test_offset", len(header))  # size header
        # size header + test size + num elm test
        header.set_label("validation_offset", len(header) + test_size + 4)
        header.set_label("train_offset", len(header) +
                         test_size + validation_size + 8)  # size header + test size + validation size + num elm test + num elm validation

        if UI:
            yield 70
        # print(header)

        print("+++ Write header")
        gz_file.write(header.binary)
        # print(gz_file.tell())

        ##
        # TEST
        #
        # + num. elems (unsigned long)
        # + data
        # + labels
        print("+++ Write test data and labels")
        gz_file.write(struct.pack("<I", len(test_data)))
        gz_file.write(to_bin(test_data, type_))
        gz_file.write(to_bin(test_labels, type_))

        if UI:
            yield 80

        # print(gz_file.tell())

        ##
        # VALIDATION
        #
        # + num. elems (unsigned long)
        # + data
        # + labels
        print("+++ Write validation data and labels")
        gz_file.write(struct.pack("<I", len(validation_data)))
        gz_file.write(to_bin(validation_data, type_))
        gz_file.write(to_bin(validation_labels, type_))

        if UI:
            yield 90

        # print(gz_file.tell())

        ##
        # TRAIN
        #
        # [
        #   + current batch num (unsigned int)
        #   + num. elems (unsigned long)
        #   + data
        #   + labels
        # ]
        print("+++ Write all batches")
        with tqdm(total=len(train_data)) as pbar:
            if version == 1:
                for index, t_data in enumerate(train_data):
                    gz_file.write(struct.pack("<I", index))
                    gz_file.write(struct.pack("<I", len(t_data)))
                    gz_file.write(to_bin(t_data, type_))
                    gz_file.write(to_bin(train_labels[index], type_))
                    pbar.update(1)
            elif version == 2:
                for idx_batch, batch in enumerate(train_data):
                    for index, elem in enumerate(batch):
                        gz_file.write(to_bin(elem, type_))
                        gz_file.write(
                            to_bin(train_labels[idx_batch][index], type_))
                    pbar.update(1)
            else:
                raise Exception("Unexpected version '{}'".format(version))

        if UI:
            yield 100

    print("+! Dataset {} completed!".format(file_name))


def new_dataset(*args, **kwargs):
    for _ in new(*args, **kwargs):
        pass


def gen_test_dataset():
    from dataset_loaders import TestDataset

    dataset = TestDataset().real_load()

    version = 1
    batch_size = 100
    type_ = 'float'

    train_data = np.array(np.split(dataset.train.data, batch_size))
    train_labels = np.array(np.split(dataset.train.labels, batch_size))

    header = Header(
        "<H5if3I",
        [
            ("version", version),
            # Dataset values
            ("n_batch", len(train_data)),
            ("n_features", train_data.shape[-1]),  # size individual
            ("n_classes", train_labels.shape[-1]),  # size labels
            ##
            # Type of values:
            #   1 -> float
            #   2 -> double
            ('type', 2 if type_ == "double" else 1),
            # Stats
            ('seed', -1),
            ('train_percentage', 1.0),
            # Offset
            ('test_offset', 0),
            ('validation_offset', 0),
            ('train_offset', 0)
        ]
    )

    with gzip.GzipFile("TestDataset.gz", mode='wb') as gz_file:

        print("+++ Calculate test size")
        test_size = struct.calcsize("{}{}".format(
            len(dataset.test.data) * header.n_features,
            "d" if type_ == "double" else "f")
        ) + struct.calcsize("{}{}".format(
            len(dataset.test.data) * header.n_classes,
            "d" if type_ == "double" else "f")
        )

        print("+++ Calculate validation size")
        validation_size = struct.calcsize("{}{}".format(
            len(dataset.validation.data) * header.n_features,
            "d" if type_ == "double" else "f")
        ) + struct.calcsize("{}{}".format(
            len(dataset.validation.data) * header.n_classes,
            "d" if type_ == "double" else "f")
        )

        # print('header', len(header))
        # print('validation', validation_size)
        # print('test', test_size)

        print("+++ Update offsets in header")
        header.set_label("test_offset", len(header))  # size header
        # size header + test size + num elm test
        header.set_label("validation_offset", len(header) + test_size + 4)
        header.set_label("train_offset", len(header) +
                         test_size + validation_size + 8)  # size header + test size + validation size + num elm test + num elm validation
        # print(header)

        print("+++ Write header")
        gz_file.write(header.binary)
        # print(gz_file.tell())

        ##
        # TEST
        #
        # + num. elems (unsigned long)
        # + data
        # + labels
        print("+++ Write test data and labels")
        gz_file.write(struct.pack("<I", len(dataset.test.data)))
        gz_file.write(to_bin(dataset.test.data, type_))
        gz_file.write(to_bin(dataset.test.labels, type_))

        # print(gz_file.tell())

        ##
        # VALIDATION
        #
        # + num. elems (unsigned long)
        # + data
        # + labels
        print("+++ Write validation data and labels")
        gz_file.write(struct.pack("<I", len(dataset.validation.data)))
        gz_file.write(to_bin(dataset.validation.data, type_))
        gz_file.write(to_bin(dataset.validation.labels, type_))

        # print(gz_file.tell())

        ##
        # TRAIN
        #
        # [
        #   + current batch num (unsigned int)
        #   + num. elems (unsigned long)
        #   + data
        #   + labels
        # ]
        print("+++ Write all batches")
        with tqdm(total=len(train_data)) as pbar:
            if version == 1:
                for index, t_data in enumerate(train_data):
                    gz_file.write(struct.pack("<I", index))
                    gz_file.write(struct.pack("<I", len(t_data)))
                    gz_file.write(to_bin(t_data, type_))
                    gz_file.write(to_bin(train_labels[index], type_))
                    pbar.update(1)
            elif version == 2:
                for idx_batch, batch in enumerate(train_data):
                    for index, elem in enumerate(batch):
                        gz_file.write(to_bin(elem, type_))
                        gz_file.write(
                            to_bin(train_labels[idx_batch][index], type_))
                    pbar.update(1)
            else:
                raise Exception("Unexpected version '{}'".format(version))

    print("+! Dataset {} completed!".format("TestDataset.gz"))


def main():
    gen_test_dataset()


if __name__ == '__main__':
    main()
