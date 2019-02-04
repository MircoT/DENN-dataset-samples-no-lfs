"""Smoke tests."""
import unittest
from os import path
from random import randint

import numpy as np

from data.reader import (Container, Dataset, FashionMNISTDataset, MNISTDataset,
                         Resource)


class ResourceObj(unittest.TestCase):

    """Resource object tests."""

    def test_creation(self):
        """Creation of the resource object."""
        with self.assertRaises(AssertionError) as err:
            Resource(range(randint(1, 100)), range(randint(1, 100)))

        with self.assertRaises(AssertionError) as err:
            Resource(np.random.randint(10, size=10), range(randint(1, 100)))

        resource = Resource(np.random.randint(10, size=10),
                            np.random.randint(10, size=5))
        self.assertIsInstance(resource, Resource)

    def test_getitem(self):
        """Resource __getitem__."""
        rand_size = randint(1, 100)
        resource = Resource(np.array(range(rand_size)),
                            np.array(range(rand_size)))

        for idx in range(rand_size):
            attribute, output = resource[idx]
            self.assertEqual(attribute, idx)
            self.assertEqual(output, idx)

        with self.assertRaises(IndexError):
            resource[rand_size]

    def test_iterator(self):
        """Resource iterator."""
        rand_size = randint(1, 100)
        resource = Resource(np.array(range(rand_size)),
                            np.array(range(rand_size)))

        val = 0
        for attribute, output in resource:
            self.assertEqual(attribute, val)
            self.assertEqual(output, val)
            val += 1


class DatasetObj(unittest.TestCase):

    """Dataset base tests."""

    def test_creation(self):
        """Creation of the dataset object."""
        with self.assertRaises(AssertionError) as err:
            dataset = Dataset(kind="foo")

        with self.assertRaises(AssertionError) as err:
            dataset = Dataset()
            dataset.insert(42)

        resources = [
            Resource(np.random.randint(10, size=10),
                     np.random.randint(10, size=5))
            for _ in range(randint(0, 10))
        ]

        dataset = Dataset()
        for resource in resources:
            self.assertIsInstance(resource, Resource)
            dataset.insert(resource)
        self.assertIsInstance(dataset, Dataset)

    def test_insert_train(self):
        """Insert of train set in dataset object."""
        dataset = Dataset()
        dataset.insert(
            Resource(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3])), "train")
        self.assertIsInstance(dataset.train, Container)
        self.assertEqual(len(dataset.train), 5)
        self.assertEqual(len(dataset), len(dataset.train))

    def test_insert_validation(self):
        """Insert of validation set in dataset object."""
        dataset = Dataset()
        dataset.insert(
            Resource(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3])), "validation")
        self.assertIsInstance(dataset.validation, Container)
        self.assertEqual(len(dataset.validation), 5)
        self.assertEqual(len(dataset), len(dataset.validation))

    def test_insert_test(self):
        """Insert of test set in dataset object."""
        dataset = Dataset()
        dataset.insert(
            Resource(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3])), "test")
        self.assertIsInstance(dataset.test, Container)
        self.assertEqual(len(dataset.test), 5)
        self.assertEqual(len(dataset), len(dataset.test))

    def test_getitem(self):
        """Dataset __getitem__."""
        dataset = Dataset()
        dataset.insert(
            Resource(np.array([1, 2, 3]), np.array([1, 2, 3])), "train")
        dataset.insert(
            Resource(np.array([4, 5]), np.array([4, 5])), "validation")
        dataset.insert(
            Resource(np.array([6, 7, 8]), np.array([6, 7, 8])), "test")

        for idx in range(8):
            attribute, output = dataset[idx]
            self.assertEqual(attribute, idx + 1)
            self.assertEqual(output, idx + 1)

        with self.assertRaises(IndexError):
            dataset[8]

    def test_iterator(self):
        """Dataset iterator."""
        dataset = Dataset()
        dataset.insert(
            Resource(np.array([1, 2, 3]), np.array([1, 2, 3])), "train")
        dataset.insert(
            Resource(np.array([4, 5]), np.array([4, 5])), "validation")
        dataset.insert(
            Resource(np.array([6, 7, 8]), np.array([6, 7, 8])), "test")

        val = 1
        for attribute, output in dataset:
            self.assertEqual(attribute, val)
            self.assertEqual(output, val)
            val += 1


class MNISTObj(unittest.TestCase):

    """MNIST dataset tests."""

    def test_creation(self):
        """Creation of the MNIST dataset object."""
        mnist_folder_path = path.join("..", "source_data", "MNIST")

        # Base dataset
        base_dataset = MNISTDataset(
            mnist_folder_path, normalized=False, onehot=False)
        base_first_attr, base_first_label = base_dataset[0]

        self.assertEqual(
            base_first_attr[np.nonzero(base_first_attr)[0][42]], 56)
        self.assertEqual(base_first_label, 5)

        self.assertEqual(len(base_dataset.train), 60000)
        self.assertEqual(len(base_dataset.validation), 0)
        self.assertEqual(len(base_dataset.test), 10000)
        self.assertEqual(len(base_dataset), 70000)

        self.assertListEqual(
            base_dataset[59999][0].tolist(),
            base_dataset.train[-1][0].tolist()
        )
        self.assertListEqual(
            base_dataset[60000][0].tolist(),
            base_dataset.test[0][0].tolist()
        )

        # Normalized and onehot dataset
        dataset = MNISTDataset(mnist_folder_path, normalized=True, onehot=True)
        first_attr, first_labels = dataset[0]

        self.assertEqual(first_attr[np.nonzero(
            first_attr)[0][42]], 0.2196078431372549)
        self.assertListEqual(first_labels.tolist(), [
                             0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        
        self.assertEqual(len(dataset.train), 60000)
        self.assertEqual(len(dataset.validation), 0)
        self.assertEqual(len(dataset.test), 10000)
        self.assertEqual(len(dataset), 70000)

        self.assertListEqual(
            dataset[59999][0].tolist(),
            dataset.train[-1][0].tolist()
        )
        self.assertListEqual(
            dataset[60000][0].tolist(),
            dataset.test[0][0].tolist()
        )


class FashionMNISTObj(unittest.TestCase):

    """Fashion-MNIST dataset tests."""

    def test_creation(self):
        """Creation of the Fashion-MNIST dataset object."""
        mnist_folder_path = path.join("..", "source_data", "Fashion-MNIST")

        # Base dataset
        base_dataset = MNISTDataset(
            mnist_folder_path, normalized=False, onehot=False)
        base_first_attr, base_first_label = base_dataset[0]

        self.assertEqual(
            base_first_attr[np.nonzero(base_first_attr)[0][42]], 69)
        self.assertEqual(base_first_label, 9)

        self.assertEqual(len(base_dataset.train), 60000)
        self.assertEqual(len(base_dataset.validation), 0)
        self.assertEqual(len(base_dataset.test), 10000)
        self.assertEqual(len(base_dataset), 70000)

        self.assertListEqual(
            base_dataset[59999][0].tolist(),
            base_dataset.train[-1][0].tolist()
        )
        self.assertListEqual(
            base_dataset[60000][0].tolist(),
            base_dataset.test[0][0].tolist()
        )

        # Normalized and onehot dataset
        dataset = MNISTDataset(mnist_folder_path, normalized=True, onehot=True)
        first_attr, first_labels = dataset[0]

        self.assertEqual(first_attr[np.nonzero(
            first_attr)[0][42]], 0.27058823529411763)
        self.assertListEqual(first_labels.tolist(), [
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        
        self.assertEqual(len(dataset.train), 60000)
        self.assertEqual(len(dataset.validation), 0)
        self.assertEqual(len(dataset.test), 10000)
        self.assertEqual(len(dataset), 70000)

        self.assertListEqual(
            dataset[59999][0].tolist(),
            dataset.train[-1][0].tolist()
        )
        self.assertListEqual(
            dataset[60000][0].tolist(),
            dataset.test[0][0].tolist()
        )


if __name__ == '__main__':
    unittest.main()
