# DENN-dataset-samples

### Install dependencies

```bash
pip install -r requirements.txt
```

### Open generator

```bash
python dataset_generator.py 
```

### Open viewer

```bash
python dataset_viewer.py
```

### Use the generator from a Python script

```python
import datasetgenerator

def main():
    ##
    # Example to generate an MNIST dataset
    datasetgenerator.new_dataset(
        './source_data/MNIST',  # base_path, where the loader will read the original data
        datasetgenerator.MNISTLoader(),  # an instance of the data loader
                                         # you can use your loader here
        100,  # size
        "MNIST_DATASET",  # base output name
        n_shuffle=1,  # number of shuffles of the dataset
                      # 2 shuffle means that the dataset will be used 2 times
                      # NOTE: data are extracted always randomly
        batch_size=True,  # Indicates that size is the size of the batch and
                          # not the number of batches
        seed=None,  # random seed
        train_percentage=0.8,  # Percentage of the train set
        validation_percentage=0.1,  # Percentage of the validation set
        test_percentage=0.1,   # Percentage of the test set
        autobalance=False,  # Balance the size of the batches
        type_="float",  # Type of the records
        extraction_type='random',  # Kind of extraction
                                   # - random: extract records randomly
                                   # - equal_division: divides the number of
                                   #   record classes equally
                                   # - respect_ratio: respect the class division
                                   #   in the dataset
        version=1,  # version of the dataset generated (only 1 is supported right now)
        train_perc_items=1.0,  # Percentage of train set selected at the end
                               # used to cut the final train set if needed
        test_perc_items=1.0    # Same thing but for the test set
    )

if __name__ == '__main__':
    main()
```

### Create a new loader

```python
import datasetgenerator

class MyLoader(datasetgenerator.DatasetLoader):

    """My personalized loader."""

    def load(self, path):
        # load train data as numpy array
        # ...
        # load train labels as numpy array
        # ...

        return datasetgenerator.SetObj(
            datasetgenerator.SetComponent(
                train_data,
                train_labels
            ),
            None,  # or a SetComponent like the train set above
            None   # or a SetComponent like the train set above
        )

```

After that you can use it ind `datasetgenerator.new_dataset`.

### Pyinstaller on Windows:

pyinstaller -F -w --path C:\Users\USER\AppData\Local\Programs\Python\Python35\Lib\site-packages\PyQt5\Qt\bin .\dataset_generator.py