import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import datasetgenerator
import numpy as np
from os import path
from ast import literal_eval


class Communicator(QObject):

    datasetChanged = pyqtSignal()


class StatsLabel(QLabel):

    def __init__(self, title, comm):
        super(self.__class__, self).__init__(title)
        self.setAcceptDrops(True)
        self.dataset = None
        self.__comm = comm

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        else:
            raise AttributeError()

    def dragEnterEvent(self, e):
        e.accept() if e.mimeData().hasUrls() else e.ignore()

    def dropEvent(self, e):
        cur_file = e.mimeData().urls()[0].toLocalFile()
        self.dataset = datasetgenerator.Dataset(cur_file)
        self.setText("{}\n\n{}".format(
            path.basename(cur_file),
            str(self.dataset.stats))
        )
        self.__comm.datasetChanged.emit()


class DatasetGenerator(QWidget):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.title = 'DENN Dataset Viewer'
        self.width = 640
        self.height = 480

        ##
        # Data
        self.__batch = None

        ##
        # Comunicator
        self.__gen_comm = Communicator()

        ##
        # UI components
        self.dbObj = StatsLabel("Drag dataset here...", self.__gen_comm)
        self.l_batch = QLabel("Batch")
        self.combo_batch = QComboBox(self)
        self.l_batch_stats = QLabel("Batch info")
        self.l_validation_stats = QLabel("Validation info")
        self.l_test_stats = QLabel("Test info")
        self.l_data_shape = QLabel("Data shape")
        self.textbox_data_shape = QLineEdit(self)
        self.l_batch_data_title = QLabel("Batch record")
        self.textbox_batch_record = QLineEdit(self)
        self.l_batch_data = QLabel("Batch data")
        self.l_validation_data_title = QLabel("Validation record")
        self.textbox_validation_record = QLineEdit(self)
        self.l_validation_data = QLabel("Validation data")
        self.l_test_data_title = QLabel("Test record")
        self.textbox_test_record = QLineEdit(self)
        self.l_test_data = QLabel("Test data")

        ##
        # init UI
        self.initUI()

        ##
        # Events
        self.__gen_comm.datasetChanged.connect(self.changeDataset)
        self.combo_batch.activated.connect(self.batchSelected)
        self.textbox_data_shape.returnPressed.connect(self.updateAllData)
        self.textbox_batch_record.returnPressed.connect(self.updateBatchData)
        self.textbox_validation_record.returnPressed.connect(self.updateValidationData)
        self.textbox_test_record.returnPressed.connect(self.updateTestData)

        self.show()

    def initUI(self):
        font = QFont()
        font.setFamily('Courier New')
        font.StyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(14)

        for component in [
            self.dbObj,
            self.l_batch_stats,
            self.l_validation_stats,
            self.l_test_stats,
            self.l_batch_data,
            self.l_validation_data,
            self.l_test_data
        ]:
            component.setStyleSheet("""
                padding: 12px;
                border-radius: 6px;
                background-color: rgb(245, 245, 245); 
                color: rgb(333); 
            """)
            component.setFont(font)
            component.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.l_batch.setStyleSheet("""
            qproperty-alignment: AlignRight;
        """)

        self.scroll_b_stats = QScrollArea()
        self.scroll_b_stats.setWidgetResizable(True)
        self.scroll_b_stats.setWidget(self.l_batch_stats)
        self.scroll_b_stats.setFixedSize(265, 345)

        scroll_v_stats = QScrollArea()
        scroll_v_stats.setWidgetResizable(True)
        scroll_v_stats.setWidget(self.l_validation_stats)

        scroll_t_stats = QScrollArea()
        scroll_t_stats.setWidgetResizable(True)
        scroll_t_stats.setWidget(self.l_test_stats)

        scroll_b = QScrollArea()
        scroll_b.setWidgetResizable(True)
        scroll_b.setWidget(self.l_batch_data)

        scroll_v = QScrollArea()
        scroll_v.setWidgetResizable(True)
        scroll_v.setWidget(self.l_validation_data)

        scroll_t = QScrollArea()
        scroll_t.setWidgetResizable(True)
        scroll_t.setWidget(self.l_test_data)

        vbox_info = QVBoxLayout()
        hbox_header_batch = QHBoxLayout()
        hbox_batch_selector = QHBoxLayout()
        hbox_val_test = QHBoxLayout()
        hbox_data_shape = QHBoxLayout()
        vbox_stats = QVBoxLayout()
        vbox_data = QVBoxLayout()
        hbox_all = QHBoxLayout()
        hbox_batch_record = QHBoxLayout()
        hbox_validation_record = QHBoxLayout()
        hbox_test_record = QHBoxLayout()

        hbox_batch_selector.addWidget(self.l_batch)
        hbox_batch_selector.addWidget(self.combo_batch)

        vbox_stats.addLayout(hbox_batch_selector)
        vbox_stats.addWidget(self.scroll_b_stats)

        # hbox.addWidget(scroll)
        hbox_header_batch.addWidget(self.dbObj)
        hbox_header_batch.addLayout(vbox_stats)

        hbox_val_test.addWidget(scroll_v_stats)
        hbox_val_test.addWidget(scroll_t_stats)

        hbox_data_shape.addWidget(self.l_data_shape)
        hbox_data_shape.addWidget(self.textbox_data_shape)

        vbox_info.addLayout(hbox_header_batch)
        vbox_info.addLayout(hbox_val_test)

        hbox_batch_record.addWidget(self.l_batch_data_title)
        hbox_batch_record.addWidget(self.textbox_batch_record)

        hbox_validation_record.addWidget(self.l_validation_data_title)
        hbox_validation_record.addWidget(self.textbox_validation_record)

        hbox_test_record.addWidget(self.l_test_data_title)
        hbox_test_record.addWidget(self.textbox_test_record)

        vbox_data.addLayout(hbox_data_shape)
        vbox_data.addLayout(hbox_batch_record)
        vbox_data.addWidget(scroll_b)
        vbox_data.addLayout(hbox_validation_record)
        vbox_data.addWidget(scroll_v)
        vbox_data.addLayout(hbox_test_record)
        vbox_data.addWidget(scroll_t)

        hbox_all.addLayout(vbox_info)
        hbox_all.addLayout(vbox_data)

        self.setLayout(hbox_all)

    def changeDataset(self):
        self.combo_batch.clear()
        for num_batch in range(self.dbObj.stats.n_batch):
            self.combo_batch.addItem(str(num_batch))
        ##
        # Data
        self.textbox_batch_record.setText("0")
        self.textbox_validation_record.setText("0")
        self.textbox_test_record.setText("0")
        self.textbox_data_shape.setText(str(self.dbObj.dataset[0].data.shape[1:]))
        ##
        # Stats
        self.batchSelected(0)
        #validation
        self.l_validation_stats.setText(
            self.getDataStats(self.dbObj.validation_labels, "Validation set"))
        #test
        self.l_test_stats.setText(
            self.getDataStats(self.dbObj.test_labels, "Test set"))
        ##
        # Update
        self.updateValidationData()
        self.updateTestData()
        
    
    def updateBatchData(self):
        self.l_batch_data.setText(
            self.getData(
                self.__batch,
                int(self.textbox_batch_record.text())
            )
        )
    
    def updateValidationData(self):
        self.l_validation_data.setText(
            self.getData(
                (self.dbObj.validation_data, self.dbObj.validation_labels),
                int(self.textbox_validation_record.text())
            )
        )
    
    def updateTestData(self):
        self.l_test_data.setText(
            self.getData(
                (self.dbObj.test_data, self.dbObj.test_labels),
                int(self.textbox_test_record.text())
            )
        )

    def updateAllData(self):
        self.updateBatchData()
        self.updateValidationData()
        self.updateTestData()
        

    def getData(self, cur_data, idx):
        data, labels = cur_data
        shape = self.__batch.data.shape[1:]
        new_shape = literal_eval(self.textbox_data_shape.text())
        if isinstance(new_shape, tuple):
            shape = new_shape
        if idx < len(data):
            return "Record[{}]-> class({})\n{}\n\n".format(
                idx, np.argmax(labels[idx]),
                np.array2string(
                    data[idx].reshape(shape),
                    precision=2,
                    suppress_small=True,
                    max_line_width=1000,
                    separator=""
                )
            )
        else:
            return "Index out of range"

    def getDataStats(self, cur_data, title):
        if cur_data.shape[-1] > 1:
            try:
                classes = np.argmax(cur_data, axis=1).tolist()
                data_stats = []
                for class_ in range(cur_data.shape[-1]):
                    class_elems = classes.count(class_)
                    data_stats.append(
                        (class_, (float(class_elems / len(cur_data)) * 100.), class_elems)
                    )
                out_string = "{}: {} elements\n".format(title, len(cur_data))
                out_string += "───────────────────────────\n"
                out_string += " Class │   %   │ num. elem.\n"
                out_string += "───────────────────────────\n"
                for class_, percent, elems in data_stats:
                    percent_str = " {:5.2f} ".format(percent)
                    out_string += str(class_).rjust(7, " ")
                    out_string += "│{}│ ".format(percent_str)
                    out_string += str(elems) + "\n"
                return out_string
            except Exception as e:
                return "ERROR!!!" + str(e)
        else:
            raise Exception(
                "Only classifier data are visible at the moment...")

    def batchSelected(self, idx):
        self.__batch = self.dbObj.dataset[idx]
        self.l_batch_stats.setText(
            self.getDataStats(self.__batch.labels, "Batch"))
        self.updateBatchData()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DatasetGenerator()
    sys.exit(app.exec_())
