import warnings
import math
from PyQt5.QtCore import Qt, right
# from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5 import QtGui, QtWidgets
# from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout
from numpy.core.defchararray import array, count
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import angle
# from matplotlib.colors import same_color
import pyqtgraph
# from pyqtgraph.Point import Point
# from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
from GUI import Ui_MainWindow
import sys
import numpy as np
from PyQt5 import QtCore as qtc
from math import *
import logging
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib
import qdarkstyle

from worker import worker
matplotlib.use('Qt5Agg')


class MathTextLabel(QtWidgets.QWidget):

    def __init__(self, mathText, parent=None, **kwargs):
        super(QtWidgets.QWidget, self).__init__(parent, **kwargs)

        l = QVBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)

        r, g, b, a = self.palette().base().color().getRgbF()

        self._figure = Figure(edgecolor=(r, g, b), facecolor=(r, g, b))
        self._canvas = FigureCanvasQTAgg(self._figure)
        l.addWidget(self._canvas)
        self._figure.clear()
        text = self._figure.suptitle(
            mathText,
            x=0.0,
            y=1.0,
            horizontalalignment='left',
            verticalalignment='top',
            size=QtGui.QFont().pointSize()*1
        )
        self._canvas.draw()


class MainWindow(QtWidgets.QMainWindow):
    construct_matrix = qtc.pyqtSignal(list, list, int, int, int, list)
    cancel_the_error_map_operation = qtc.pyqtSignal()
    transpose_the_matrix = qtc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gui = Ui_MainWindow()
        self.gui.setupUi(self)
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.gui.error_map_widget.hide()
        self.gui.error_settings_widget.hide()
        self.gui.progress_bar_widget.hide()
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Critical)
        self.msg.setText("Error")
        self.msg.setWindowTitle("Error")
        self.main_graph_data = []
        self.main_graph_time = []
        # self.new_ys = [[]]
        self.new_y = []
        self.new_x = []
        self.list_of_arrays_of_coeff = []
        self.length_of_signal = 0.0
        self.number_of_chunks = 1
        self.overlap_value = 0
        self.overlaped_indices = 0
        self.length_of_chunk = self.length_of_signal
        self.chunks_margins = [[0, 6]]
        self.generate_botton_is_clicked = False
        self.pen1 = pyqtgraph.mkPen((255, 0, 0), width=3)
        self.pen2 = pyqtgraph.mkPen((0, 255, 0), width=3)
        self.gui.inter_settings_botton.clicked.connect(
            self.show_interpolation_settings)
        self.gui.error_settings_botton.clicked.connect(
            self.show_error_map_settings)
        self.gui.open_action.triggered.connect(self.open_signal)
        self.gui.degree_spinbox.valueChanged.connect(
            self.degree_of_polynomial_changed)
        self.gui.num_of_chunks_spinbox.valueChanged.connect(
            self.interpolation_extrapolation)
        self.gui.points_slider.valueChanged.connect(
            self.interpolation_extrapolation)
        self.gui.overlap_slider.valueChanged.connect(
            self.interpolation_extrapolation)
        self.gui.chunk_number_combobx.activated.connect(
            self.generate_latex_equation)
        self.gui.generat_cancel_botton.clicked.connect(self.generate_error_map)
        self.gui.x_combobox.addItems(
            ["number of chunks", "degree of polynomial", "overlaping between chunks"])
        self.gui.y_combobox.addItems(
            ["number of chunks", "degree of polynomial", "overlaping between chunks"])
        self.gui.z_combobox.addItems(
            ["number of chunks", "degree of polynomial", "overlaping between chunks"])
        self.gui.x_combobox.setCurrentText("number of chunks")
        self.gui.y_combobox.setCurrentText("degree of polynomial")
        self.gui.z_combobox.setCurrentText("overlaping between chunks")
        self.worker = worker()
        self.worker_thread = qtc.QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()
        self.construct_matrix.connect(self.worker.constuct_error_map)
        self.worker.matrix_constructed.connect(self.draw_error_map)
        self.worker.progress_par_icreased.connect(self.update_progress_bar)
        self.cancel_the_error_map_operation.connect(
            self.worker.cancel_operaion)
        # self.transpose_the_matrix.connect(self.worker.transpose_matrix)
        self.show()

    # def update_axies_comboboxes(self):
    #     if self.gui.x_axis_combobox.currentText() == "number of chunks":
    #         self.gui.y_axis_combobox.setCurrentText("degree of polynomial")
    #     else:
    #         self.gui.y_axis_combobox.setCurrentText("number of chunks")
    #     self.transpose_the_matrix.emit()
    #"number of chunks","degree of polynomial","overlaping between chunks"
    # def x_combobox_changing(self):
    #     self.gui.x_combobox.model().item(0).setEnabled(True)
    #     self.gui.x_combobox.model().item(1).setEnabled(True)
    #     self.gui.x_combobox.model().item(2).setEnabled(True)
    #     self.gui.y_combobox.model().item(0).setEnabled(True)
    #     self.gui.y_combobox.model().item(1).setEnabled(True)
    #     self.gui.y_combobox.model().item(2).setEnabled(True)
    #     self.gui.z_combobox.model().item(0).setEnabled(True)
    #     self.gui.z_combobox.model().item(1).setEnabled(True)
    #     self.gui.z_combobox.model().item(2).setEnabled(True)
    #     if self.gui.x_combobox.currentText() == "number of chunks":
    #         self.gui.y_combobox.setCurrentText("degree of polynomial")
    #         self.gui.z_combobox.setCurrentText("overlaping between chunks")
    #         self.gui.y_combobox.model().item(0).setEnabled(False)
    #         self.gui.z_combobox.model().item(0).setEnabled(False)
    #     elif self.gui.x_combobox.currentText() == "degree of polynomial":
    #         self.gui.y_combobox.setCurrentText("number of chunks")
    #         self.gui.z_combobox.setCurrentText("overlaping between chunks")
    #         self.gui.y_combobox.model().item(1).setEnabled(False)
    #         self.gui.z_combobox.model().item(1).setEnabled(False)
    #     else:
    #         self.gui.y_combobox.setCurrentText("degree of polynomial")
    #         self.gui.z_combobox.setCurrentText("number of chunks")
    #         self.gui.y_combobox.model().item(2).setEnabled(False)
    #         self.gui.z_combobox.model().item(2).setEnabled(False)

    # def y_combobox_changing(self):
    #     self.gui.x_combobox.model().item(0).setEnabled(True)
    #     self.gui.x_combobox.model().item(1).setEnabled(True)
    #     self.gui.x_combobox.model().item(2).setEnabled(True)
    #     self.gui.y_combobox.model().item(0).setEnabled(True)
    #     self.gui.y_combobox.model().item(1).setEnabled(True)
    #     self.gui.y_combobox.model().item(2).setEnabled(True)
    #     self.gui.z_combobox.model().item(0).setEnabled(True)
    #     self.gui.z_combobox.model().item(1).setEnabled(True)
    #     self.gui.z_combobox.model().item(2).setEnabled(True)
    #     if self.gui.y_combobox.currentText() == "number of chunks":
    #         self.gui.x_combobox.setCurrentText("degree of polynomial")
    #         self.gui.z_combobox.setCurrentText("overlaping between chunks")
    #         self.gui.x_combobox.model().item(0).setEnabled(False)
    #         self.gui.z_combobox.model().item(0).setEnabled(False)
    #     elif self.gui.y_combobox.currentText() == "degree of polynomial":
    #         self.gui.x_combobox.setCurrentText("number of chunks")
    #         self.gui.z_combobox.setCurrentText("overlaping between chunks")
    #         self.gui.x_combobox.model().item(1).setEnabled(False)
    #         self.gui.z_combobox.model().item(1).setEnabled(False)
    #     else:
    #         self.gui.x_combobox.setCurrentText("degree of polynomial")
    #         self.gui.z_combobox.setCurrentText("number of chunks")
    #         self.gui.x_combobox.model().item(2).setEnabled(False)
    #         self.gui.z_combobox.model().item(2).setEnabled(False)

    # def z_combobox_changing(self):
    #     self.gui.x_combobox.model().item(0).setEnabled(True)
    #     self.gui.x_combobox.model().item(1).setEnabled(True)
    #     self.gui.x_combobox.model().item(2).setEnabled(True)
    #     self.gui.y_combobox.model().item(0).setEnabled(True)
    #     self.gui.y_combobox.model().item(1).setEnabled(True)
    #     self.gui.y_combobox.model().item(2).setEnabled(True)
    #     self.gui.z_combobox.model().item(0).setEnabled(True)
    #     self.gui.z_combobox.model().item(1).setEnabled(True)
    #     self.gui.z_combobox.model().item(2).setEnabled(True)
    #     if self.gui.z_combobox.currentText() == "number of chunks":
    #         self.gui.y_combobox.setCurrentText("degree of polynomial")
    #         self.gui.x_combobox.setCurrentText("overlaping between chunks")
    #         self.gui.y_combobox.model().item(0).setEnabled(False)
    #         self.gui.x_combobox.model().item(0).setEnabled(False)
    #     elif self.gui.z_combobox.currentText() == "degree of polynomial":
    #         self.gui.y_combobox.setCurrentText("number of chunks")
    #         self.gui.x_combobox.setCurrentText("overlaping between chunks")
    #         self.gui.y_combobox.model().item(1).setEnabled(False)
    #         self.gui.x_combobox.model().item(1).setEnabled(False)
    #     else:
    #         self.gui.y_combobox.setCurrentText("degree of polynomial")
    #         self.gui.x_combobox.setCurrentText("number of chunks")
    #         self.gui.y_combobox.model().item(2).setEnabled(False)
    #         self.gui.x_combobox.model().item(2).setEnabled(False)

    # def transpose_matrix(self):
    #     self.transpose_the_matrix.emit()

    @qtc.pyqtSlot()
    def update_progress_bar(self):
        self.gui.progressBar.setValue(self.gui.progressBar.value()+1)
        if self.gui.progressBar.value() == 100:
            self.gui.progress_bar_widget.hide()

    def generate_error_map(self):
        if self.generate_botton_is_clicked == False:
            self.generate_botton_is_clicked = True
            self.gui.generat_cancel_botton.setText("Cancel")
            self.construct_matrix.emit(
                self.main_graph_data, self.main_graph_time, self.gui.x_spinbox.value(), self.gui.y_spinbox.value(), self.gui.z_spinbox.value(), [self.gui.x_combobox.currentText(), self.gui.y_combobox.currentText(), self.gui.z_combobox.currentText()])
            self.gui.progress_bar_widget.show()
        else:
            self.generate_botton_is_clicked = False
            self.gui.generat_cancel_botton.setText("generate")
            self.gui.error_map_widget.hide()
            self.gui.progress_bar_widget.hide()
            self.cancel_the_error_map_operation.emit()
            self.color_bar.remove()
            self.gui.progressBar.setValue(0)
            self.worker.Matrix = np.arange(float(self.gui.x_spinbox.value()*self.gui.y_spinbox.value())).reshape(
            self.gui.x_spinbox.value(), self.gui.y_spinbox.value())

    @qtc.pyqtSlot(np.ndarray)
    def draw_error_map(self, matrix):
        # self.Matrix = np.arange(float(self.number_of_chunks*self.gui.degree_spinbox.value())).reshape(
        #     self.number_of_chunks, self.gui.degree_spinbox.value())
        for i in reversed(range(self.gui.map_layout.count())):
            self.gui.map_layout.itemAt(i).widget().setParent(None)
        # a = np.random.random((16, 16))
        self.sc.axes.set_xticks(np.arange(self.gui.x_spinbox.value()))
        self.sc.axes.set_xticklabels(np.arange(1,(self.gui.x_spinbox.value()+1)))
        self.sc.axes.set_yticks(np.arange(self.gui.y_spinbox.value()))
        self.sc.axes.set_yticklabels(np.arange(1,(self.gui.y_spinbox.value()+1)))
        # print(self.Matrix)
        img = self.sc.axes.imshow(matrix,
                            cmap='hot', origin="lower", interpolation='nearest',vmin=0,vmax=0.3)
        self.color_bar = plt.colorbar(img, ax=self.sc.axes)
        self.sc.draw()
        # sc.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
        self.gui.map_layout.addWidget(self.sc)
        self.gui.error_map_widget.show()

    def degree_of_polynomial_changed(self):
        slider_value = int(self.gui.points_slider.value())
        self.gui.points_label.setText(f"{str(slider_value)}%")
        # num_of_chunks = self.gui.num_of_chunks_spinbox.value()
        # degree_of_poynomial = self.gui.degree_spinbox.value()
        if slider_value != 100:
            self.gui.num_of_chunks_spinbox.setValue(1)
            self.gui.overlap_slider.setValue(0)
            self.gui.overlap_label.setText("0%")
            self.gui.chunk_number_combobx.clear()
            self.gui.chunk_number_combobx.addItem("1")
            self.extrapolation()
        else:
            overlap_value = int(self.gui.overlap_slider.value())
            self.gui.overlap_label.setText(f"{str(overlap_value)}%")
            self.interpolation_process()

    def interpolation_extrapolation(self):
        slider_value = int(self.gui.points_slider.value())
        self.gui.points_label.setText(f"{str(slider_value)}%")
        # num_of_chunks = self.gui.num_of_chunks_spinbox.value()
        # degree_of_poynomial = self.gui.degree_spinbox.value()
        if slider_value != 100:
            self.gui.num_of_chunks_spinbox.setValue(1)
            self.gui.overlap_slider.setValue(0)
            self.gui.overlap_label.setText("0%")
            self.gui.chunk_number_combobx.clear()
            self.gui.chunk_number_combobx.addItem("1")
            self.extrapolation()
        else:
            overlap_value = int(self.gui.overlap_slider.value())
            self.gui.overlap_label.setText(f"{str(overlap_value)}%")
            self.construct_chunks_margins()
            self.interpolation_process()

    def generate_root_mean_square(self, original_data, fitted_data):
        if len(original_data) > len(fitted_data):
            original_data = original_data[:len(fitted_data)]
        elif len(original_data) < len(fitted_data):
            fitted_data = fitted_data[:len(original_data)]
        MSE = np.mean(np.square(np.subtract(original_data, fitted_data)))
        RMSE = math.sqrt(MSE)
        return RMSE

    def generate_percentage_error(self, error):
        percentage_error = round((error*100),2)
        self.gui.percentage_error_label.setText(str(percentage_error))

    def open_signal(self):
        self.main_graph_data.clear()
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Open File", r"D:/SBME_3/1st_term/DSP/tasks/task4/Signals", "Signals(*.csv)", options=options)
        if file_path == '':
            self.warning_message("please choose file!! ")
        else:
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                # next(csv_reader)
                for line in csv_reader:
                    self.main_graph_data.append(line[1])
            for counter in range(len(self.main_graph_data)):
                self.main_graph_data[counter] = float(
                    self.main_graph_data[counter])
            self.length_of_signal = len(self.main_graph_data)/100
            # print(self.length_of_signal)
            self.main_graph_time = list(
                np.arange(0, self.length_of_signal, 0.01))
            for index in range(len(self.main_graph_time)):
                self.main_graph_time[index] = round(
                    self.main_graph_time[index], 2)
            # print(self.main_graph_time)
            # print(
            #     f"len of array of data and time = {len(self.main_graph_time)}")
            self.plot_main_graph()
            # self.interpolation_process()

    def construct_chunks_margins(self):
        self.chunks_margins.clear()
        self.number_of_chunks = self.gui.num_of_chunks_spinbox.value()
        self.length_of_chunk = round(
            (self.length_of_signal/self.number_of_chunks), 2)
        self.overlap_value = self.gui.overlap_slider.value()
        self.overlaped_indices = int(self.length_of_chunk *
                                     self.overlap_value)
        print(self.length_of_chunk)
        print(f"overlaped_indices = {self.overlaped_indices}")
        counter = 0
        self.gui.chunk_number_combobx.clear()
        for chunk in range(0, self.number_of_chunks):
            if (round(((chunk)*self.length_of_chunk), 2)) > 5.99:
                left_index = 599 - (self.overlaped_indices*counter)
            else:
                left_index = self.main_graph_time.index(
                    round((chunk*self.length_of_chunk), 2)) - (self.overlaped_indices*counter)
            if (round(((chunk+1)*self.length_of_chunk), 2)) > 5.99:
                right_index = 599 - (self.overlaped_indices*counter)
            else:
                right_index = self.main_graph_time.index(round((
                    (chunk+1)*self.length_of_chunk), 2)) - (self.overlaped_indices*counter)
            counter += 1
            self.gui.chunk_number_combobx.addItem(
                f"{counter}")
            # right_index = self.main_graph_time.index(
            #     round(((chunk+1)*self.length_of_chunk), 2))
            # print(self.main_graph_time.index(left_margin))
            print(left_index)
            print(right_index)
            self.chunks_margins.append([left_index, right_index])

    def interpolation_process(self):
        # self.new_ys.clear()
        self.new_y.clear()
        self.new_x.clear()
        self.list_of_arrays_of_coeff.clear()
        deg = self.gui.degree_spinbox.value()
        array_of_coeff = np.array([])
        for chunk in self.chunks_margins:
            array_of_coeff = np.polyfit(
                self.main_graph_time[chunk[0]:chunk[1]], self.main_graph_data[chunk[0]:chunk[1]], deg)
            self.list_of_arrays_of_coeff.append(array_of_coeff)
            for item in self.main_graph_time[chunk[0]:chunk[1]]:
                new_item = 0
                for i in range(0, deg+1):
                    new_item += (array_of_coeff[deg-i]*(item**i))
                self.new_y.append(new_item)
            new_time = self.main_graph_time[chunk[0]:chunk[1]]
            self.new_x.extend(new_time)
            # print(self.main_graph_time[chunk[0]:chunk[1]])
        # print(len(self.new_x))
        # print(len(self.new_y))  # self.new_ys.append(new_y)
        self.plot_modified_graph()
        self.generate_percentage_error(self.generate_average_error())

    def extrapolation(self):
        # self.gui.points_label.setText(
        #     str(int(self.gui.points_slider.value())) + "%")
        self.new_y.clear()
        self.list_of_arrays_of_coeff.clear()
        deg = self.gui.degree_spinbox.value()
        current_value = int(self.gui.points_slider.value())
        extrapolation_length = int(
            (len(self.main_graph_data)*current_value)/100)
        array_of_coeff = np.polyfit(
            self.main_graph_time[:extrapolation_length], self.main_graph_data[:extrapolation_length], deg)
        self.list_of_arrays_of_coeff.append(array_of_coeff)
        for item in self.main_graph_time:
            new_item = 0
            for i in range(0, deg+1):
                new_item += (array_of_coeff[deg-i]*(item**i))
            self.new_y.append(new_item)
        self.plot_extrapolation_graph()
        for i in reversed(range(self.gui.fitting_layout.count())):
            self.gui.fitting_layout.itemAt(i).widget().setParent(None)
        self.gui.fitting_layout.addWidget(MathTextLabel(self.LateX(array_of_coeff), self),
                                          alignment=Qt.AlignHCenter)
        self.generate_percentage_error(
            self.generate_root_mean_square(self.main_graph_data, self.new_y))

    # def generate_average_error(self):
    #     list_of_errors = []
    #     left = 0
    #     right = len(self.main_graph_data)
    #     left_overlaped = 0
    #     for index, chunk in enumerate(self.chunks_margins):
    #         if index == 0:
    #             left = 0
    #             right = chunk[1] - self.overlaped_indices
    #             left_overlaped = 0
    #         elif index == (len(self.chunks_margins)-1):
    #             right = len(self.main_graph_data)
    #             left = chunk[0] + self.overlaped_indices
    #             left_overlaped = left - self.overlaped_indices
    #         else:
    #             left = chunk[0] + self.overlaped_indices
    #             right = chunk[1] - self.overlaped_indices
    #             left_overlaped = left - self.overlaped_indices
    #         list_of_errors.append(self.generate_root_mean_square(
    #             self.main_graph_data[left:right], self.new_y[left:right]))
    #         list_of_errors.append(self.generate_root_mean_square(
    #             self.main_graph_data[left_overlaped:left+1], self.new_y[left_overlaped:left+1]))
    #     # print(mean(list_of_errors))
    #     return mean(list_of_errors)

    def generate_average_error(self):
        list_of_errors = []
        left = 0
        right = len(self.main_graph_data)
        left_overlaped = 0
        for index, chunk in enumerate(self.chunks_margins):
            if index == 0:
                left = 0
                right = chunk[1] - self.overlaped_indices
                left_overlaped = 0
            elif index == (len(self.chunks_margins)-1):
                right = len(self.main_graph_data)
                left = chunk[0] + self.overlaped_indices
                left_overlaped = left - self.overlaped_indices
            else:
                left = chunk[0] + self.overlaped_indices
                right = chunk[1] - self.overlaped_indices
                left_overlaped = left - self.overlaped_indices
            list_of_errors.append(self.generate_root_mean_square(
                self.main_graph_data[left:right], self.new_y[left:right]))
            list_of_errors.append(self.generate_root_mean_square(
                self.main_graph_data[left_overlaped:left+1], self.new_y[left_overlaped:left+1]))
        # print(mean(list_of_errors))
        return mean(list_of_errors)

    def generate_latex_equation(self):
        number_of_chunk = int(self.gui.chunk_number_combobx.currentText())
        array_of_coeff = self.list_of_arrays_of_coeff[number_of_chunk-1]
        text = self.LateX(array_of_coeff)
        for i in reversed(range(self.gui.fitting_layout.count())):
            self.gui.fitting_layout.itemAt(i).widget().setParent(None)
        self.gui.fitting_layout.addWidget(MathTextLabel(text, self),
                                          alignment=Qt.AlignHCenter)

    def plot_extrapolation_graph(self):
        self.gui.main_graph.plotItem.clear()
        self.gui.main_graph.plotItem.plot(
            self.main_graph_time, self.main_graph_data, symbol="o", pen=None)
        self.gui.main_graph.plotItem.plot(
            self.main_graph_time, self.new_y, pen=self.pen1)

    def plot_modified_graph(self):
        counter = 0
        self.gui.main_graph.plotItem.clear()
        print(f"length of main graph time = {len(self.main_graph_time)}")
        print(f"length of main graph data = {len(self.main_graph_data)}")
        print(f"length of new y is = {len(self.new_y)}")
        self.gui.main_graph.plotItem.plot(
            self.main_graph_time[:len(self.new_y)], self.main_graph_data[:len(self.new_y)], symbol="o", pen=None)
        # for data in self.new_ys:
        #     left = self.chunks_margins[counter][0]
        #     right = self.chunks_margins[counter][1]
        #     counter += 1

        self.gui.main_graph.plotItem.plot(
            self.new_x, self.new_y, pen=self.pen1)

    def plot_main_graph(self):
        self.gui.main_graph.plotItem.clear()
        plotitem = self.gui.main_graph.plotItem.plot(
            self.main_graph_time, self.main_graph_data, symbol='o', pen=None)
        print(len(self.main_graph_data))
        print(len(self.main_graph_time))

    def warning_message(self, message):
        self.msg.setInformativeText(message)
        # logger.warning(message)
        self.msg.exec_()

    def show_error_map_settings(self):
        self.gui.interpolation_settings_widget.hide()
        self.gui.error_settings_widget.show()

    def show_interpolation_settings(self):
        self.gui.error_settings_widget.hide()
        self.gui.interpolation_settings_widget.show()

    def LateX(self, array_of_coeff):
        text = ""
        # y = range(int(self.gui.degree_spinbox.value))
        for i in range(len(array_of_coeff)):
            if i != len(self.main_graph_time)-1:
                text += str("{:.3f}".format(array_of_coeff[i])) + "$X^" + "{"+str(
                    len(array_of_coeff)-1-i)+"}"+"$+"
            else:
                text += str("{:.3f}".format(array_of_coeff[i]))
        return text


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


if __name__ == '__main__':
    import sys
    warnings.simplefilter('ignore', np.RankWarning)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    Interpoltor = MainWindow()
    logger = logging.getLogger("main.py")
    logger.setLevel(level=logging.DEBUG)
    logging.basicConfig(filename="logging_file.log")
    logger.info("lunching of the Application ")
    sys.exit(app.exec_())
