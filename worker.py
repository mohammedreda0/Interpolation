import math
from PyQt5.QtCore import Qt, right
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtGui, QtWidgets
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout
from numpy.core.defchararray import array, count
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import angle
# from matplotlib.colors import same_color
import pyqtgraph
from pyqtgraph.Point import Point
from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
from GUI import Ui_MainWindow
import sys
from PyQt5 import QtCore as qtc
import numpy as np
from math import *
import logging
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


class worker(QtCore.QObject):
    matrix_constructed = qtc.pyqtSignal(np.ndarray)
    progress_par_icreased = qtc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_chunks = 20
        self.degree_of_polynomial = 20
        self.Matrix = np.arange(float(self.number_of_chunks*self.degree_of_polynomial)).reshape(
            self.number_of_chunks, self.degree_of_polynomial)
        self.chunks_margins = [[]]
        self.main_graph_data = []
        self.main_graph_time = []
        self.new_y = []
        self.new_x = []
        self.operation_number = 0
        self.cancel_is_clicked = False
        self.squares_per_1 = 0
        self.x_axis = 10
        self.y_axis = 10
        self.z_axis = 10

    @qtc.pyqtSlot()
    def transpose_matrix(self):
        self.Matrix = np.transpose(self.Matrix)
        self.matrix_constructed.emit(self.Matrix)

    @qtc.pyqtSlot()
    def cancel_operaion(self):
        self.cancel_is_clicked = True

    @qtc.pyqtSlot(list, list, int, int, int, list)
    def constuct_error_map(self, main_graph_data, main_graph_time, x_spin, y_spin, z_spin, comboboxes_text):
        self.cancel_is_clicked = False
        self.x_axis = x_spin
        self.y_axis = y_spin
        self.z_axis = z_spin
        self.squares_per_1 = int(
            self.x_axis*self.y_axis)/100
        self.Matrix = np.arange(float(self.x_axis*self.y_axis)).reshape(
            self.y_axis, self.x_axis)
        if comboboxes_text[0] == "number of chunks":
            if comboboxes_text[1] == "degree of polynomial":
                self.operation_number = 1
            else:
                self.operation_number = 2
        elif comboboxes_text[0] == "degree of polynomial":
            if comboboxes_text[1] == "number of chunks":
                self.operation_number = 3
            else:
                self.operation_number = 4
        else:
            if comboboxes_text[1] == "number of chunks":
                self.operation_number = 5
            else:
                self.operation_number = 6
        print(self.operation_number)
        print(self.x_axis)
        print(self.y_axis)
        print(self.z_axis)
        # print(main_graph_data[100])
        # print(main_graph_time[100])
        self.construct_matrix(main_graph_data, main_graph_time)
        # print(self.Matrix)
        self.matrix_constructed.emit(self.Matrix)

    def construct_matrix(self, main_graph_data, main_graph_time):
        if self.operation_number == 1:
            number_of_squares = 0
            for y_value in range(self.y_axis):
                for x_value in range(1, self.x_axis+1):
                    # print(self.x_axis)
                    # print(self.y_axis)
                    self.construct_chunks_margins(
                        x_value, main_graph_data, main_graph_time, self.z_axis)
                    self.Matrix[y_value][x_value -
                                         1] = self.interpolation_process(y_value)
                    # print(
                    #     f"the average error = {round(self.interpolation_process(degree),2)}")
                    # print(self.Matrix.dtype)
                    number_of_squares += 1
                    if number_of_squares == self.squares_per_1:
                        self.progress_par_icreased.emit()
                        number_of_squares = 0
                    # print(self.cancel_is_clicked)
                    if self.cancel_is_clicked == True:
                        return
        elif self.operation_number == 2:
            number_of_squares = 0
            for y_value in range(self.y_axis):
                for x_value in range(1, self.x_axis+1):
                    self.construct_chunks_margins(
                        x_value, main_graph_data, main_graph_time, y_value)
                    self.Matrix[y_value][x_value -
                                         1] = self.interpolation_process(self.z_axis)
                    # print(
                    #     f"the average error = {round(self.interpolation_process(degree),2)}")
                    # print(self.Matrix.dtype)
                    number_of_squares += 1
                    if number_of_squares == self.squares_per_1:
                        self.progress_par_icreased.emit()
                        number_of_squares = 0
                    if self.cancel_is_clicked == True:
                        return
        elif self.operation_number == 3:
            number_of_squares = 0
            for y_value in range(1, self.y_axis+1):
                for x_value in range(self.x_axis):
                    self.construct_chunks_margins(
                        y_value, main_graph_data, main_graph_time, self.z_axis)
                    self.Matrix[y_value -
                                1][x_value] = self.interpolation_process(x_value)
                    # print(
                    #     f"the average error = {round(self.interpolation_process(degree),2)}")
                    # print(self.Matrix.dtype)
                    number_of_squares += 1
                    if number_of_squares == self.squares_per_1:
                        self.progress_par_icreased.emit()
                        number_of_squares = 0
                    if self.cancel_is_clicked == True:
                        return
        elif self.operation_number == 4:
            number_of_squares = 0
            for y_value in range(self.y_axis):
                for x_value in range(self.x_axis):
                    self.construct_chunks_margins(
                        self.z_axis, main_graph_data, main_graph_time, y_value)
                    self.Matrix[y_value][x_value] = self.interpolation_process(
                        x_value)
                    # print(
                    #     f"the average error = {round(self.interpolation_process(degree),2)}")
                    # print(self.Matrix.dtype)
                    number_of_squares += 1
                    if number_of_squares == self.squares_per_1:
                        self.progress_par_icreased.emit()
                        number_of_squares = 0
                    if self.cancel_is_clicked == True:
                        return
        elif self.operation_number == 5:
            number_of_squares = 0
            for y_value in range(1, self.y_axis+1):
                for x_value in range(self.x_axis):
                    self.construct_chunks_margins(
                        y_value, main_graph_data, main_graph_time, x_value)
                    self.Matrix[y_value -
                                1][x_value] = self.interpolation_process(self.z_axis)
                    # print(
                    #     f"the average error = {round(self.interpolation_process(degree),2)}")
                    # print(self.Matrix.dtype)
                    number_of_squares += 1
                    if number_of_squares == self.squares_per_1:
                        self.progress_par_icreased.emit()
                        number_of_squares = 0
                    if self.cancel_is_clicked == True:
                        return
        elif self.operation_number == 6:
            number_of_squares = 0
            for y_value in range(self.y_axis):
                for x_value in range(self.x_axis):
                    self.construct_chunks_margins(
                        self.z_axis, main_graph_data, main_graph_time, x_value)
                    self.Matrix[y_value][x_value] = self.interpolation_process(
                        y_value)
                    # print(
                    #     f"the average error = {round(self.interpolation_process(degree),2)}")
                    # print(self.Matrix.dtype)
                    number_of_squares += 1
                    if number_of_squares == self.squares_per_1:
                        self.progress_par_icreased.emit()
                        number_of_squares = 0
                    if self.cancel_is_clicked == True:
                        return

    def construct_chunks_margins(self, current_number_of_chunks, main_graph_data, main_graph_time, overlap):
        self.chunks_margins.clear()
        self.main_graph_data = main_graph_data
        self.main_graph_time = main_graph_time
        length_of_signal = len(self.main_graph_data)/100
        self.length_of_chunk = round(
            (length_of_signal/current_number_of_chunks), 2)
        self.overlap_value = overlap
        self.overlaped_indices = int(self.length_of_chunk *
                                     self.overlap_value)
        # print(self.length_of_chunk)
        # print(f"overlaped_indices = {self.overlaped_indices}")
        counter = 0
        for chunk in range(0, current_number_of_chunks):
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
            # self.gui.chunk_number_combobx.addItem(
            #     f"{counter}")
            # right_index = self.main_graph_time.index(
            #     round(((chunk+1)*self.length_of_chunk), 2))
            # print(self.main_graph_time.index(left_margin))
            # print(left_index)
            # print(right_index)
            self.chunks_margins.append([left_index, right_index])
        # print("chunks margins has endend successfully")

    def interpolation_process(self, degree):
        # print(f"len of main_graph_time = {len(self.main_graph_time)}")
        # self.new_ys.clear()
        self.new_y.clear()
        self.new_x.clear()
        # self.list_of_arrays_of_coeff.clear()
        deg = degree
        array_of_coeff = np.array([])
        for chunk in self.chunks_margins:
            # print(len(self.chunks_margins))
            # print(self.main_graph_time[chunk[0]:chunk[1]])
            # print("dslkfjsdklfjadslkfjsadlkfjsdjfskldjfalskdfjsakldjfsklfg")
            # print(self.main_graph_data[chunk[0]:chunk[1]])
            array_of_coeff = np.polyfit(
                self.main_graph_time[chunk[0]:chunk[1]], self.main_graph_data[chunk[0]:chunk[1]], deg)
            # self.list_of_arrays_of_coeff.append(array_of_coeff)
            for item in self.main_graph_time[chunk[0]:chunk[1]]:
                new_item = 0
                for i in range(0, deg+1):
                    new_item += (array_of_coeff[deg-i]*(item**i))
                self.new_y.append(new_item)
            # new_time = self.main_graph_time[chunk[0]:chunk[1]]
            # self.new_x.extend(new_time)
            # print(self.main_graph_time[chunk[0]:chunk[1]])
        # print(len(self.new_x))
        # print(len(self.new_y))  # self.new_ys.append(new_y)
        # self.plot_modified_graph()
        # print(self.main_graph_data)
        # print(self.new_y)
        return self.generate_average_error()

    def generate_root_mean_square(self, original_data, fitted_data):
        if len(original_data) > len(fitted_data):
            original_data = original_data[:len(fitted_data)]
        elif len(original_data) < len(fitted_data):
            fitted_data = fitted_data[:len(original_data)]
        MSE = np.mean(np.square(np.subtract(original_data, fitted_data)))
        RMSE = math.sqrt(MSE)
        return RMSE

    # def generate_percentage_error(self, error):
    #     percentage_error = error*100
    #     self.gui.percentage_error_label.setText(str(percentage_error))

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
