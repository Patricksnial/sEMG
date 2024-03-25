from tensorflow import keras
import numpy as np
from data import dataset
import pickle
import math
from activations import Mish

from optimizers import Ranger
import losses as l
import callbacks as cb
from layers import Attention, LayerNormalization
from data import dataset
from generator import generator

with keras.utils.custom_object_scope(
    {
        "Mish": Mish,
        "Ranger": Ranger,
        "Attention": Attention,
        "LayerNormalization": LayerNormalization,
        "l": l,
        "cb": cb,
        "dataset": dataset,
        "generator": generator,
    }
):
    model = keras.models.load_model("main.h5")
file_name_imu = "fai_hospital_7_imu_rec.p"
file_name_emg = "fai_hospital_7_emg_rec.p"
path = "./data/"


def find_nearest_in_sorted(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        print(array[idx - 1], value)
        return idx - 1
    else:
        print(array[idx], value)
        return idx


def _load_single_file(path, rep, ex):
    emg_file = pickle.load(open(path + file_name_imu, "rb"))
    imu_file = np.array(pickle.load(open(path + file_name_emg, "rb")))
    imu = []
    imu_ticks = []
    for i in range(imu_file.shape[0]):
        imu_ticks.append(imu_file[i][-1])
    for i in range(len(emg_file)):
        if emg_file[i][1] == True:
            emg_tick = emg_file[i][-1]
            nearest_imu_index = find_nearest_in_sorted(imu_ticks, emg_tick)
            imu.append(imu_file[nearest_imu_index][1])

    emg = []
    for i in emg_file:
        if i[1] == True:
            emg.append(i[0])
    emg = np.array(emg)
    lab = np.array([[ex]] * emg.shape[0])
    rep = np.array([[rep]] * emg.shape[0])
    del emg_file, imu_file
    return emg, lab, imu, rep
