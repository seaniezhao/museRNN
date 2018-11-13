import torch
import os
import math
import ast
import random
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import fnmatch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import xlrd

class MeasureDataset(torch.utils.data.Dataset):
    def __init__(self, path, isTrain = True):

        self.h5file = h5py.File(path, "r")
        self.isTrain = isTrain

        self.tl = math.floor(len(self.h5file) * 0.8)

    def __getitem__(self, index):


        if not self.isTrain:
            index += self.tl

        key = str(index)

        g = self.h5file[key]
        file_ins_notes = {}
        file_chord_cut = []
        for k in g:
            if k == 'label':
                file_chord_cut = g[k]
            else:
                file_ins_notes[k] = g[k]

        r_value = None
        for ins in file_ins_notes:

            #总共19种乐器 one-hot
            ins_arr = [0 for i in range(19)]
            ins_num = int(ins)
            ins_arr[ins_num] = 1

            ins_arr = np.tile(ins_arr, (len(file_chord_cut), 1))

            temp = np.array(file_ins_notes[ins])
            c_array = np.array(file_chord_cut).reshape(1, -1)
            r_value_seg = np.append( c_array, temp, axis=0)

            r_value_seg = np.concatenate((r_value_seg, ins_arr.T), axis=2)

            if r_value is None:
                r_value = r_value_seg
            else:
                r_value = np.concatenate((r_value, r_value_seg), axis=1)

        return r_value

    def __len__(self):



        if self.isTrain:
            return self.tl
        else:
            return len(self.h5file) - self.tl






class FileLoader:
    def __init__(self):
        self.data_path = './data/1/'

        self.cache_path = './data/file_data.hdf5'

        self.file_datas = []

        if os.path.exists(self.cache_path):
            f = h5py.File(self.cache_path, "r")

            for key in f:
                g = f[key]
                file_ins_notes = {}
                file_chord_cut = []
                for k in g:
                    if k =='label':
                        file_chord_cut=g[k]
                    else:
                        file_ins_notes[k] = g[k]
                self.file_datas.append((file_ins_notes, file_chord_cut))
        else:
            file_num = 0
            for dirpath, dirs, files in os.walk(self.data_path):
                for file in files:
                    if 'notes' in file:
                        file_chord_cut = []
                        file_ins_notes = {}
                        #key:col value:len
                        len_chord_cut = {}

                        path = os.path.join(dirpath, file)
                        workbook = xlrd.open_workbook(path)
                        sheet = workbook.sheet_by_name('map')


                        for row in range(sheet.nrows):

                            for col in range(1, sheet.ncols):
                                xl_cell = sheet.cell(row, col)
                                if xl_cell == None:
                                    continue
                                #和弦行 - 和弦label
                                if row == 2:
                                    chord_array = ast.literal_eval(xl_cell.value)
                                    #如果和弦中有None说明此小节没有音 直接跳过本列
                                    if chord_array is None:
                                        continue
                                    current_chord = ''
                                    chord_cut = []
                                    for chord_str in chord_array:
                                        #'-' 表示补位，这里不考虑补位
                                        if chord_str == '-':
                                            break
                                        if len(chord_cut) == 0:
                                            chord_cut.append(0)
                                            current_chord = chord_str
                                        else:
                                            if chord_str != current_chord and chord_str != '-':
                                                chord_cut.append(1)
                                                current_chord = chord_str
                                            else:
                                                chord_cut.append(0)
                                    file_chord_cut.extend(chord_cut)
                                    len_chord_cut[col] = len(chord_cut)
                                    #print(chord_array)
                                    #print(chord_cut)
                                # 音符数据
                                elif row > 2:
                                    notes_array_2dim = []
                                    cell_values = xl_cell.value.split('\n')
                                    if len(cell_values)<7:
                                        continue
                                    cut_index = -1
                                    for cell_value in cell_values:
                                        if cell_value =='':
                                            continue
                                        note_array = ast.literal_eval(cell_value)
                                        try:
                                            cut_index = note_array.index('-')
                                        except:
                                            cut_index = -1
                                        notes_array_2dim.append(note_array)
                                    notes_array_2dim = np.array(notes_array_2dim)
                                    if cut_index != -1:
                                        notes_array_2dim = notes_array_2dim[:,0:cut_index]

                                    notes_array_2dim = notes_array_2dim.astype(np.int64)
                                    if len_chord_cut[col] !=len(notes_array_2dim[0]):
                                        print("wtf")
                                    if row-3 not in file_ins_notes:
                                        file_ins_notes[row-3] = notes_array_2dim
                                    else:
                                        temp = file_ins_notes[row-3]
                                        file_ins_notes[row-3] = np.concatenate((temp, notes_array_2dim),axis=1)

                        self.file_datas.append((file_ins_notes, file_chord_cut))
                        for k,v in file_ins_notes.items():
                            if len(v[0])!=len(file_chord_cut):
                                print(path, ' dim error')
                                break
                        file_num += 1
                        print(file_num,'file has been processed')


            f = h5py.File(self.cache_path, "w")
            index = 0
            for file_data in self.file_datas:
                g = f.create_group(str(index))
                g['label'] = file_data[1]
                for k,v in file_data[0].items():
                    g[str(k)] = v

                index+=1

        print('fileloader init finish')


    #通过乐器编号获取数据
    def get_data_by_ins_index(self, ins):
        ins = str(ins)
        r_value = None
        for file_data in self.file_datas:
            if ins in file_data[0]:
                temp = np.array(file_data[0][ins])
                c_array = np.array(file_data[1]).reshape(1,-1)
                r_value_seg = np.append(temp,c_array,axis=0)

                if r_value is None:
                    r_value = r_value_seg
                else:
                    r_value = np.concatenate((r_value,r_value_seg),axis=1)

        return


    def get_data_by_file(self,index):

        g = self.file_datas[index]
        file_ins_notes = g[0]
        file_chord_cut = g[1]


        r_value = None
        for ins in file_ins_notes:

            # 总共19种乐器 one-hot
            ins_arr = [0 for i in range(19)]
            ins_num = int(ins)
            ins_arr[ins_num] = 1

            ins_arr = np.tile(ins_arr, (len(file_chord_cut), 1)).T

            temp = np.array(file_ins_notes[ins])
            c_array = np.array(file_chord_cut).reshape(1, -1)
            r_value_seg = np.append(c_array, temp, axis=0)

            r_value_seg = np.concatenate((r_value_seg, ins_arr), axis=0)

            if r_value is None:
                r_value = r_value_seg
            else:
                r_value = np.concatenate((r_value, r_value_seg), axis=1)

        return r_value



    def get_file_len(self):
        return len(self.file_datas)


if __name__ =="__main__":
    x = [i for i in range(570)]
    slice = random.sample(x, 100)
    print(slice)