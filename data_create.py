import cv2
import os
import random
import glob

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

class datacreate:
    def __init__(self, mag = 4):
        self.mag = mag
        self.num = 0
        self.LR_num = 4
        self.gaussian_kernelsize = 5

#任意のフレーム数を切り出すプログラム
    def datacreate(self,
                video_path,   #切り取る動画が入ったファイルのpath
                data_number,  #データセットの生成数
                cut_frame,    #1枚の画像から生成するデータセットの数
                cut_height,   #保存サイズ
                cut_width,
                ext='jpg'):

        #データセットのリストを生成
        low_data_list = [[] for _ in range(self.LR_num)]#LRは4枚で生成。
        high_data_list = []

        video_path = video_path + "/*"
        files = glob.glob(video_path)
    
        while self.num < data_number:
            file_num = random.randint(0, len(files) - 1)
            photo_files = glob.glob(files[file_num] + "/*")
            photo_num = random.randint(0, len(photo_files)- 1)
        
            img = cv2.imread(photo_files[photo_num])
            height, width = img.shape[:2]

            if cut_height > height or cut_width > width:
                break

            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            gray_img = color_img[:, :, 0]

            low_k_15 = cv2.GaussianBlur(gray_img, (self.gaussian_kernelsize, self.gaussian_kernelsize), 1.5, 1.5)
            low_k_18 = cv2.GaussianBlur(gray_img, (self.gaussian_kernelsize, self.gaussian_kernelsize), 1.8, 1.8)
            low_k_21 = cv2.GaussianBlur(gray_img, (self.gaussian_kernelsize, self.gaussian_kernelsize), 2.1, 2.1)
            low_bi = cv2.resize(gray_img , (int(width // self.mag), int(height // self.mag)), interpolation=cv2.INTER_CUBIC)
            low_bi = cv2.resize(low_bi , (int(width), int(height)), interpolation = cv2.INTER_CUBIC)

            for p in range(cut_frame):
                ram_h = random.randint(0, height - cut_height)
                ram_w = random.randint(0, width - cut_width)

                #HR生成・格納
                high_data_list.append(gray_img[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width])

                #LR生成・格納
                cut_low_k_15 = low_k_15[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
                cut_low_k_18 = low_k_18[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
                cut_low_k_21 = low_k_21[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
                cut_low_bi = low_bi[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]

                low_data_list[0].append(cut_low_k_15)
                low_data_list[1].append(cut_low_k_18)
                low_data_list[2].append(cut_low_k_21)
                low_data_list[3].append(cut_low_bi)

                self.num += 1

                if self.num == data_number:
                    break
    
        return low_data_list, high_data_list

    # def test_datacreate(self,
    #                 video_path,   #切り取る動画が入ったファイルのpath
    #                 data_number,  #データセットの生成数
    #                 cut_height,   #保存サイズ
    #                 cut_width,
    #                 ext='jpg'):

    #     #データセットのリストを生成
    #     low_data_list = [[] for _ in range(self.LR_num)]#LRは4枚で生成。
    #     high_data_list = []

    #     video_path = video_path + "/*"
    #     files = glob.glob(video_path)
    
    #     while self.num < data_number:
    #         file_num = random.randint(0, len(files) - 1)
    #         photo_files = glob.glob(files[file_num] + "/*")
    #         photo_num = random.randint(0, len(photo_files) - self.LR_num)

    #         img = cv2.imread(photo_files[photo_num])
    #         height, width = img.shape[:2]

    #         if cut_height > height or cut_width > width:
    #             break
            
    #         ram_h = random.randint(0, height - cut_height)
    #         ram_w = random.randint(0, width - cut_width)

    #         color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #         gray_img = color_img[:, :, 0]

    #         low_k_15 = cv2.GaussianBlur(gray_img,(3, 3), 1.5, 1.5)
    #         cut_low_bi = low_k_15[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
    #         low_data_list[0].append(cut_low_bi)

    #         for op in range(self.LR_num - 1):
    #             img = cv2.imread(photo_files[photo_num + op + 1])

    #             color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #             gray_img = color_img[:, :, 0]

    #             if op == self.LR_num - 2:
    #                 low_bi = cv2.resize(gray_img , (int(width // self.mag), int(height // self.mag)), interpolation=cv2.INTER_CUBIC)
    #                 low_bi = cv2.resize(low_bi , (int(width), int(height)), interpolation = cv2.INTER_CUBIC)

    #                 cut_low_bi = low_bi[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
    #                 low_data_list[self.LR_num - 1].append(cut_low_bi)

    #                 high_data_list.append(gray_img[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width])

    #             elif (op + 1) % 3 == 0:
    #                 low_k_15 = cv2.GaussianBlur(gray_img,(3, 3), 1.5, 1.5)
    #                 low_k_15 = low_k_15[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
    #                 low_data_list[op + 1].append(low_k_15)

    #             elif (op + 1) % 3 == 1:
    #                 low_k_18 = cv2.GaussianBlur(gray_img,(3, 3), 1.8, 1.8)
    #                 low_k_18 = low_k_18[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
    #                 low_data_list[op + 1].append(low_k_18)

    #             elif (op + 1) % 3 == 2:
    #                 low_k_21 = cv2.GaussianBlur(gray_img,(3, 3), 2.1, 2.1)
    #                 low_k_21 = low_k_21[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
    #                 low_data_list[op + 1].append(low_k_21)

    #         self.num += 1

    #         if self.num == 3:
    #             plt.imshow(low_k_21, cmap=plt.cm.gray)
    #             plt.show()

    #         if self.num == data_number:
    #             break

    #     return low_data_list, high_data_list

                
