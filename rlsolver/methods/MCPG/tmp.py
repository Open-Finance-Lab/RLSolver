
import numpy as np
import os
import sys
import copy
cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cur_path, 'data')
sys.path.append(os.path.dirname(data_path))


def write2(d):
    d_shape = d.shape
    if len(d_shape) == 1:
        s2 = str(d)
        # s3 = ["{:.6f}".format(d[i]) for i in range(len(d))]
        s3 = [str(d[i]) + ", " for i in range(len(d))]
        # s3 = str(s3)
        file.writelines(s3)
        file.write("\n")
        all_ = all_int(d)
        # file.write(s2)
        # if all_:
        #     file.write(s2)
        # else:
        #     file.write(s3)
    elif len(d_shape) == 2:
        lines2 = [str(d[i]) for i in range(d_shape[0])]
        lines3 = []
        for i in range(d_shape[0]):
            # line = ["{:.6f}".format(d[i][j]) for j in range(d_shape[1])]
            line = [str(d[i][j]) + ", " for j in range(d_shape[1])]
            lines3.append(line)
            file.writelines(line)
            file.write("\n")
        # all_ = all_int(d)
        # for line in lines2:
        #     file.write(line)
        #     file.write("\n")
    else:
        pass

dir = data_path + "/mimo2"
dir2 = os.listdir(dir)

def all_int(d):
    d_shape = d.shape
    if len(d_shape) == 1:
        res = True
        for i in d:
            if i % 1 != 0:
                res = False
                break
    elif len(d_shape) == 2:
        res = True
        for i in range(len(d)):
            if not res:
                break
            for j in range(len(d[0])):
                if d[i][j] % 1 != 0:
                    res = False
                    break
    else:
        raise ValueError
    return res

for dir3 in dir2:
    dir3b = dir + "/" + dir3
    dir4 = os.listdir(dir3b)
    dir4b = [dir3b + "/" + dir4[i] for i in range(len(dir4))]
    # dir5 = os.listdir(dir4b)
    for dir5 in dir4b:
        if not os.path.isdir(dir5):
            continue
        dir6 = os.listdir(dir5)
        dir6b = [dir5 + "/" + dir6[i] for i in range(len(dir6))]
        H = None
        v = None
        X = None
        SNR = None
        K = None
        f_txt_new1 = copy.deepcopy(dir5) + ".txt"
        f_txt_new1 = f_txt_new1.replace("mimo2", "mimo3")
        print("f_txt_new1: ", f_txt_new1)
        ind1 = 0
        ind2 = 0
        for i in range(len(f_txt_new1)):
            if f_txt_new1[i] == "/":
                ind1 = i
            elif f_txt_new1[i] == "_":
                ind2 = i
        # f_txt_new2 = copy.deepcopy(f_txt_new1)
        # print("f_txt_new2: ", f_txt_new2)
        # f_txt_new = f_txt_new2.replace(f_txt_new2[ind1: ind2 + 1], "_ID")
        f_txt_new = ""
        f_txt_new += f_txt_new1[: ind1]
        f_txt_new += "_ID"
        f_txt_new += f_txt_new1[ind2 + 1: ]
        print("f_txt_new: ", f_txt_new)
        for f_npy in dir6b:

            if ".txt" in f_npy:
                continue
            d = np.load(f_npy)
            print("d: ", d)
            if f_npy[-5] == "X":
                X = d
            elif f_npy[-5] == "v":
                v = d
            elif f_npy[-5] == "H":
                H = d
            elif f_npy[-5] == "R":
                SNR = d
            elif f_npy[-5] == "K":
                K = d
            else:
                pass
            f_txt = f_npy.replace("npy", "txt")
            f_txt = f_txt.replace("mimo2", "mimo3")
            # 获取文件所在目录的绝对路径
            current_dir = os.path.abspath(os.path.dirname(f_txt))

            # # 获取上级目录的路径
            # parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
            if not os.path.exists(current_dir):
                os.makedirs(current_dir)
            # np.savetxt(f2, d)
            d_shape = d.shape
            print("shape: ", d_shape)
            if len(d_shape) == 0:
                continue
            with open(f_txt, 'w', encoding='utf-8') as file:
                pass
                # write2(d)
                    # if all_:
                    #     for line in lines2:
                    #         file.write(line)
                    #         file.write("\n")
                    # else:
                    #     for line in lines3:
                    #         file.write(line)
                    #         file.write("\n")
                    # file.writelines(lines2)
                    # for i in range(d_shape[0]):
                    #     s2 = str(d[i])
                    #     file.write(s2)
                    # raise ValueError
        print("f_txt_new: ", f_txt_new)
        with open(f_txt_new, 'w', encoding='utf-8') as file:
            file.write("SNR\n")
            file.write(str(SNR))
            file.write("\n\n")
            file.write("K\n")
            file.write(str(K))
            file.write("\n\n")
            file.write("H\n")
            write2(H)
            file.write("\n\n")
            file.write("v\n")
            write2(v)
            file.write("\n\n")
            file.write("X\n")
            write2(X)
            file.write("\n\n")

