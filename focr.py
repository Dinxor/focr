import os
import time
import sys
import pyperclip
import pytesseract
import configparser
from PIL import Image
import numpy as np
import cv2 as cv

def recogn(im, cn = '--psm 8'):
    tn = pytesseract.image_to_string(im, config=cn)
    symbs = ''
    if len(tn) > 0:
        for s in tn:
            if s.isalpha() or s.isdigit():
                symbs = symbs + s.lower()
    return symbs

if __name__ == '__main__':
    if len (sys.argv) > 1 :
        in_file = sys.argv[1]
    else:
        if os.path.exists('turbobit_net_v50_GDL_03.png'):
            in_file = 'turbobit_net_v50_GDL_03.png'
        else:
            sys.exit()
    if len (sys.argv) > 2 :
        out_name = sys.argv[2]
    else:
        out_name = ''

    path = "focr.ini"
    changes = []
    koefs = []
    if os.path.exists(path):
        config = configparser.ConfigParser()
        config.read(path)
        tdir = config.get("Tesseract", "dir")
        opt = config.get("Tesseract", "opt")
        mask = int(config.get("Image", "mask"), 2)
        save_img = int(config.get("Image", "save_img"))
        save_parts = int(config.get("Image", "save_parts"))
        if save_img + save_parts > 0:
            save_dir = config.get("Image", "save_dir")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            basename = str(int(time.time()))
        start = int(config.get("Angle", "start"))
        finish = int(config.get("Angle", "finish"))
        step = int(config.get("Angle", "step"))
        for option in config.options("Changes"):
            changes.append([option, config.get("Changes", option)])
        for option in config.options("Koeff"):
            koefs.append([option, float(config.get("Koeff", option))])
        limit = int(config.get("Break", "limit"))
        mode = config.get("Output", "mode")
        if mode == 'file' and out_name == '':
            out_name = config.get("Output", "filename")
        parts = []
        for option in config.options("Parts"):
            parts.append([int(option), int(config.get("Parts", option))])
    else:
        tdir = '.\\Tesseract\\'
        opt = '--psm 10 turbobit'
        mask = 192
        save_img, save_parts = 0, 0
        start, finish, step = -30, 31, 5
        limit = 0
        mode = 'file'
        if out_name == '':
            out_name = 'rezocr.txt'
        parts = [[0,43], [35,75], [70,107], [102, 148]]

    tess_cmd = tdir + 'tesseract.exe'
    tessdata = tdir.replace('\\', '/') + 'tessdata'
    pytesseract.pytesseract.tesseract_cmd = tess_cmd

    rez = ''
    img = cv.imread(in_file)
    dim = (300,100)
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    blur = cv.blur(resized,(4,4))
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    for x0, x1 in parts:
        img0 = gray[0:100, 2*x0:2*x1]
        thresh = cv.inRange( img0, 170, 246 )
        contours0, hierarchy = cv.findContours( thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img1 = img0.copy()
        img1.fill(255)
        cv.fillPoly(img1, contours0, 0)
        mask = cv.bitwise_not(img1)
        # cv.imshow('contours', mask)
        # cv.waitKey()
        # cv.destroyAllWindows()
        img2 = Image.fromarray(mask)
        symbs = {}
        for angle in range(start, finish, step):
            tn = recogn(img2.rotate(angle), '--tessdata-dir ' + tessdata + ' ' + opt)
#            print(tn)
            if len(tn) > 0:
                if len(tn) == 2:
                    for ch in changes:
                        if tn == ch[0]:
                            tn = ch[1]
                            break
                for k in tn:
                    if symbs.get(k) == None:
                        symbs.update({k:1})
                    else:
                        symbs.update({k:symbs.get(k)+1})
                if limit > 0 and max(symbs.values()) > limit:
                    break
        if len(symbs):
            for k in koefs:
                sk = symbs.get(k[0], 0)
                if sk > 1:
                    symbs.update({k[0]:sk*k[1]})
            symb = max(symbs.items(), key = lambda x: x[1])[0]
            rez += symb
        if save_parts > 0:
            img2.save(save_dir + basename + '_' + symb + '.bmp')
    if mode == 'file':
        fout = open(out_name, 'w')
        fout.write(rez)
        fout.close()
    elif mode == 'clipboard':
        pyperclip.copy(rez)
    elif mode == 'print':
        print(rez)
    elif mode == 'stdout':
        sys.stdout.write(rez)
    if save_img > 0:
#        sys.stdout = open(os.devnull, 'w')
        os.system('copy ' + str(in_file) + ' ' + save_dir + basename + '_' + rez + '.png > nul')
