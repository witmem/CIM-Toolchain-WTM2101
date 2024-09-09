import sys
import os
import re
import getopt
import glob
import json
import random
import colorsys
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

WIDTH = 1024
HEIGHT = 2048
PADDING = 200

RF_Y = PADDING + HEIGHT + PADDING

XMARKS = {}
YMARKS = {}
GRIDS = []

COLOR_MAPS = [
    '#FFB900',
    '#FF8C00',
    '#F7630C',
    '#CA5010',
    '#DA3B01',
    '#EF6950',
    '#D13438',
    '#FF4343',

    '#E74856',
    '#E81123',
    '#EA005E',
    '#C30052',
    '#E3008C',
    '#BF0077',
    '#C239B3',
    '#9A0089',

    '#0078D7',
    '#0063B1',
    '#8E8CD8',
    '#6B69D6',
    '#8764B8',
    '#744DA9',
    '#B146C2',
    '#881798',

    '#0099BC',
    '#2D7D9A',
    '#00B7C3',
    '#038387',
    '#00B294',
    '#018574',
    '#00CC6A',
    '#10893E',
]


def get_block_text_color(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    if l < 0.8:
        r, g, b = colorsys.hls_to_rgb(h, l+0.2, s)
    elif s < 0.8:
        r, g, b = colorsys.hls_to_rgb(h, l, s+0.2)
    else:
        r, g, b = colorsys.hls_to_rgb(h, l/2, s/2)
    r = int(r*255)
    g = int(g*255)
    b = int(b*255)
    return (r, g, b)


def get_front_text_color(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    h = 0.99 - h
    if l > 0.6:
        l = 0.0
    elif l < 0.4:
        l = 0.99
    elif s > 0.6:
        s = 0.0
    else:
        s = 0.99
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r = int(r*255)
    g = int(g*255)
    b = int(b*255)
    return (r, g, b)


def create_image(n_configs, max_width):
    size = (max_width + PADDING*2, PADDING + HEIGHT +
            PADDING + 20*n_configs + PADDING//4)
    img = Image.new('RGB', size, color='#EEEEEE')

    #font = ImageFont.truetype('tahoma.ttf', 20)
    font_normal = ImageFont.truetype('Gargi.ttf', 20)
    font_large = ImageFont.truetype('Gargi.ttf', 32)

    # -------------------------------------------------------
    #
    #   array
    #

    # draw zero point and maximal point
    draw = ImageDraw.Draw(img)

    loc = (PADDING, PADDING, PADDING + max_width, PADDING + HEIGHT)
    draw.rectangle(loc, fill='white', outline='black')

    loc = (PADDING - 60, PADDING - 30)
    draw.text(loc, '(0,0)', font=font_normal, fill='black')

    loc = (PADDING + max_width + 10, PADDING + HEIGHT + 10)
    draw.text(loc, '(%d,%d)' % (max_width, HEIGHT),
              font=font_normal, fill='black')

    loc = (PADDING + max_width//2, PADDING - 80)
    draw.text(loc, 'array', font=font_large, fill='black')

    # draw x-title
    loc = (PADDING + max_width//2, PADDING + HEIGHT + PADDING//4)
    draw.text(loc, 'X', font=font_large, fill='black')

    # draw y-title
    loc = (PADDING//4, PADDING + HEIGHT//2)
    draw.text(loc, 'Y', font=font_large, fill='black')

    return (img, font_normal, font_large)


def render(img, font_normal, font_large, config, configs, i, max_width):
    global XMARKS
    global YMARKS
    global GRIDS
    global WIDTH
    global HEIGHT
    global PADDING

    if (configs.size == 7):
        config = configs

    index = i

    draw = ImageDraw.Draw(img)

    # Ys, Ye: DAC send to Array
    Ys = config[2]
    Ye = config[3]
    height = Ye - Ys + 1

    # Xs, Xe: Array output to ADC
    Xs = config[0]
    Xe = config[1]
    width = Xe - Xs + 1

    # image
    #
    #   0 ---> x
    #   |
    #   y
    #
    if index < len(COLOR_MAPS):
        select_color = (5*index) % len(COLOR_MAPS)
        color_bg = COLOR_MAPS[select_color]
    else:
        color_bg = '#%06X' % random.randint(0, 16777216-1)

    r = int(color_bg[1:3], 16)
    g = int(color_bg[3:5], 16)
    b = int(color_bg[5:7], 16)

    colors = get_block_text_color(r, g, b)
    color_blk_text = '#%02X%02X%02X' % colors

    colors = get_front_text_color(r, g, b)
    color_text = '#%02X%02X%02X' % colors

    # ------------------------------------------------------------------
    #
    #   block rectangle
    #
    loc = (Xs + PADDING, Ys + PADDING, Xe + PADDING, Ye + PADDING)
    draw.rectangle(loc, fill=color_bg)
    loc = (Xs + PADDING, 1807 + config[4]*15 + PADDING,
           Xe + PADDING, 1807 + (config[5] + 1)*15 + PADDING)
    draw.rectangle(loc, fill=color_bg)

    # ------------------------------------------------------------------
    #
    #   block X-end edge text
    #
    if Xe >= 1000:
        loc = (Xe+PADDING-45, Ys+PADDING)
    elif Xe >= 100:
        loc = (Xe+PADDING-35, Ys+PADDING)
    elif Xe >= 10:
        loc = (Xe+PADDING-25, Ys+PADDING)
    else:
        loc = (Xe+PADDING-15, Ys+PADDING)
    draw.text(loc, '%d' % (Xe), font=font_normal, fill=color_text)

    loc = (Xs+PADDING+width/3, Ys+PADDING+height/2-10)
    draw.text(loc, 'L%d' % index, font=font_large, fill=color_blk_text)

    # ------------------------------------------------------------------
    #
    #   Horizontal and Vertical marks
    #
    xmark = '%d' % Xs
    if xmark not in XMARKS:
        loc = (Xs + PADDING, PADDING - 30)
        draw.text(loc, xmark, font=font_normal, fill='black')
        loc = (Xs + PADDING, HEIGHT + PADDING + 10)
        draw.text(loc, xmark, font=font_normal, fill='black')
        loc = ((Xs + PADDING, PADDING - 30),
               (Xs + PADDING, HEIGHT + PADDING + 30))
        GRIDS.append((loc, '#333333'))
        XMARKS[xmark] = True

    ymark = '%d' % Ys
    y1 = '%d' % config[4]
    y2 = '%d' % config[5]
    if ymark not in YMARKS:
        loc = (PADDING - 50, 1807 + config[4]*15 + PADDING)
        draw.text(loc, y1, font=font_normal, fill='black')
        loc = ((PADDING - 50, 1807 + config[4]*15 + PADDING),
               (max_width + PADDING + 30, 1807 + config[4]*15 + PADDING))
        GRIDS.append((loc, '#333333'))
        loc = (PADDING - 50, Ys + PADDING)
        draw.text(loc, ymark, font=font_normal, fill='black')
        loc = ((PADDING - 50, Ys + PADDING),
               (max_width + PADDING + 30, Ys + PADDING))
        GRIDS.append((loc, '#333333'))
        YMARKS[ymark] = True

    ymark = '%d' % Ye
    if ymark not in YMARKS:
        loc = (PADDING - 50, 1807 + config[5]*15 + PADDING)
        draw.text(loc, y2, font=font_normal, fill='black')
        loc = ((PADDING - 50, 1807 + config[5]*15 + PADDING),
               (max_width + PADDING + 30, 1807 + config[5]*15 + PADDING))
        GRIDS.append((loc, '#333333'))
        loc = (PADDING - 50, Ye + PADDING - 18)
        draw.text(loc, ymark, font=font_normal, fill='black')
        loc = ((PADDING - 50, Ye+1 + PADDING),
               (max_width + PADDING + 30, Ye+1 + PADDING))
        GRIDS.append((loc, '#333333'))
        YMARKS[ymark] = True


def draw_array_space(layer_txt, out_path):
    i = 0
    print(layer_txt)
    if os.path.exists(layer_txt) and os.path.getsize(layer_txt) > 0:
        configs = np.loadtxt(layer_txt, dtype=np.int16, delimiter=',')
        
        if (configs.size == 0):
            return
        if (configs.size == 7):
            max_width = configs[3]
        if (configs.size != 7):
            max_width = configs.max(axis=0)[1]
        img, font_normal, font_large = create_image(10, max_width)
        for cfg in configs:
            render(img, font_normal, font_large, cfg,
                configs, i, max_width)
            i = i + 1

        # draw marks
        draw = ImageDraw.Draw(img)
        loc = (200, 1993, max_width + 200, 2005)
        draw.rectangle(loc, fill="#ffff33")
        loc = (PADDING + max_width/2, 1785 + PADDING)
        draw.text(loc, "bias", font=font_normal, fill='black')
        for loc, color in GRIDS:
            draw.line(loc, fill=color)

        # dump image
        fnout = out_path + '/array_space.png'
        print('create', fnout)
        img.save(fnout)


if __name__ == '__main__':
    i = 0
    if os.path.exists("./layers.txt") and os.path.getsize("./layers.txt") > 0:
        configs = np.loadtxt("./layers.txt", dtype=np.int16, delimiter=',')
        if (configs.ndim == 1):
            max_width = configs[3]
        else:
            max_width = configs.max(axis=0)[1]
        img, font_normal, font_large = create_image(10, max_width)
        for cfg in configs:
            render(img, font_normal, font_large, cfg,
                configs, i, max_width)
            i = i + 1

        # draw marks
        draw = ImageDraw.Draw(img)
        loc = (200, 1993, max_width + 200, 2005)
        draw.rectangle(loc, fill="#ffff33")
        loc = (PADDING + 512, 1785 + PADDING)
        draw.text(loc, "bias", font=font_normal, fill='black')
        for loc, color in GRIDS:
            draw.line(loc, fill=color)

        # dump image
        fnout = './array_space.png'
        print('create', fnout)
        img.save(fnout)
