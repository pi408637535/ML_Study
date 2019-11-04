import random
import pandas as pd
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter


path = "/Users/piguanghua/Downloads"

def numRandom():
    return random.randint(0,9)

def numRandom1():
    return random.randint(1,9)


#返回随机字母
def charRandom():
    return chr((random.randint(65,90)))

#随机长生颜色2
def colorRandom2():
    return (random.randint(32,127),random.randint(32,127),random.randint(32,127))


def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))  # 随机颜色


def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))  # 随机颜色

# In[17]:
def gen_veri(i):
    width = 180 * 4
    height = 180
    image = Image.new('RGB', (width, height), (255, 255, 255));
    # 创建font对象
    font = ImageFont.truetype('Arial.ttf', 18);

    # 创建draw对象
    draw = ImageDraw.Draw(image)
    # 填充每一个颜色
    #     for x in range(width):
    #         for y in range(height):
    #             draw.point((x,y), fill=colorRandom1())

    texts = ''
    # 输出文字
    for t in range(4):
        if t== 0:
            c = numRandom1()
        else:
            c = numRandom()

        c = str(c)
        texts = texts + c
        draw.text((180 * t + 60+10, 60+10), c, font=font, fill=colorRandom2())

    image.save(path+'/GenPics/' + str(i) + '.jpg', 'jpeg')
    del draw
    del image
    del font



    return [i, texts]


# # 返还图片序号及验证码字符串

# In[18]:

lables = []
for i in range(0, 6000):
    lab = gen_veri(i)

    lables.append(lab)

# In[19]:
def str_map(str):
    if len(str[1]) < 4:
        print("aaa")
    return {str[0]:str[1]}

import csv

csvfile = open(path+'/GenPics/lables.csv', 'wb')
#writer = csv.writer(csvfile)
#writer.writerows(lables)
data_map = {}
for ele in map(str_map, lables):
    data_map.update(ele)
df = pd.Series(data_map)
#df = df.astype(str)
def convert_currency(value):
    return np.str(value)
df.apply(convert_currency)
df.to_csv(path+'/GenPics/lables.csv', mode='a+',header=False)