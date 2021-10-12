from PIL import Image
import pytesseract
print(pytesseract.get_languages(config=''))
result = pytesseract.image_to_string(Image.open('data/input/test.png'), lang='chi_sim')
# print(result)




import os
import paddle
paddle.fluid.install_check.run_check()

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

img_path = 'data/input/test.jpg'

def customer_ocr(img_path):
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=False)
    result = ocr.ocr(img_path, cls=True)
    out_txt = 'data/output/' + os.path.basename(img_path).rsplit('.',1)[0] + '.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        for line in result:
            # print(line[1])
            f.write(line[1][0]+'\n')
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='data/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('data/output/' + os.path.basename(img_path).rsplit('.',1)[0] + '.jpg', dpi=(300.0,300.0))

customer_ocr(img_path)


def get_img_path(dir_path):
    img_l = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            if name.endswith('.png'):
                fn = os.path.join(root,name)
                img_l.append(fn)
    return img_l
   



if __name__=="__mian__":
    dir_path = 'data/input/other_1983.01-03（延安工务段）'
    img_l = get_img_path(dir_path)
    for img_path in img_l:
        outImgPath = 'data/output/other_1983.01-03（延安工务段）/' + os.path.basename(img_path).rsplit('.',1)[0] + '.jpg'
        out_txt = 'data/output/other_1983.01-03（延安工务段）/' + os.path.basename(img_path).rsplit('.',1)[0] + '.txt'
        customer_ocr(img_path, outImgPath, out_txt)
