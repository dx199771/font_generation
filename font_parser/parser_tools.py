import os, re, cv2
import zipfile
from PIL import Image,ImageDraw,ImageFont
import numpy as np

def list_allfile(path,all_files=[],all_font_files=[],num=3200):
    """

    :param path:
    :param all_files:
    :param all_font_files:
    :param num:
    :return:
    """
    if os.path.exists(path):
        files=os.listdir(path)
    else:
        print('this path not exist')
    for file in files:
        if os.path.isdir(os.path.join(path,file)):
            list_allfile(os.path.join(path,file),all_files)
        else:
            all_files.append(os.path.join(path,file))
    for file in all_files:
        if re.match('.+\.ttf$',file):
            all_font_files.append(file)
    return all_font_files[:num]

def unzip_file(zip_src, dst_dir):
    """
    Function to unzip a target file from zip_src.
    :param zip_src: zipped file directory
    :param dst_dir: output unzipped file directory
    :return:
    """
    if zipfile.is_zipfile(zip_src):
        fz = zipfile.ZipFile(zip_src,'r')
        for file in fz.namelist():
            fz.extract(file,dst_dir)
    else:
        print('this is not a zip file')

def create_folder(newpath):
    """
    Function to create folder by newpath which not already exist

    :param newpath: new folder that will be created
    :return:
    """
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def read_img(all_ttf_file, opt_char, crop_path):
    """
    Function to read all the font .ttf files and convert to image datatype,
    Crop and process the font image to proper size and save to training data folder.

    :param all_ttf_file: all the .ttf files from parser folder
    :param opt_char: characters in the font that will be generated
    :param crop_path: directory that processed font image will be saved (training data dir)
    :return:
    """
    # capital characters will be trained
    capital_alphbet = opt_char.upper()

    for j in range(len(capital_alphbet)):
        # target output font image size
        W, H = (300, 300)
        for i in range(len(all_ttf_file)):
            print("Character: {} is being processed, total {}/{} characters.".format(capital_alphbet[j],j,len(capital_alphbet)))

            first_slice = all_ttf_file[i].split('/')[-1]
            second_slice = first_slice.split('\\')[-1]
            print(".ttf file: {} is being processed, total {}/{} files.".format(second_slice,i,len(all_ttf_file)))

            image = Image.new('RGB', (W, H))  # create new canvas to display font images
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(all_ttf_file[i], 100)
            try:
                draw.text(((W - 150) / 2, (H - 150) / 2), capital_alphbet[j], font=font)
            except:
                print("draw text error")

            image = np.array(image)[:, :, ::-1].copy()
            imgray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(imgray, 1, 255, cv2.THRESH_BINARY)

            # find uncrop image contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lst_contours = []
            for cnt in contours:
                ctr = cv2.boundingRect(cnt)
                lst_contours.append(ctr)

            # save cropped image into training folder
            if len(lst_contours) != 0:
                x, y, w, h = sorted(lst_contours, key=lambda coef: coef[3])[-1]

                crop = imgray[y - 10:y + h + 10, x - 10:x + w + 10]
                crop_img = crop_path + str(capital_alphbet[j])
                create_folder(crop_img)
                cv2.imwrite(crop_img + '/' + str(second_slice[:-4]) + "_crop_" + str(j) + ".jpg", crop)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
    print("Total {} characters, {} images successfully processed and saved.".format(len(capital_alphbet),len(all_ttf_file)))

