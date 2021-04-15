import re, os, cv2, argparse, requests
from bs4 import BeautifulSoup as bs
from urllib.request import urlretrieve
from PIL import Image, ImageFont, ImageDraw
import parser_tools
""" google font parser """
"""
    
"""
parser = argparse.ArgumentParser()
parser.add_argument("--font_srccode", type=str, default="./google_font_src.txt", help="source code of google font")
parser.add_argument("--opt_zip_dir", type=str, default="../data/Glyph_parser_data/zip/", help="downloaded zip font file dir")
parser.add_argument("--opt_font_dir", type=str, default="../data/Glyph_parser_data/unzip/", help="downloaded after zip font file dir")
parser.add_argument("--opt_char", type=str, default="abcdefghijklmnopqrstuvwxyz", help="characters that will be trained")
opt = parser.parse_args()



def google_font_parser(font_source,zip_opt,font_file_opt):
    """

    """
    with open(font_source, 'r') as f:
        lines = f.read().replace('\n','')
    soup = bs(lines, 'html.parser')

    # get all font family <style> block content
    font_family = [data for data in soup.select('style') if 'font-family'in str(data)][1:]
    for i in range(len(font_family)):
        split_head = re.search(r'gf-font-style="',str(font_family[i])).span()[1]
        split_tail = re.search(r'script=',str(font_family[i])).span()[0]
        font_familys = str(font_family[i])[split_head+1:split_tail]
        font_familys = font_familys.strip().replace(" ","%20")
        # get the download url
        url = "http://fonts.google.com/download?family="+font_familys

        # download font zip file
        urlretrieve(url,zip_opt+font_familys+".zip")

        print(font_familys.strip().replace('',''),":",url)
        if not os.path.exists(zip_opt):
            os.makedirs(zip_opt)
        if not os.path.exists(font_file_opt):
            os.makedirs(font_file_opt)

        font_file_opt_ = font_file_opt+font_familys+"zip"
        zip_src = zip_opt+font_familys+".zip"
        # unzip file
        parser_tools.unzip_file(zip_src,font_file_opt_)

    print("Parser google font files done!")

google_font_parser(opt.font_srccode,opt.opt_zip_dir,opt.opt_font_dir)