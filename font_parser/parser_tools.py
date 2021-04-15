import os, re
import zipfile


def list_allfile(path,all_files=[],all_font_files=[]):
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
    return all_font_files

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src,'r')
        for file in fz.namelist():
            fz.extract(file,dst_dir)
    else:
        print('this is not a zip file')
