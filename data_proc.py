__author__ = 'Albert Qu'

import tifffile
import os
import numpy as np
from skimage import io
from tifffile import TiffWriter


def split_to_series(filename, destF, seq=(0,), ns='online_{}.tiff', numplanes=6):
    """
    Taking filename and split the big tiff into frames
    :param filename: filename of the big tiff, unsplitted
    :param destF: destination folder format, e.g. ROOT/series_{}
    :param seq: sequence of planes needed to record
    :param ns: namescheme of the tiffs saved
    :param numplanes: number of planes
    """
    with tifffile.TiffFile(filename) as ims:
        series_len = len(ims.pages)
        dur = series_len // numplanes
        for i in range(dur):
            for j in range(numplanes):
                if j in seq:
                    dest = destF.format(j)
                    if not os.path.exists(dest):
                        os.mkdir(dest)
                    io.imsave(os.path.join(dest, ns.format(i)),
                              ims.asarray(i * numplanes + j), plugin='tifffile')


def merge_tiffs_lazy(fls, outpath, num=1, tifn='bigtif{}.tif'):
    # Takes in a list of single tiff fls and save them in memmap
    tifn = os.path.join(outpath, tifn)
    imgs = tifffile.TiffSequence(fls)
    totlen = imgs.shape[0]
    #dims = imgs.imread(imgs.files[0]).shape
    chunklen = totlen // num
    fnames = []
    for j in range(num):
        fname = tifn.format(j)
        with TiffWriter(fname, bigtiff=True, imagej=True) as tif:
            fnames.append(fname)
            for i in range(chunklen):
                pointer = i + chunklen * j
                if pointer >= totlen:
                    break
                else:
                    tif.save(imgs.imread(imgs.files[pointer]), compress=6)
    return fnames


def merge_tiffs_obsolete(fls, outpath, num=1, tifn='bigtif{}.tif', memmap=True):
    # Takes in a list of single tiff fls and save them in memmap
    imgs = tifffile.TiffSequence(fls)
    totlen = imgs.shape[0]
    data = imgs.asarray(memmap=memmap, tempdir=outpath)
    chunklen = totlen // num
    fnames = []
    for j in range(num):
        fname = tifn.format(j)
        with TiffWriter(fname, bigtiff=True, imagej=True) as tif:
            fnames.append(fname)
            endpoint = chunklen * (j +1)
            if endpoint >= totlen:
                tif.save(data[chunklen * j:], compress=6)
            else:
                tif.save(data[chunklen * j: endpoint], compress=6)
    return fnames


def merge_tiffs(fls, outpath, num=1, fmm='bigmem', tifn='mergetif{}.tif', order='F', del_mmap=True):
    # Takes in a list of single tiff fls and save them in memmap
    imgs = tifffile.TiffSequence(fls)
    totlen = len(imgs.files)
    dims = imgs.imread(imgs.files[0]).shape
    d3 = dims[2] if len(dims) == 3 else 1
    d1, d2 = dims[0], dims[1]
    fnamemm = os.path.join(outpath, '{}_d1_{}_d2_{}_d3_{}_order_{}_frames_{}_.mmap'.format(fmm, d1, d2, d3, order,
                                                                                          totlen))
    bigmem = np.memmap(fnamemm, mode='w+', dtype=np.float32, shape=(totlen, dims[0], dims[1]), order=order)
    # Fill the mmap
    for i in range(totlen):
        bigmem[i, :, :] = imgs.imread(imgs.files[i])

    bigmem.flush()
    del imgs

    # Read from mmap, save as tifs
    chunklen = totlen // num
    fnames = []
    tifn = os.path.join(outpath, tifn)
    for j in range(num):
        fname = tifn.format(j)
        fnames.append(fname)
        endpoint = chunklen * (j+1)
        if endpoint >= totlen:
            #tifffile.imsave(fname, bigmem[chunklen * j:], imagej=True)
            io.imsave(fname, bigmem[chunklen * j:], plugin='tifffile')
        else:
            #tifffile.imsave(fname, bigmem[chunklen * j: endpoint], imagej=True)
            io.imsave(fname, bigmem[chunklen * j: endpoint], plugin='tifffile')
    # Delete mmap
    try:
        if del_mmap:
            os.remove(fnamemm)
            del bigmem
    except:
        print("Error deleting mmap", fnamemm)
    return fnames


def active_disc_ratio(C, thres=0.8):
    count = 0
    for i in range(C.shape[0]):
        target = len(np.where(C[i] != 0)[0])
        if target / C.shape[1] >= thres:
            count += 1
    return count / C.shape[0]


def demo():
    data_root = "/Users/albertqu/Documents/7.Research/BMI/"
    baseline = os.path.join(data_root, "full_data/181031/baseline_00001.tif")
    destF = os.path.join(data_root, 'basedir{}')
    split_to_series(baseline, destF=destF, ns="base_{}.tiff")
    merged = merge_tiffs([os.path.join(destF.format(0), "base_{}.tiff".format(i)) for i in range(500)], destF)


def merge_tiff_from_folder(input_folder, outpath):
    merge_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if ('.tif' in f)]
    return merge_tiffs(merge_files, outpath, tifn=input_folder.split(os.path.sep)[-1]+'_merge{}.tif')
    
