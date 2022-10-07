# herpich 2022-10-05.
import os.path

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
import multiprocessing
import time
import os

def calibrate_indimgs(tab, imgs):
    """calibrate individual frames used for SPLUS MS coadding"""
    for img in imgs:
        print('calculating for image', img)
        image_mask = tab[1].data['exposure_id'] == img
        filtername = np.unique(tab[1].data['filter'][image_mask])
        print('filter is', filtername.item())
        if filtername.size > 1:
            raise IOError('duplicated filter for image', img)
        tiles = np.array([i[:-7].replace('_', '-') for i in tab[1].data['obs_id']])
        tile = np.unique(tiles[image_mask]).item()
        print('tile is', tile)

        # get calibrated table of the tile of the image
        mstabname = '/storage/Documents/pos-doc/t80s/asteroids/idr4_query_' + tile + '.csv'
        print('reading MS tab', mstabname)
        mstab = pd.read_csv(mstabname)
        mstab.rename(columns={'u_auto': 'U', 'J0378_auto': 'F378', 'J0395_auto': 'F395',
                              'J0410_auto': 'F410', 'J0430_auto': 'F430', 'g_auto': 'G',
                              'J0515_auto': 'F515', 'r_auto': 'R', 'J0660_auto': 'F660',
                              'i_auto': 'I', 'J0861_auto': 'F861', 'z_auto': 'Z'},
                     inplace=True)

        # create the sky coordinates for both tables and match them
        print('preparing coordinates for matching')
        c1 = SkyCoord(ra=tab[1].data['ra'][image_mask], dec=tab[1].data['dec'][image_mask], unit=(u.deg, u.deg))
        c2 = SkyCoord(ra=mstab['RA'], dec=mstab['DEC'], unit=(u.deg, u.deg))
        print('matching image', img, 'with table', mstabname)
        idx, d2d, d3d = c2.match_to_catalog_sky(c1)
        max_sep = 1.0 * u.arcsec
        sep_constraint = d2d < max_sep

        # finding ZP of image
        mask = (mstab[filtername.item()] > 14) & (mstab[filtername.item()] < 18)
        a = mstab[filtername.item()][sep_constraint & mask] - tab[1].data['mag'][image_mask][idx][sep_constraint & mask]
        zp = np.median(a)
        percs = np.percentile(a, [16, 84])
        num_obj = a.size
        print('ZP for image', img, 'filter', filtername.item(), 'field', tile, 'is', zp)

        # applying correction to individual catalogue
        print('calibrating image', img, 'for filter', filtername.item(), 'and field', tile)
        newt = Table(data=tab[1].data[image_mask], names=tab[1].data.columns.names)
        newt['mag'] += zp
        imgtabname = '/home/herpich/Documents/pos-doc/t80s/asteroids/indImgsDiag/' + img + '_phot.csv'
        newt.to_pandas().to_csv(imgtabname, index=False)

        # save diagnostic things
        save_tabs4imgs(img, tile, zp, percs, num_obj)

    return

def save_tabs4imgs(img, tile, zp, percs, num_obj):
    """save a table for each image containing the resulting parameters of calibration"""
    # data = np.array([img, tile, zp, percs[0], percs[1], num_obj])
    # cols = []
    df = pd.DataFrame({'ImgName': [img], 'tile': [tile], 'ZP': [zp], 'p16': [percs[0]], 'p84': percs[1], 'N': [num_obj]})
    dir2save = '/home/herpich/Documents/pos-doc/t80s/asteroids/indImgsDiag/'
    if not os.path.isdir(dir2save):
        os.mkdir(dir2save)
    imgtabname = dir2save + img + '_zp.csv'
    df.to_csv(imgtabname, index=False)

    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # create directory that will keep the byproducts of calibration
    dir2save = '/home/herpich/Documents/pos-doc/t80s/asteroids/indImgsDiag/'
    if not os.path.isdir(dir2save):
        os.mkdir(dir2save)

    # read table with the photometry of individual images
    tabname = '/storage/Documents/pos-doc/t80s/asteroids/allsplusdetections-0-1080.csv.fits'
    print('reading table', tabname)
    tab = fits.open(tabname)

    # initialize the number of processes to run in parallel
    num_procs = 4
    images = np.unique(tab[1].data['exposure_id']).reshape((num_procs, int(1080/num_procs)))
    print('calculating for a total of', images.size, 'images')
    jobs = []
    print('creating', num_procs, 'jobs...')
    for imgs in images:
        process = multiprocessing.Process(target=calibrate_indimgs, args=(tab, imgs))
        jobs.append(process)

    # start jobs
    print('starting', num_procs, 'jobs!')
    for j in jobs:
        j.start()

    # check if any of the jobs initialized previously still alive
    # save resulting table after all are finished
    proc_alive = True
    while proc_alive:
        if any(proces.is_alive() for proces in jobs):
            proc_alive = True
            time.sleep(1)
        else:
            print('All jobs finished')
            finaltabname = '/storage/Documents/pos-doc/t80s/asteroids/allsplusdetections-0-1080-calib.fits'
            print('saving table', finaltabname)
            tab.writeto(finaltabname, overwrite=True)
            proc_alive = False

    print('Done!')