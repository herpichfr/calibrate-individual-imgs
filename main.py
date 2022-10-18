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
import matplotlib.pyplot as plt
import glob


def calibrate_indimgs(tab, imgs):
    """calibrate individual frames used for SPLUS MS coadding"""
    for img in imgs:
        print('calculating for image', img)
        # create path for output image
        dir2save = '/storage/splus/Catalogues/asteroids/indImgsDiag/'
        imgdiagname = dir2save + img + '_diag.png'
        if os.path.isfile(imgdiagname):
            print('Image', img, 'already done! Skipping...')
        elif img == 'fakeimagename':
            print('Filler image name. Skipping...')
        else:
            image_mask = tab[1].data['exposure_id'] == img
            filtername = np.unique(tab[1].data['filter'][image_mask])
            print('filter is', filtername.item())
            if filtername.size > 1:
                raise IOError('duplicated filter for image', img)
            tiles = np.array([i[:-7].replace('_', '-') for i in tab[1].data['obs_id']])
            tile = np.unique(tiles[image_mask]).item()
            print('tile is', tile)

            # get calibrated table of the tile of the image
            mstabname = '/storage/splus/Catalogues/asteroids/idr4_query_' + tile + '.csv'
            if not os.path.isfile(mstabname):
                print('skipping calibration of', img, '. Failed to find tile', tile)
                os.system('echo %s,%s >> /storage/splus/Catalogues/asteroids/failed_images.txt' % (img, tile))
            else:
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
                percs = np.percentile(a, [16, 84, 0.05, 99.95])
                num_obj = a.size
                print('ZP for image', img, 'filter', filtername.item(), 'field', tile, 'is', zp)

                # applying correction to individual catalogue
                print('calibrating image', img, 'for filter', filtername.item(), 'and field', tile)
                newt = Table(data=tab[1].data[image_mask], names=tab[1].data.columns.names)
                newt['mag'] += zp
                imgtabname = '/storage/splus/Catalogues/asteroids/indImgsDiag/' + img + '_phot.csv'
                newt.to_pandas().to_csv(imgtabname, index=False)

                # save diagnostic things
                print('saving table with the params')
                save_tabs4imgs(img, tile, zp, percs, num_obj)

                print('saving diagnostic figure', img + '_diag.png')
                # plot_diagnostics(img, a, zp, percs, num_obj, c1, c2,
                #                 mstab[filtername.item()][sep_constraint & mask], sample=sep_constraint & mask)
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax1.scatter(c1.ra, c1.dec, marker='o', c='c', s=20, label='ind')
                ax1.scatter(c2.ra, c2.dec, marker='.', c='k', s=5, label='MS', alpha=0.5)
                ax1.set_xlabel('RA')
                ax1.set_ylabel('Dec')
                ax1.set_title(img + ' f: ' + filtername.item(), fontsize=10)
                ax1.legend(loc='upper left', fontsize=8)

                ax2 = fig.add_subplot(222)
                ax2.scatter(c1.ra[idx][sep_constraint & mask], c1.dec[idx][sep_constraint & mask],
                            marker='o', c='c', s=20, label='ind')
                ax2.scatter(c2.ra[sep_constraint & mask], c2.dec[sep_constraint & mask],
                            marker='.', c='k', s=5, label='MS', alpha=0.5)
                ax2.set_xlabel('RA')
                ax2.set_title('N = %i' % int(num_obj), fontsize=10)
                ax2.legend(loc='upper left', fontsize=8)

                ax3 = fig.add_subplot(223)
                y, x, _ = ax3.hist(a, bins=100, color='r', range=(percs[2], percs[3]))
                ax3.set_xlabel('mag_auto_MS - mag_auto_ind')
                ax3.plot([zp, zp], [-0.01, max(y) + 1], '--', c='k', lw=1., label='median: %.2f' % zp)
                ax3.plot([percs[0], percs[0]], [-0.01, max(y) + 1], '-.', c='k', lw=1.5, label='p16: %.2f' % percs[0])
                ax3.plot([percs[1], percs[1]], [-0.01, max(y) + 1], '-.', c='k', lw=1.5, label='p84: %.2f' % percs[1])
                ax3.legend(loc='upper left', fontsize=8)

                ax4 = fig.add_subplot(224)
                ax4.scatter(mstab[filtername.item()][sep_constraint & mask], a, marker='.', c='b')
                ax4.plot([min(mstab[filtername.item()][sep_constraint & mask]),
                          max(mstab[filtername.item()][sep_constraint & mask])],
                         [zp, zp], '-', c='k', lw=1.5)
                ax4.set_xlabel('mag_auto_MS')
                ax4.set_ylabel('MS - ind')

                plt.tight_layout()

                plt.savefig(imgdiagname, format='png', dpi=120)
                plt.close()

    return


def save_tabs4imgs(img, tile, zp, percs, num_obj):
    """save a table for each image containing the resulting parameters of calibration"""
    # data = np.array([img, tile, zp, percs[0], percs[1], num_obj])
    # cols = []
    df = pd.DataFrame(
        {'ImgName': [img], 'tile': [tile], 'ZP': [zp], 'p16': [percs[0]], 'p84': percs[1], 'N': [num_obj]})
    dir2save = '/storage/splus/Catalogues/asteroids/indImgsDiag/'
    if not os.path.isdir(dir2save):
        os.mkdir(dir2save)
    imgtabname = dir2save + img + '_zp.csv'
    df.to_csv(imgtabname, index=False)

    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # create directory that will keep the byproducts of calibration
    dir2save = '/storage/splus/Catalogues/asteroids/indImgsDiag/'
    if not os.path.isdir(dir2save):
        os.mkdir(dir2save)

    # read table with the photometry of individual images
    list_table = glob.glob('/storage/splus/Catalogues/asteroids/allsplusdetections-*.csv.fits')
    for tabname in list_table:
        print('reading table', tabname)
        tab = fits.open(tabname)

        # initialize the number of processes to run in parallel
        num_procs = 8
        images = np.unique(tab[1].data['exposure_id'])
        b = list(images)
        num_images = np.unique(tab[1].data['exposure_id']).size
        if num_images % num_procs > 0:
            print('reprojecting', num_images, 'images')
            increase_to = int(num_images / num_procs) + 1
            i = 0
            while i < (increase_to - num_images):
                b.append('fakeimagename')
                i += 1
            else:
                print(num_images, 'already fulfill the conditions')

        images = np.array(b).reshape((num_procs, int(np.array(b).size / num_procs)))
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
                proc_alive = False

        print('Done!')
