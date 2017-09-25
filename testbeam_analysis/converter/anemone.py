''' Conversion of raw data with trigger time stamps taken with pyBAR.

    Read out hardware is USBpix3 based MMC3 board.
    6 Mimosa26 planes for pointing resolution are read out triggerless and
    1 FE-I4 plane for time reference is read out with a trigger from the TLU.

    The correlation between the two data streams is done using a trigger time
    stamp feature of the readout system.

    Note:
    -----
    So far only working with:
      - pyBAR_fei4_interpreter development branch
      - pyBAR_mimosa26_interpreter development branch
'''

import os
import shutil
import logging

import numexpr  # BUG: right now with anaconda this needs to be there
import tables as tb
import progressbar
import pyBAR_mimosa26_interpreter.simple_converter_combined as m26_cv

from testbeam_analysis.converter import pybar_fei4_converter as pyfei4_cv


def interpret_anemone_data(raw_data_files):
    ''' Runs all step to analyse telescope raw data recorded with pyBAR.

    Parameters
    ----------
    raw_data_files : string, iterable of strings
        File name of the raw data or several raw data files of one run.
    '''

    # Step 1: Interpret FE-I4 raw data
    # Output: file with _aligned.h5 suffix
    pyfei4_cv.process_dut(raw_data_files,
                          # Data format has trigger time stamp
                          trigger_data_format=2)

    # Step 2a: Combine several files if needed
    # to allow Mimosa26 interpreter to work
    if isinstance(raw_data_files, list):
        raw_data_file = combine_raw_data(raw_data_files)
        fe_event_aligned = raw_data_files[0][:-3] + '_event_aligned.h5'
    else:
        raw_data_file = raw_data_files
        fe_event_aligned = raw_data_files[:-3] + '_event_aligned.h5'

    # Step 2: Interpret MIMOSA26 planes
    # Output: file with _aligned.h5 suffix with plane number
    for plane in range(1, 7):
        m26_cv.m26_converter(fin=raw_data_file,  # Input file
                             fout=raw_data_file[:-3] + \
                             '_frame_aligned_%d.h5' % plane,  # Output file
                             plane=plane)  # Plane number
        # Step 3: Combine FE with Mimosa data
        # Output: file with
        m26_cv.align_event_number(
            fin=raw_data_file[:-3] + '_frame_aligned_%d.h5' % plane,  # Mimosa
            fe_fin=fe_event_aligned,
            fout=raw_data_file[:-3] + '_run_aligned_%d.h5' % plane,
            tr=True,  # Switch column / row (transpose)
            frame=False)  # Add frame info (not working?)


def combine_raw_data(raw_data_files, chunksize=10000000):
    file_combined = raw_data_files[0][:-3] + '_combined.h5'
    # Use first tmp file as result file
    shutil.move(raw_data_files[0], file_combined)
    with tb.open_file(raw_data_files[0][:-3] + '_combined.h5', 'r+') as out_f:
        combined_data = in_f.root.raw_data
        status = 0
        progress_bar = progressbar.ProgressBar(
                            widgets=['',
                                     progressbar.Percentage(),
                                     ' ',
                                     progressbar.Bar(marker='*',
                                                     left='|',
                                                     right='|'),
                                     ' ',
                                     progressbar.AdaptiveETA()],
                            maxval=in_f.root.raw_data.shape[0] * (len(raw_data_files) - 1) + chunksize,
                            term_width=80)
        progress_bar.start()
        for raw_data_file in raw_data_files[1:]:
            with tb.open_file(raw_data_file) as in_f:
                for chunk in range(0, in_f.root.raw_data.shape[0], chunksize):
                    data = in_f.root.raw_data[chunk:chunk + chunksize]    
                    combined_data.append(data)
                    status += chunksize
                    progress_bar.update(status)
        progress_bar.finish()
    return file_combined


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    data_folder = r'/media/davidlp/Data new/Monopix_TBA_CERN/20170922_TestBeamSPSdata/Run01_-100V/Data_Telescope/'
    raw_data_file = [os.path.join(data_folder,
                                  '37_20170920_sps_m26_telescope_scan.h5'),
                     os.path.join(data_folder,
                                  '37_20170920_sps_m26_telescope_scan_1.h5')]
#     raw_data_file = os.path.join(data_folder,
#                                   '37_20170920_sps_m26_telescope_scan.h5')
    interpret_anemone_data(raw_data_file)
