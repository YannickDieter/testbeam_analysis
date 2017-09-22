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
import logging
import pyBAR_mimosa26_interpreter.simple_converter_combined as m26_cv

from testbeam_analysis.converter import pybar_fei4_converter as pyfei4_cv


def interpret_anemone_data(raw_data_file):
    ''' Runs all step to analyse telescope raw data '''

    # Step 1: Interpret FE-I4 raw data
    # Output: file with _aligned.h5 suffix
    pyfei4_cv.process_dut(raw_data_file,
                          # Data format has trigger time stamp
                          trigger_data_format=2)

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
            fe_fin=raw_data_file[:-3] + '_event_aligned.h5',
            fout=raw_data_file[:-3] + '_run_aligned_%d.h5' % plane,
            tr=True,  # Switch column / row (transpose)
            frame=False)  # Add frame info (not working?)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    data_folder = r'/media/davidlp/Data new/Monopix_TBA_CERN/20170922_TestBeamSPSdata/Run01_-100V/Data_Telescope/'
    raw_data_file = os.path.join(data_folder, '37_20170920_sps_m26_telescope_scan.h5')

    interpret_anemone_data(raw_data_file)
