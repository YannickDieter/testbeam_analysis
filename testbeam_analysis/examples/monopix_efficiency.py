''' This example investigates the efficiency of a radiation hard DMAP sensor.

    It is a prototype called Monopix.
    A Mimosa26 + ATLAS-FE-I4 telescope (named Anemone) is used that is read out
    with pyBAR.
'''

import os
import logging
import numpy as np
import tables as tb
from numba import njit

from testbeam_analysis import hit_analysis
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis
from testbeam_analysis.tools import data_selection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def run_analysis():
    data_folder = r'/media/davidlp/Data new/Monopix_TBA_CERN/20170922_TestBeamSPSdata/Run01_-100V/'
    # The location of the example data files, one file per DUT
    # Only 8 devices supported yet, skip first
    data_files = [  # The first device is the reference for the coordinate system
        # os.path.join(data_folder, '37_20170920_sps_m26_telescope_scan_run_aligned_6'),
        os.path.join(data_folder, 'Data_Telescope/37_20170920_sps_m26_telescope_scan_run_aligned_5.h5'),
        os.path.join(data_folder, 'Data_Telescope/37_20170920_sps_m26_telescope_scan_run_aligned_4.h5'),
        os.path.join(data_folder, 'Data_B18/20170921_225457_scan_simple_ev.h5'),  # Mono 1, DUT
        os.path.join(data_folder, 'Data_B19/20170921_225500_scan_simple_ev.h5'),  # Mono 2, reference
        os.path.join(data_folder, 'Data_Telescope/37_20170920_sps_m26_telescope_scan_run_aligned_3.h5'),
        os.path.join(data_folder, 'Data_Telescope/37_20170920_sps_m26_telescope_scan_run_aligned_2.h5'),
        os.path.join(data_folder, 'Data_Telescope/37_20170920_sps_m26_telescope_scan_run_aligned_1.h5'),
        os.path.join(data_folder, 'Data_Telescope/37_20170920_sps_m26_telescope_scan_aligned.h5')]  # FE-I4 time reference]

    # Pixel dimesions and matrix size of the DUTs
    pixel_size = [(18.4, 18.4), (18.4, 18.4),
                  (50, 250), (50, 250),  # Transposed
                  (18.4, 18.4), (18.4, 18.4), (18.4, 18.4),  # (18.4, 18.4),
                  (250, 50)]  # (Column, row) pixel pitch in um

    n_pixels = [(576, 1152), (576, 1152),
                (129, 36), (129, 36),
                (576, 1152), (576, 1152), (576, 1152),  # (576, 1152),
                (80, 336)]  # (Column, row) dimensions of the pixel matrix

    z_positions = np.array([47700., 96100.,
                            115750., 137750.,
                            173100., 222500., 270500.,
                            291500.
                            ]) - 47700.

    dut_names = (  # "Tel 0",
        "Tel 5", "Tel 4",
        "Mono 1", "Mono 2",
        "Tel 3", "Tel 2", "Tel 1",
        "FEI4 Reference",)  # Friendly names for plotting

    # Folder where all output data and plots are stored
    output_folder = r'/media/davidlp/Data new/Monopix_TBA_CERN/20170922_TestBeamSPSdata/Run01_-100V/output/'

    data_files[2], data_files[3] = fix_mono_data(data_files=[data_files[2],
                                                             data_files[3]])

    for i, hit_file in enumerate(data_files):
        hit_analysis.check_file(
            input_hits_file=hit_file,
            n_pixel=n_pixels[i],
        )

    # The following shows a complete test beam analysis by calling the seperate
    # function in correct order

    # Generate noisy pixel mask for all DUTs
    threshold = [2, 2, 10, 10, 2, 2, 2, 10]
    for i, data_file in enumerate(data_files):
        hit_analysis.generate_pixel_mask(
            input_hits_file=data_file,
            n_pixel=n_pixels[i],
            pixel_mask_name='NoisyPixelMask',
            pixel_size=pixel_size[i],
            threshold=threshold[i],
            dut_name=dut_names[i])

    # Cluster hits from all DUTs
    column_cluster_distance = [3, 3, 2, 2, 3, 3, 3, 1]
    row_cluster_distance = [3, 3, 3, 3, 3, 3, 3, 2]
    frame_cluster_distance = [0, 0, 255, 255, 0, 0, 0, 1]
    for i, data_file in enumerate(data_files):
        hit_analysis.cluster_hits(
            input_hits_file=data_file,
            input_noisy_pixel_mask_file=os.path.splitext(data_file)[0] + '_noisy_pixel_mask.h5',
            min_hit_charge=0,
            max_hit_charge=2**15 if i != 7 else 15,
            column_cluster_distance=column_cluster_distance[i],
            row_cluster_distance=row_cluster_distance[i],
            frame_cluster_distance=frame_cluster_distance[i],
            dut_name=dut_names[i])

    # Generate filenames for cluster data
    input_cluster_files = [os.path.splitext(data_file)[0] + '_clustered.h5'
                           for data_file in data_files]

    # Correlate the row / column of each DUT
    dut_alignment.correlate_cluster(
        input_cluster_files=input_cluster_files,
        output_correlation_file=os.path.join(output_folder,
                                             'Correlation.h5'),
        n_pixels=n_pixels,
        pixel_size=pixel_size,
        dut_names=dut_names)

    # Create prealignment relative to the first DUT from the correlation data
    dut_alignment.prealignment(
        input_correlation_file=os.path.join(output_folder,
                                            'Correlation.h5'),
        output_alignment_file=os.path.join(output_folder,
                                           'Alignment.h5'),
        z_positions=z_positions,
        pixel_size=pixel_size,
        dut_names=dut_names,
        fit_background=True,
        non_interactive=True,
        iterations=5)

    # Merge the cluster tables to one merged table aligned at the event number
    dut_alignment.merge_cluster_data(
        input_cluster_files=input_cluster_files,
        output_merged_file=os.path.join(output_folder,
                                        'Merged.h5'),
        n_pixels=n_pixels,
        pixel_size=pixel_size)

    # Apply the prealignment to the merged cluster table to create tracklets
    dut_alignment.apply_alignment(
        input_hit_file=os.path.join(output_folder,
                                    'Merged.h5'),
        input_alignment_file=os.path.join(output_folder,
                                          'Alignment.h5'),
        output_hit_file=os.path.join(output_folder,
                                     'Tracklets_prealigned.h5'),
        force_prealignment=True)

    # Find tracks from the prealigned tracklets and stores the with quality
    # indicator into track candidates table
    track_analysis.find_tracks(
        input_tracklets_file=os.path.join(output_folder,
                                          'Tracklets_prealigned.h5'),
        input_alignment_file=os.path.join(output_folder,
                                          'Alignment.h5'),
        output_track_candidates_file=os.path.join(
            output_folder,
            'TrackCandidates_prealignment.h5'))

    # Select tracks with any hit in the time reference and all position
    # devices to increase analysis speed due to data reduction
    data_selection.select_hits(
        hit_file=os.path.join(output_folder,
                              'TrackCandidates_prealignment.h5'),
        track_quality=0b11110011,
        track_quality_mask=0b11110011)

    # Do an alignment step with the track candidates, corrects rotations and is therefore much more precise than simple prealignment
    dut_alignment.alignment(
        input_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates_prealignment_reduced.h5'),
        input_alignment_file=os.path.join(
            output_folder, 'Alignment.h5'),
        # Order of combinaions of planes to align, one should start with high resoultion planes (here: telescope planes)
        align_duts=[[0, 1, 4, 5, 6],  # align the telescope planes first
                    [7],  # align the time reference after the telescope alignment
                    [2],
                    [3]],  # align the DUT last and not with the reference since it is rather small and would make the time reference alinmnt worse
        # The DUTs to be used in the fit, always just the high resolution Mimosa26 planes used
        selection_fit_duts=[0, 1, 4, 5, 6],
        # The DUTs to be required to have a hit for the alignment
        selection_hit_duts=[[0, 1, 4, 5, 6, 7],  # Take tracks with time reference hit
                            [0, 1, 4, 5, 6, 7],  # Take tracks with time reference hit
                            [0, 1, 2, 3, 4, 5, 6],
                            [0, 1, 3, 4, 5, 6]],
        # The required track quality per alignment step and DUT
        selection_track_quality=[[1, 1, 1, 1, 1, 0],  # Do not require a good hit in the time refernce
                                 [1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1]],  # Do not require a good hit in the small DUT
        # DUTS 2, 3, 7 are heavily rotated (inverted),
        # this is not implemented now. Thus one has to set
        # the correct rotation angles here manually, sorry
        initial_rotation=[[0., 0., 0.],
                          [0., 0., 0.],
                          [np.pi, 0., 0.],
                          [np.pi, 0., 0.],
                          [0., 0, 0.],
                          [0., 0, 0.],
                          [0., 0, 0.],
                          [0., np.pi, 0.0]],
        # DUTS 2, 3, 7 are heavily rotated (inverted),
        # this is not implemented now. Thus one has to set
        # the correct rotation angles here manually, sorry
        initial_translation=[[0., 0, 0.],
                             [0., 0, 0.],
                             [-883.+0.6, -331.7-2.4-3.6, 0.],
                             [-591.9+2.2+1.7, -535.3-5.4-6.8, 0.],
                             [0., 0., 0.],
                             [0., 0, 0.],
                             [0., 0, 0.],
                             [1567.-21.2, -1485.-17.9, 0.]],
        n_pixels=n_pixels,
        # Do the alignment only on a subset of data, for reasonable run time
        use_n_tracks=200000,
        pixel_size=pixel_size,
        plot=True)  # Show result residuals after alignment

    # Apply new alignment to data
    # Revert alignment from track candidates. Usually one would just apply the
    # alignment to the merged data.
    # Due to the large beam angle track finding fails on aligned data.
    # Thus rely on the found tracks from prealignment.
    dut_alignment.apply_alignment(
        input_hit_file=os.path.join(output_folder, 'TrackCandidates_prealignment_reduced.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        # This is the new not aligned but preselected merged data file to apply (pre-) alignment on
        output_hit_file=os.path.join(output_folder, 'Merged_small.h5'),
        inverse=True,
        force_prealignment=True)

    # Apply the alignment to the merged cluster table to create tracklets
    dut_alignment.apply_alignment(
        input_hit_file=os.path.join(output_folder, 'Merged_small.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_hit_file=os.path.join(output_folder, 'TrackCandidates.h5'))

    # Fit track using alignment
    track_analysis.fit_tracks(
        input_track_candidates_file=os.path.join(output_folder,
                                                 'TrackCandidates.h5'),
        input_alignment_file=os.path.join(output_folder,
                                          'Alignment.h5'),
        output_tracks_file=os.path.join(output_folder,
                                        'Tracks.h5'),
        fit_duts=[2, 3, 4],
        selection_hit_duts=[[0, 1, 3, 4, 5, 6, 7],
                            [0, 1, 2, 4, 5, 6, 7],
                            [0, 1, 4, 5, 6, 7]],
        selection_fit_duts=[0, 1, 4, 5, 6],
        # Take all tracks with any hits, but good time reference hits
        selection_track_quality=[[0, 0, 1, 0, 0, 0, 1],
                                 [0, 0, 1, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 1]],
        min_track_distance=True)

    # Create unconstrained residuals with aligned data
    result_analysis.calculate_residuals(
        input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals.h5'),
        n_pixels=n_pixels,
        pixel_size=pixel_size)

    # Calculate efficiency with aligned data
    result_analysis.calculate_efficiency(
        input_tracks_file=os.path.join(output_folder,
                                       'Tracks.h5'),
        input_alignment_file=os.path.join(output_folder,
                                          'Alignment.h5'),
        output_efficiency_file=os.path.join(output_folder,
                                            'Efficiency.h5'),
        bin_size=[(50, 50)] * 8,
        pixel_size=pixel_size,
        n_pixels=n_pixels,
        minimum_track_density=1,
        max_chi2=1000.,
        sensor_size=[(pixel_size[i][0] * n_pixels[i][0],
                      pixel_size[i][1] * n_pixels[i][1]) for i in range(8)])


def fix_mono_data(data_files):
    ''' Fixes data files containing Monopix hit data

        Needed fixes are:
          1. filter out hits with col=255 info
          2. fix event number reset at 2^15
          3. Rearrange hit columns

        Returns list with new, fixed files names.
    '''

    fixed_files = []

    for data_file in data_files:
        reorder_columns(hits_file_in=data_file,
                        hits_file_out=data_file[:-3] + '_tmp.h5')

        data_selection.select_hits(hit_file=data_file[:-3] + '_tmp.h5',
                                   # is column since transposed
                                   condition='row < 36',
                                   output_file=data_file[:-3] + '_tmp_2.h5')

        # Fix event number overflow at 2^15
        fix_event_number(hits_file_in=data_file[:-3] + '_tmp_2.h5',
                         hits_file_out=data_file[:-3] + '_fixed.h5')

        fixed_files.append(data_file[:-3] + '_fixed.h5')

        os.remove(data_file[:-3] + '_tmp.h5')
        os.remove(data_file[:-3] + '_tmp_2.h5')

    return fixed_files


def reorder_columns(hits_file_in, hits_file_out):
    ''' Reorder columns to follow TBA ordering '''

    hit_dtype = np.dtype([('event_number', np.int64),
                          ('frame', np.uint8),
                          ('column', np.uint16),
                          ('row', np.uint16),
                          ('charge', np.uint16)])

    with tb.open_file(hits_file_in) as in_file:
        node = in_file.root.Hits
        hits = node[:]
        with tb.open_file(hits_file_out, 'w') as out_file:
            hits_out_array = np.zeros(shape=hits.shape, dtype=hit_dtype)
            for col_name in hits.dtype.names:
                hits_out_array[col_name] = hits[col_name]
            hits_out = out_file.create_table(out_file.root, name=node.name,
                                             description=hit_dtype,
                                             title=node.title,
                                             filters=tb.Filters(
                                                 complib='blosc',
                                                 complevel=5,
                                                 fletcher32=False))
            hits_out.append(hits_out_array)


def fix_event_number(hits_file_in, hits_file_out):
    with tb.open_file(hits_file_in) as in_file:
        node = in_file.root.Hits
        hits = node[:]
        with tb.open_file(hits_file_out, 'w') as out_file:
            hits_out = out_file.create_table(out_file.root, name=node.name,
                                             description=node.dtype,
                                             title=node.title,
                                             filters=tb.Filters(
                                                 complib='blosc',
                                                 complevel=5,
                                                 fletcher32=False))
            fix_hits(hits)
            hits_out.append(hits)


@njit
def add_offset(hits, index, offset):
    for i in range(index, hits.shape[0]):
        hits[i]['event_number'] += offset


@njit
def fix_hits(hits):
    ''' Event number overflow at 2^15 fix '''

    old_hit = hits[0]
    for i, hit in enumerate(hits):
        if hit['event_number'] < old_hit['event_number']:
            add_offset(hits,
                       index=i,
                       offset=2**15)
        old_hit = hit


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows
    run_analysis()
