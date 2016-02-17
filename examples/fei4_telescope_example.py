''' Example script to run a full analysis on FE-I4 telescope data. The original data was recorded at DESY with pyBar.
The telescope consists of 6 DUTs with ~ 2 cm distance between the planes. Only the first two and last two planes were taken here.
The first and last plane were IBL n-in-n planar sensors and the 2 devices in the middle 3D CNM/FBK sensors.
'''

import os
import logging
from multiprocessing import Pool

from testbeam_analysis import hit_analysis
from testbeam_analysis import geometry_utils
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis
from testbeam_analysis import plot_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


if __name__ == '__main__':  # main entry point is needed for multiprocessing under windows
    # The location of the data files, one file per DUT
    data_files = [r'data/TestBeamData_FEI4_DUT0.h5',  # the first DUT is the reference DUT defining the coordinate system, called internally DUT0
                  r'data/TestBeamData_FEI4_DUT1.h5',  # DUT1
                  r'data/TestBeamData_FEI4_DUT4.h5',  # DUT2
                  r'data/TestBeamData_FEI4_DUT5.h5'   # DUT3
                  ]

    # Dimensions
    pixel_size = [(250, 50), (250, 50), (250, 50), (250, 50)]  # in um
    n_pixel = [(80, 336), (80, 336), (80, 336), (80, 336)]
    z_positions = [0., 19500, 108800, 128300]  # in um; optional, can be also deduced from data, but usually not with high precision (~ mm)

    output_folder = os.path.split(data_files[0])[0]  # define a folder where all output data and plots are stored
    cluster_files = [os.path.splitext(data_file)[0] + '_cluster.h5' for data_file in data_files]

    # The following shows a complete test beam analysis by calling the seperate function in correct order

# FIXME: need major rework
# Create the initial geometry (to be done once)
#     geometry_utils.create_initial_geometry(geo_file, z_positions)
#     geo_file = os.path.join(output_folder, 'FEI4Geometry.h5')

    # Cluster hits off all DUTs
    args = [{'data_file': data_files[i],
             'max_x_distance': 2,
             'max_y_distance': 1,
             'max_time_distance': 2,
             'max_cluster_hits':1000} for i in range(0, len(data_files))]
    pool = Pool()
    pool.map(hit_analysis.cluster_hits_wrapper, args)  # find cluster on all DUT data files in parallel on multiple cores
    pool.close()
    pool.join()
    plot_utils.plot_cluster_size(cluster_files,
                                 output_pdf=os.path.join(output_folder, 'Cluster_Size.pdf'))

    # Correlate the row / column of each DUT
    dut_alignment.correlate_hits(data_files,
                                 alignment_file=os.path.join(output_folder, 'Correlation.h5'))
    plot_utils.plot_correlations(alignment_file=os.path.join(output_folder, 'Correlation.h5'),
                                 output_pdf=os.path.join(output_folder, 'Correlations.pdf'))

    # Create alignment data for the DUT positions to the first DUT from the correlation data
    # When needed, set offset and error cut for each DUT as list of tuples
    dut_alignment.coarse_alignment(correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                                   alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                   output_pdf=os.path.join(output_folder, 'Alignment.pdf'),
                                   pixel_size=pixel_size)

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(cluster_files,
                                     alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                     tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
                                     pixel_size=pixel_size)

    dut_alignment.check_hit_alignment(tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
                                      output_pdf=os.path.join(output_folder, 'Alignment_Check.pdf'),
                                      combine_n_hits=1000000)

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
                               alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'))

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
                              tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                              output_pdf=os.path.join(output_folder, 'Tracks.pdf'),
                              z_positions=z_positions,
                              fit_duts=[0, 1, 2, 3],
                              include_duts=[-3, -2, -1, 1, 2, 3],
                              track_quality=1)

    # Optional: plot some tracks (or track candidates) of a selected event range
    plot_utils.plot_events(track_file=os.path.join(output_folder, 'Tracks.h5'),
                           output_pdf=os.path.join(output_folder, 'Event.pdf'),
                           z_positions=z_positions,
                           event_range=(0, 10),
                           dut=1)

    # Calculate the residuals to check the alignment
    result_analysis.calculate_residuals(tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                        output_pdf=os.path.join(output_folder, 'Residuals.pdf'),
                                        z_positions=z_positions,
                                        max_chi2=10000)

    # Plot the track density on selected DUT planes
    plot_utils.plot_track_density(tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                  output_pdf=os.path.join(output_folder, 'TrackDensity.pdf'),
                                  z_positions=z_positions,
                                  dim_x=80,
                                  dim_y=336,
                                  pixel_size=pixel_size,
                                  use_duts=None)

    plot_utils.plot_charge_distribution(trackcandidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
                                        output_pdf=os.path.join(output_folder, 'ChargeDistribution.pdf'),
                                        dim_x=(80, 80, 80, 80),
                                        dim_y=(336, 336, 336, 336),
                                        pixel_size=pixel_size)

    # Calculate the efficiency and mean hit/track hit distance
    # When needed, set included column and row range for each DUT as list of tuples
    result_analysis.calculate_efficiency(tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                         output_pdf=os.path.join(output_folder, 'Efficiency.pdf'),
                                         z_positions=z_positions,
                                         bin_size=(250, 50),
                                         minimum_track_density=2,
                                         use_duts=None,
                                         cut_distance=500,
                                         max_distance=500,
                                         col_range=(1250, 17500),
                                         row_range=(1000, 16000))
