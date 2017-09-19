''' All functions acting on the hits of one DUT are listed here'''
from __future__ import division

import logging
import os.path
import re

import tables as tb
import numpy as np
from scipy.ndimage import median_filter
from pixel_clusterizer.clusterizer import HitClusterizer

from testbeam_analysis.tools import smc
from testbeam_analysis.tools import analysis_utils, plot_utils
from testbeam_analysis.tools.plot_utils import plot_masked_pixels, plot_cluster_size


def check_file(input_hits_file, n_pixel, output_check_file=None,
               event_range=1, plot=True, chunk_size=1000000):
    '''Checks the hit table to have proper data.

    The checks include:
      - hit definitions:
          - position has to start at 1 (not 0)
          - position should not exceed number of pixels (n_pixel)
      - event building
          - event number has to be strictly monotone
          - hit position correlations of consecutive events are
            created. Should be zero for distinctly
            built events.

    Parameters
    ----------
    input_hits_file : string
        File name of the hit table.
    output_check_file : string
        Filename of the output file with the correlation histograms.
    n_pixel : tuple
        Tuple of the total number of pixels (column/row).
    event_range : integer
        The range of events to correlate.
        E.g.: event_range = 2 correlates to predecessing event hits.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''

    logging.info('=== Check data of hit file %s ===', input_hits_file)

    if output_check_file is None:
        output_check_file = input_hits_file[:-3] + '_check.h5'

    with tb.open_file(output_check_file, mode="w") as out_file_h5:
        with tb.open_file(input_hits_file, 'r') as input_file_h5:
            shape_column = (n_pixel[0], n_pixel[0])
            shape_row = (n_pixel[1], n_pixel[1])
            col_corr = np.zeros(shape_column, dtype=np.int)
            row_corr = np.zeros(shape_row, dtype=np.int)
            last_event = None
            out_dE = out_file_h5.create_earray(out_file_h5.root, name='EventDelta',
                                               title='Change of event number per non empty event',
                                               shape=(0, ),
                                               atom=tb.Atom.from_dtype(np.dtype(np.uint64)),
                                               filters=tb.Filters(complib='blosc',
                                                                  complevel=5,
                                                                  fletcher32=False))
            out_E = out_file_h5.create_earray(out_file_h5.root, name='EventNumber',
                                              title='Event number of non empty event',
                                              shape=(0, ),
                                              atom=tb.Atom.from_dtype(np.dtype(np.uint64)),
                                              filters=tb.Filters(complib='blosc',
                                                                 complevel=5,
                                                                 fletcher32=False))

            for hits, _ in analysis_utils.data_aligned_at_events(
                    input_file_h5.root.Hits,
                    chunk_size=chunk_size):
                if not np.all(np.diff(hits['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. \
                    The hits cannot be used like this!')
                if np.any(hits['column'] < 1) or np.any(hits['row'] < 1):
                    raise RuntimeError('The column/row definition does not \
                    start at 1!')
                if (np.any(hits['column'] > n_pixel[0])
                        or np.any(hits['row'] > n_pixel[1])):
                    raise RuntimeError('The column/row definition exceed the nuber \
                    of pixels (%s/%s)!', n_pixel[0], n_pixel[1])

                analysis_utils.correlate_hits_on_event_range(hits,
                                                             col_corr,
                                                             row_corr,
                                                             event_range)

                event_numbers = np.unique(hits['event_number'])
                event_delta = np.diff(event_numbers)

                if last_event:
                    event_delta = np.concatenate((np.array([event_numbers[0] - last_event]),
                                                  event_delta))
                last_event = event_numbers[-1]

                out_dE.append(event_delta)
                out_E.append(event_numbers)

            out_col = out_file_h5.create_carray(out_file_h5.root, name='CorrelationColumns',
                                                title='Column Correlation with event range=%s' % event_range,
                                                atom=tb.Atom.from_dtype(col_corr.dtype),
                                                shape=col_corr.shape,
                                                filters=tb.Filters(complib='blosc',
                                                                   complevel=5,
                                                                   fletcher32=False))
            out_row = out_file_h5.create_carray(out_file_h5.root, name='CorrelationRows',
                                                title='Row Correlation with event range=%s' % event_range,
                                                atom=tb.Atom.from_dtype(row_corr.dtype),
                                                shape=row_corr.shape,
                                                filters=tb.Filters(complib='blosc',
                                                                   complevel=5,
                                                                   fletcher32=False))
            out_col[:] = col_corr
            out_row[:] = row_corr

    if plot:
        plot_utils.plot_checks(input_corr_file=output_check_file)


def generate_pixel_mask(input_hits_file, n_pixel, pixel_mask_name="NoisyPixelMask", output_mask_file=None, pixel_size=None, threshold=10.0, filter_size=3, dut_name=None, plot=True, chunk_size=1000000):
    '''Generating pixel mask from the hit table.

    Parameters
    ----------
    input_hits_file : string
        File name of the hit table.
    n_pixel : tuple
        Tuple of the total number of pixels (column/row).
    pixel_mask_name : string
        Name of the node containing the mask inside the output file.
    output_mask_file : string
        File name of the output mask file.
    pixel_size : tuple
        Tuple of the pixel size (column/row). If None, assuming square pixels.
    threshold : float
        The threshold for pixel masking. The threshold is given in units of
        sigma of the pixel noise (background subtracted). The lower the value
        the more pixels are masked.
    filter_size : scalar or tuple
        Adjust the median filter size by giving the number of columns and rows.
        The higher the value the more the background is smoothed and more
        pixels are masked.
    dut_name : string
        Name of the DUT. If None, file name of the hit table will be printed.
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Generating %s for %s ===', ' '.join(item.lower() for item in re.findall('[A-Z][^A-Z]*', pixel_mask_name)), input_hits_file)

    if output_mask_file is None:
        output_mask_file = os.path.splitext(input_hits_file)[0] + '_' + '_'.join(item.lower() for item in re.findall('[A-Z][^A-Z]*', pixel_mask_name)) + '.h5'

#     # Create occupancy histogram
#     def work(hit_chunk):
#         col, row = hit_chunk['column'], hit_chunk['row']
#         return analysis_utils.hist_2d_index(col - 1, row - 1, shape=n_pixel)
#                  
#     smc.SMC(table_file_in=input_hits_file,
#         file_out=output_mask_file,
#         func=work,
#         node_desc={'name':'HistOcc'},
#         n_cores=1,
#         chunk_size=chunk_size)
#  
#     # Create mask from occupancy histogram
#     with tb.open_file(output_mask_file, 'r+') as out_file_h5:
#         occupancy = out_file_h5.root.HistOcc[:]
#         # Run median filter across data, assuming 0 filling past the edges to get expected occupancy
#         blurred = median_filter(occupancy.astype(np.int32), size=filter_size, mode='constant', cval=0.0)
#         # Spot noisy pixels maxima by substracting expected occupancy
#         difference = np.ma.masked_array(occupancy - blurred)
#         std = np.ma.std(difference)
#         abs_occ_threshold = threshold * std
#         occupancy = np.ma.masked_where(difference > abs_occ_threshold, occupancy)
#         logging.info('Masked %d pixels at threshold %.1f in %s', np.ma.count_masked(occupancy), threshold, input_hits_file)
#         # Generate tuple col / row array of hot pixels, do not use getmask()
#         pixel_mask = np.ma.getmaskarray(occupancy)
#  
#         # Create masked pixels array
#         masked_pixel_table = out_file_h5.create_carray(out_file_h5.root, name=pixel_mask_name, title='Pixel Mask', atom=tb.Atom.from_dtype(pixel_mask.dtype), shape=pixel_mask.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
#         masked_pixel_table[:] = pixel_mask
#        
#     if plot:
#         plot_masked_pixels(input_mask_file=output_mask_file, pixel_size=pixel_size, dut_name=dut_name)
    occupancy = None
    # Calculating occupancy array
    with tb.open_file(input_hits_file, 'r') as input_file_h5:
        for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
            col, row = hits['column'], hits['row']
            chunk_occ = analysis_utils.hist_2d_index(col - 1, row - 1, shape=n_pixel)
            if occupancy is None:
                occupancy = chunk_occ
            else:
                occupancy = occupancy + chunk_occ
 
    # Run median filter across data, assuming 0 filling past the edges to get expected occupancy
    blurred = median_filter(occupancy.astype(np.int32), size=filter_size, mode='constant', cval=0.0)
    # Spot noisy pixels maxima by substracting expected occupancy
    difference = np.ma.masked_array(occupancy - blurred)
    std = np.ma.std(difference)
    abs_occ_threshold = threshold * std
    occupancy = np.ma.masked_where(difference > abs_occ_threshold, occupancy)
    logging.info('Masked %d pixels at threshold %.1f in %s', np.ma.count_masked(occupancy), threshold, input_hits_file)
    # Generate tuple col / row array of hot pixels, do not use getmask()
    pixel_mask = np.ma.getmaskarray(occupancy)
 
    with tb.open_file(output_mask_file, 'w') as out_file_h5:
        # Create occupancy array without masking pixels
        occupancy_array_table = out_file_h5.create_carray(out_file_h5.root, name='HistOcc', title='Occupancy Histogram', atom=tb.Atom.from_dtype(occupancy.dtype), shape=occupancy.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        occupancy_array_table[:] = np.ma.getdata(occupancy)
 
        # Create masked pixels array
        masked_pixel_table = out_file_h5.create_carray(out_file_h5.root, name=pixel_mask_name, title='Pixel Mask', atom=tb.Atom.from_dtype(pixel_mask.dtype), shape=pixel_mask.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        masked_pixel_table[:] = pixel_mask
 
    if plot:
        plot_masked_pixels(input_mask_file=output_mask_file, pixel_size=pixel_size, dut_name=dut_name)

    return output_mask_file


def cluster_hits(input_hits_file, output_cluster_file=None, create_cluster_hits_table=False, input_disabled_pixel_mask_file=None, input_noisy_pixel_mask_file=None, min_hit_charge=0, max_hit_charge=None, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, dut_name=None, plot=True, chunk_size=1000000):
    '''Clusters the hits in the data file containing the hit table.

    Parameters
    ----------
    input_hits_file : string
        Filename of the input hits file.
    output_cluster_file : string
        Filename of the output cluster file. If None, the filename will be derived from the input hits file.
    create_cluster_hits_table : bool
        If True, additionally create cluster hits table.
    input_disabled_pixel_mask_file : string
        Filename of the input disabled mask file.
    input_noisy_pixel_mask_file : string
        Filename of the input disabled mask file.
    min_hit_charge : uint
        Minimum hit charge. Minimum possible hit charge must be given in order to correcly calculate the cluster coordinates.
    max_hit_charge : uint
        Maximum hit charge. Hits wit charge above the limit will be ignored.
    column_cluster_distance : uint
        Maximum column distance between hist so that they are assigned to the same cluster. Value of 0 effectively disables the clusterizer in column direction.
    row_cluster_distance : uint
        Maximum row distance between hist so that they are assigned to the same cluster. Value of 0 effectively disables the clusterizer in row direction.
    frame_cluster_distance : uint
        Sometimes an event has additional timing information (e.g. bunch crossing ID, frame ID). Value of 0 effectively disables the clusterization in time.
    dut_name : string
        Name of the DUT. If None, filename of the output cluster file will be used.
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Clustering hits in %s ===', input_hits_file)

    if output_cluster_file is None:
        output_cluster_file = os.path.splitext(input_hits_file)[0] + '_clustered.h5'

    # Calculate the size in col/row for each cluster
    # This is a end of cluster function automatically
    # called when a cluster is finished
    def calc_cluster_dimensions(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
        min_col = hits[cluster_hit_indices[0]].column
        max_col = hits[cluster_hit_indices[0]].column
        min_row = hits[cluster_hit_indices[0]].row
        max_row = hits[cluster_hit_indices[0]].row
        for i in cluster_hit_indices[1:]:
            if i < 0:  # Not used indeces = -1
                break
            if hits[i].column < min_col:
                min_col = hits[i].column
            if hits[i].column > max_col:
                max_col = hits[i].column
            if hits[i].row < min_row:
                min_row = hits[i].row
            if hits[i].row > max_row:
                max_row = hits[i].row
        clusters[cluster_index].err_cols = max_col - min_col + 1
        clusters[cluster_index].err_rows = max_row - min_row + 1

    with tb.open_file(input_hits_file, 'r') as input_file_h5:
        with tb.open_file(output_cluster_file, 'w') as output_file_h5:
            if input_disabled_pixel_mask_file is not None:
                with tb.open_file(input_disabled_pixel_mask_file, 'r') as input_mask_file_h5:
                    disabled_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.DisabledPixelMask[:]))[0] + 1
                    input_mask_file_h5.root.DisabledPixelMask._f_copy(newparent=output_file_h5.root)
            else:
                disabled_pixels = None
            if input_noisy_pixel_mask_file is not None:
                with tb.open_file(input_noisy_pixel_mask_file, 'r') as input_mask_file_h5:
                    noisy_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.NoisyPixelMask[:]))[0] + 1
                    input_mask_file_h5.root.NoisyPixelMask._f_copy(newparent=output_file_h5.root)
            else:
                noisy_pixels = None

            clusterizer = HitClusterizer(column_cluster_distance=column_cluster_distance, row_cluster_distance=row_cluster_distance, frame_cluster_distance=frame_cluster_distance, min_hit_charge=min_hit_charge, max_hit_charge=max_hit_charge)
            clusterizer.add_cluster_field(description=('err_cols', '<f4'))  # Add an additional field to hold the cluster size in x
            clusterizer.add_cluster_field(description=('err_rows', '<f4'))  # Add an additional field to hold the cluster size in y
            clusterizer.set_end_of_cluster_function(calc_cluster_dimensions)  # Set the new function to the clusterizer

            cluster_hits_table = None
            cluster_table = None
            for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
                if not np.all(np.diff(hits['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. The hits cannot be used like this!')
                cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=noisy_pixels, disabled_pixels=disabled_pixels)  # Cluster hits
                if not np.all(np.diff(clusters['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. The cluster cannot be used like this!')
                # create cluster hits table dynamically
                if create_cluster_hits_table and cluster_hits_table is None:
                    cluster_hits_table = output_file_h5.create_table(output_file_h5.root, name='ClusterHits', description=cluster_hits.dtype, title='Cluster hits table', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                # create cluster table dynamically
                if cluster_table is None:
                    cluster_table = output_file_h5.create_table(output_file_h5.root, name='Cluster', description=clusters.dtype, title='Cluster table', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                if create_cluster_hits_table:
                    cluster_hits_table.append(cluster_hits)
                cluster_table.append(clusters)

    def get_eff_pitch(hist, cluster_size):
        ''' Effective pitch to describe the cluster
            size propability distribution

        hist : array like
            Histogram with cluster size distribution
        cluster_size : Cluster size to calculate the pitch for
        '''

        return np.sqrt(hight[int(cluster_size)].astype(np.float) / hight.sum())

    # Calculate cluster size histogram
    with tb.open_file(output_cluster_file, 'r') as input_file_h5:
        hight = None
        n_hits = 0
        n_clusters = input_file_h5.root.Cluster.nrows
        for start_index in range(0, n_clusters, chunk_size):
            cluster_n_hits = input_file_h5.root.Cluster[start_index:start_index + chunk_size]['n_hits']
            # calculate cluster size histogram
            if hight is None:
                max_cluster_size = np.amax(cluster_n_hits)
                hight = analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
            elif max_cluster_size < np.amax(cluster_n_hits):
                max_cluster_size = np.amax(cluster_n_hits)
                hight.resize(max_cluster_size + 1)
                hight += analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
            else:
                hight += analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
            n_hits += np.sum(cluster_n_hits)

    # Calculate cluster size histogram
    with tb.open_file(output_cluster_file, 'r+') as io_file_h5:
        for start_index in range(0, io_file_h5.root.Cluster.nrows, chunk_size):
            clusters = io_file_h5.root.Cluster[start_index:start_index + chunk_size]
            # Set errors for small clusters, where charge sharing enhances resolution
            for css in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                sel = np.logical_and(clusters['err_cols'] == css[0], clusters['err_rows'] == css[1])
                clusters['err_cols'][sel] = get_eff_pitch(hist=hight, cluster_size=css[0]) / np.sqrt(12)
                clusters['err_rows'][sel] = get_eff_pitch(hist=hight, cluster_size=css[1]) / np.sqrt(12)
            # Set errors for big clusters, where delta electrons reduce resolution
            sel = np.logical_or(clusters['err_cols'] > 2, clusters['err_rows'] > 2)
            clusters['err_cols'][sel] = clusters['err_cols'][sel] / np.sqrt(12)
            clusters['err_rows'][sel] = clusters['err_rows'][sel] / np.sqrt(12)
            io_file_h5.root.Cluster[start_index:start_index + chunk_size] = clusters

    if plot:
        plot_cluster_size(hight, n_hits, n_clusters, max_cluster_size,
                          dut_name=os.path.split(output_cluster_file)[1],
                          output_pdf_file=os.path.splitext(output_cluster_file)[0] + '_cluster_size.pdf')

    return output_cluster_file
