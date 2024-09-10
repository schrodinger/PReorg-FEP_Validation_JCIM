# HARD-CODED version to fix residue numbering snafu

import sys, os
import base64
import csv
import itertools
import json
import functools
import operator
import pickle
import math
import random

#FIXME
sys.setrecursionlimit(10000)

from collections import Counter

from schrodinger.application.prime.packages import executeparallel
from schrodinger.application.prime.packages import primeparser
from schrodinger import structure
from schrodinger.structutils import analyze, rmsd
import schrodinger.application.desmond.cms as cms
from schrodinger.application.desmond.packages import topo, traj_util

import numpy as np
import pylab
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform

usage = ""
version = ""


class CartesianCustomDistanceClusteringToFBHW(executeparallel.ExecuteParallel):
    def __init__(self, script_file):
        executeparallel.ExecuteParallel.__init__(self, script_file,
            usage, [], version=version)

    def addParserArguments(self, parser):
        parser.add_argument("trajectory",
                nargs='+',
                action=primeparser.StoreFile,
                help="CMS trajectories")
        parser.add_argument("-r", "--reference",
                action=primeparser.StoreFile)
        parser.add_argument("--fitasl", action=primeparser.StoreAsl,
                help='ASL for fit atoms, otherwise assume the trajectories are aligned')
        parser.add_argument("--asl", action=primeparser.StoreAsl,
                help='ASL for RMSD atoms')
        parser.add_argument("--cutoff", type=float, default=1.0,
                help="Metric to cutoff at")
        parser.add_argument("--min_size", type=int, default=5,
                help='Minimum number of frames to output a cluster')
        parser.add_argument("-w", "--whisker",
                default=1.5,
                type=float,
                help="Multiplier for IQR whisker")
        parser.add_argument('--method', default='centroid',
                help='Linkage clustering method (eg. ward/centroid)')
        parser.add_argument("--threshold", default=4.0, type=float,
                help="Threshold for sigmoid (A)")
        parser.add_argument("--softness", default=0.5, type=float,
                help="Softness for sigmoid (eg. 0.1 hard, 1.0 soft)")
        parser.add_argument("--vmin", default=None, type=float, help='Minimum RMSD for color scale')
        parser.add_argument("--vmax", default=None, type=float, help='Maximum RMSD for color scale')
        parser.add_argument("--min", default=None, type=float, help='Minimum torsion distance for boxplot')
        parser.add_argument("--max", default=None, type=float, help='Maximum torsion distance for boxplot')
        parser.add_argument("--npz", action=primeparser.StoreFile,
                help="NPZ from a previous run, plot only")
        parser.add_argument("--num_reps", type=int, default=1,
                help="number of representatives for each cluster")
        parser.add_argument("-e", "--extract",
                choices=('best', 'random', 'worst', 'per_traj', 'all'),
                default='best',
                help="Different style of selecting which frames to extract for each cluster.")
        parser.add_argument("-l", "--label",
                default=[], action='append',
                help="Label for the each trajectory, otherwise they will be referred to by index")

    def preLaunchActions(self, opts, launch):
        # Get the additional trajectory files/dirs
        for trj in opts.trajectory:
            with structure.StructureReader(self.getPath(trj)) as r:
                obj = next(r)
                for prop in ("s_m_original_cms_file", "s_chorus_trajectory_file"):
                    if prop not in obj.property: continue
                    # Need to get the relative path from the current input file, not assume it is local
                    if os.path.isfile(obj.property[prop]):
                        launch.addInputFile(obj.property[prop])
                    elif os.path.isdir(obj.property[prop]):
                        for (dirpath, dirnames, filenames) in os.walk(obj.property[prop]):
                            for fname in filenames:
                                launch.addInputFile(os.path.join(dirpath, fname))

    def runBackend(self, opts, istage):
        if opts.npz:
            data = np.load(opts.npz)
            cdm = data['condensed_distance_matrix']
            ref_vals = data['ref_vals']
            xyz_timeseries = data['xyz_timeseries']
            trj_frames = data['trj_frames']
            assert len(trj_frames) == len(opts.trajectory), f"Provided {len(opts.trajectory)} trajectories, but NPZ contains data for {len(trj_frames)}"
        else:
            cdm, ref_vals, xyz_timeseries, trj_frames = self.parseFrames(opts)
        print(f"Clustering {sum(trj_frames)} frames...")
        linkageMatrix, distances = self.clusterData(opts, cdm)
        #FIXME the pcolormesh calls explode
        #print("Plotting raw data...")
        self.plotData(opts, linkageMatrix, distances, trj_frames)
        if opts.cutoff:
            print(f"Applying cluster cutoff of {opts.cutoff} and analyzing")
            cluster_indices, num_linkages_skipped = linkage_up_to_cutoff(linkageMatrix, opts.cutoff, opts.min_size)
            self.timeseriesClusters(opts, cluster_indices, distances, trj_frames)
            cluster_composition = self.tabulateClusterComposition(opts, cluster_indices, trj_frames)
            # Extractors should be functions that obey the following conventions:
            #    :param frame_scores: List of each frame's mean distance to other cluster members. Assumed to be sorted in ascending order.
            #    :type frame_scores: list[tuple], where each tuple is (relative_idx for frame within cluster,
            #                                                          frame_idx for sequentially loaded trajectories,
            #                                                          mean_cluster_distance for the mean distance to other cluster members)
            #    :return: Frames selected for extraction
            #    :rtype: list[tuple], where each tuple is (rank within the cluster for that frame {starts at 0},
            #                                              relative_idx for the frame within the cluster,
            #                                              frame_idx for sequentially loaded trajectories)

            # Extract the 'best', or lowest mean distances from all other cluster members
            if opts.extract == "best":
                def min_num_rep_extractor(frame_scores):
                    selected_frames = []
                    num_reps = min(opts.num_reps, len(frame_scores))
                    for idx in range(num_reps):
                        selected_frames.append((idx, frame_scores[idx][0], frame_scores[idx][1]))
                    return selected_frames
                self.outputClusterSortByMean(opts, cluster_indices, distances, xyz_timeseries, trj_frames, min_num_rep_extractor)
            # Extract a 'random' set of frames
            elif opts.extract == "random":
                def random_num_rep_extractor(frame_scores):
                    selected_frames = []
                    num_reps = min(opts.num_reps, len(frame_scores))
                    for i in range(num_reps):
                        idx = random.randrange(0, len(frame_scores))
                        selected_frames.append((idx, frame_scores[idx][0], frame_scores[idx][1]))
                    return selected_frames
                self.outputClusterSortByMean(opts, cluster_indices, distances, xyz_timeseries, trj_frames, random_num_rep_extractor)
            # Extract the 'worst', or highest mean distances from all other cluster members.
            # Potentially useful when trying to determine if a cluster definition is too generous.
            elif opts.extract == "worst":
                def last_num_rep_extractor(frame_scores):
                    selected_frames = []
                    num_reps = min(opts.num_reps, len(frame_scores))
                    for idx in range(1, num_reps+1):
                        selected_frames.append((-idx, frame_scores[-idx][0], frame_scores[-idx][1]))
                    return selected_frames
                self.outputClusterSortByMean(opts, cluster_indices, distances, xyz_timeseries, trj_frames, last_num_rep_extractor)
            # Extract frames for each trajectory present within a cluster.
            # Useful when a representative with the correct template/ligand/etc. is needed.
            elif opts.extract == "per_traj":
                # Determine the frame ranges for each trajectory
                traj_frame_ranges = []
                start_idx = 0
                for num_frames in trj_frames:
                    traj_frame_ranges.append(range(start_idx, start_idx+num_frames))
                    start_idx += num_frames
                traj_frame_ranges = tuple(traj_frame_ranges)
                def per_traj_num_rep_extractor(frame_scores):
                    selected_frames = []
                    # Take num_reps from each trajectory that has members
                    for traj_idx, traj_range in enumerate(traj_frame_ranges):
                        # Explicitly skip "trajectories" with only a single frame since those are typically used as dividers
                        if len(traj_range) <= 1: continue
                        frames_from_traj = [(cluster_rank, frame_score) for cluster_rank, frame_score in enumerate(frame_scores)
                                            if frame_score[1] in traj_range]
                        if frames_from_traj:
                            num_reps = min(opts.num_reps, len(frames_from_traj))
                            for idx in range(num_reps):
                                cluster_rank, frame_score = frames_from_traj[idx]
                                selected_frames.append((cluster_rank, frame_score[0], frame_score[1]))
                    return selected_frames
                self.outputClusterSortByMean(opts, cluster_indices, distances, xyz_timeseries, trj_frames, per_traj_num_rep_extractor)
            # Extract all frames for each cluster.
            elif opts.extract == "all":
                self.outputCompleteCluster(opts, cluster_indices, trj_frames)

    def _iterateFile(self, opts, fname):
        '''Iterate over a file of unknown format.'''
        try:
            for frame_ct, frame_asl_alist in self._iterateSingleTrajectory(opts, fname):
                yield frame_ct, frame_asl_alist
        #FIXME this is a terrible thing to do, SHAME!!!
        except:
            for frame_ct, frame_asl_alist in self._iterateStructures(opts, fname):
                yield frame_ct, frame_asl_alist

    def _iterateSingleTrajectory(self, opts, trj_fname):
        '''Return a generator yielding aligned structure objects and the matching ASL atom list.'''
        if opts.reference:
            ref_st, ref_asl_alist, ref_extracted_vals, ref_fitasl_alist = self.parseReference(opts)
        traj_fname = self.getPath(trj_fname)
        _, cms_model, trj = traj_util.read_cms_and_traj(trj_fname)
        print(f"{trj_fname} has {len(trj)} frames")
        fsys_ct = cms_model.fsys_ct.copy()
        fitasl, asl = opts.fitasl, opts.asl
        if opts.fitasl:
            trj_fitasl_alist = analyze.evaluate_asl(fsys_ct, fitasl)
        trj_asl_alist = analyze.evaluate_asl(fsys_ct, asl)
        for frame_idx, frame in enumerate(trj):
            # Update the ct positions to the traj positions
            # We have to do this because traj includes pseudo-atoms
            topo.update_ct(fsys_ct, cms_model, frame)
            if opts.fitasl and opts.reference:
                rms_fit = rmsd.superimpose(ref_st, ref_fitasl_alist,
                        fsys_ct, trj_fitasl_alist, move_which=rmsd.CT)
            yield fsys_ct, trj_asl_alist


    def _iterateStructures(self, opts, st_fname):
        '''Return a generator yielding aligned structure objects if they match the ASL.'''
        if opts.reference:
            ref_st, ref_asl_alist, ref_extracted_vals, ref_fitasl_alist = self.parseReference(opts)
        st_fname = self.getPath(st_fname)
        for st_idx, st in enumerate(structure.StructureReader(st_fname)):
            st_asl_alist = analyze.evaluate_asl(st, opts.asl)
            # Component CTs should be skipped
            if not st_asl_alist: continue
            if opts.reference:
                if len(st_asl_alist) != len(ref_asl_alist):
                    print(f"--asl mismatch for {st.property['s_m_title']}: expected {len(ref_asl_alist)} atoms, found {len(st_asl_alist)}")
                    continue
                if opts.fitasl:
                    st_fitasl_alist = analyze.evaluate_asl(st, opts.fitasl)
                    if len(ref_fitasl_alist) != len(st_fitasl_alist):
                        print(f"--fitasl mismatch for {st.property['s_m_title']}: expected {len(ref_fitasl_alist)} atoms, found {len(st_fitasl_alist)}")
                        continue
                    rms_fit = rmsd.superimpose(ref_st, ref_fitasl_alist,
                            st, st_fitasl_alist, move_which=rmsd.CT)
            yield st, st_asl_alist

    def parseFrames(self, opts):
        '''

        xyz_timeseries: MxNx3 matrix, M trajectory frames and N restraints
        '''
        num_atoms = None
        if opts.reference:
            ref_st, ref_asl_alist, ref_extracted_vals, ref_fitasl_alist = self.parseReference(opts)
            num_atoms = len(ref_asl_alist)

        # Extract all of the values
        compiled_extracted_vals = None
        trj_frames = []
        for trj in opts.trajectory:
            # Read in the trajectory/structure
            for iframe, (frame_ct, frame_asl_alist) in enumerate(self._iterateFile(opts, trj)):
                if num_atoms is None:
                    num_atoms = len(frame_asl_alist)
                assert len(frame_asl_alist) == num_atoms, f"Mismatch of atom numbers between structures matching {opts.asl} ({len(frame_asl_alist)} vs {num_atoms})"
                frame_asl_alist_0dx = [idx-1 for idx in frame_asl_alist]
                extracted_vals = frame_ct.getXYZ()[frame_asl_alist_0dx]
                extracted_vals = extracted_vals.reshape((1, num_atoms, 3))
                compiled_extracted_vals = extracted_vals if compiled_extracted_vals is None else np.concatenate((compiled_extracted_vals, extracted_vals), axis=0)
            print(f"{iframe+1:04d} frames in {trj}")
            trj_frames.append(iframe+1)

        # Compute the actual distance matrix 
        num_frames = sum(trj_frames)
        num_condensed_entries = (num_frames**2 - num_frames) // 2
        condensed_distance_matrix = np.zeros(num_condensed_entries)
        def square_to_condensed(i, j, n):
            '''From https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist'''
            assert i != j
            if i < j:
                i, j = j, i
            return n*j - j*(j+1)//2 + i - 1 - j
        for i in range(num_frames):
            for j in range(i+1, num_frames):
                atom_distances = np.sqrt(np.square(compiled_extracted_vals[i,:,:] - compiled_extracted_vals[j,:,:]).sum(axis=1))
                custom_dist = np.sum(1. / (1 + np.exp(-(atom_distances - opts.threshold)/opts.softness)))
                condensed_distance_matrix[square_to_condensed(i, j, num_frames)] = custom_dist
        # Save our hard work
        data_fname = opts.jobname + ".npz"
        to_store = {
            'condensed_distance_matrix': condensed_distance_matrix,
            'xyz_timeseries': compiled_extracted_vals,
            'trj_frames': trj_frames
        }
        if opts.reference:
            to_store['ref_vals'] = ref_extracted_vals
        else:
            ref_extracted_vals = None
        np.savez(data_fname, **to_store)
        self.addOutputFile(data_fname)
        return condensed_distance_matrix, ref_extracted_vals, compiled_extracted_vals, trj_frames

    def parseReference(self, opts):
        '''Get reference atoms vals from the reference file.'''
        with structure.StructureReader(self.getPath(opts.reference)) as st_reader:
            ref_st = next(st_reader)
        ref_asl_alist = analyze.evaluate_asl(ref_st, opts.asl)
        ref_asl_alist_0dx = [idx-1 for idx in ref_asl_alist]
        ref_extracted_vals = ref_st.getXYZ()[ref_asl_alist_0dx]
        if opts.fitasl:
            ref_fitasl_alist = analyze.evaluate_asl(ref_st, opts.fitasl)
        else:
            ref_fitasl_alist = None
        return ref_st, ref_asl_alist, np.array(ref_extracted_vals), ref_fitasl_alist

    def clusterData(self, opts, cdm):
        linkageMatrix = linkage(cdm, opts.method)
        distances = squareform(cdm)
        return linkageMatrix, distances

    def plotData(self, opts, linkageMatrix, distances, trj_frames):
        # Raw RMSD matrix
        pylab.figure()
        pylab.pcolormesh(distances, vmin=opts.vmin, vmax=opts.vmax)
        pylab.xlim(0, distances.shape[0])
        pylab.ylim(0, distances.shape[1])
        for idx in np.cumsum(trj_frames)[:-1]:
            pylab.axvline(idx, color='k', linestyle='--', zorder=1)
            pylab.axhline(idx, color='k', linestyle='--', zorder=1)
        pylab.colorbar()
        pylab.xlabel('Frame')
        pylab.ylabel('Frame')
        pylab.title('Chronological Distance')
        fig_fname = opts.jobname + '_chronological_distance.png'
        pylab.savefig(fig_fname)
        pylab.close()
        self.addOutputFile(fig_fname)
        # Dendrogram
        pylab.figure()
        #dendrogram(linkageMatrix, no_labels=True, count_sort='descendent')
        dendrogram(linkageMatrix, truncate_mode='lastp', p=24,
                leaf_rotation=90, leaf_font_size=12,
                show_contracted=True)
        pylab.axhline(y=opts.cutoff, linestyle='--', color='k')
        pylab.xlabel('Number of Sorted Frames')
        pylab.ylabel(f'{opts.method.capitalize()} distance')
        pylab.title(f'Dendrogram based on a {opts.method.capitalize()} linkage analysis')
        pylab.tight_layout()
        fig_fname = opts.jobname + f'_{opts.method}_dendrogram.png'
        pylab.savefig(fig_fname)
        pylab.close()
        self.addOutputFile(fig_fname)
        # Sorted RMSD matrix
        pylab.figure()
        sortedOrder = leaves_list(linkageMatrix)
        Xs, Ys = np.meshgrid(sortedOrder, sortedOrder)
        sortedRmsds = distances[Xs, Ys]
        pylab.pcolormesh(sortedRmsds, vmin=opts.vmin, vmax=opts.vmax)
        pylab.xlim(0, distances.shape[0])
        pylab.ylim(0, distances.shape[1])
        pylab.colorbar()
        fig_fname = opts.jobname + f'_{opts.method}_sorted_distance.png'
        pylab.savefig(fig_fname)
        pylab.close()
        self.addOutputFile(fig_fname)

    def outputClusterSortByMean(self, opts, cluster_indices, distances, xyz_timeseries, trj_frames, extractor):
        '''The extractor function determines how representative frames are selected from the cluster once it is ordered.
        
        frame_scores = ((relative_idx, frame_idx, mean_cluster_distance),)
        extractor(frame_scores) -> tuple((cluster_rank, relative_idx, frame_idx),)
        '''
        cluster_rep_frames = {}
        num_frames, num_atoms, _ = xyz_timeseries.shape
        if opts.reference:
            _, _, ref_atom_centers, _ = self.parseReference(opts)
        for cluster_idx in set(cluster_indices):
            frame_indices = np.where(cluster_indices == cluster_idx)[0]
            num_in_cluster = len(frame_indices)
            frame_scores = []
            for relative_idx, frame_idx in enumerate(frame_indices):
                mean_cluster_distance = np.mean(distances[frame_idx, frame_indices])
                frame_scores.append((relative_idx, frame_idx, mean_cluster_distance))
            frame_scores.sort(key=operator.itemgetter(2))
            rep_indices = extractor(frame_scores)
            for (rank, relative_idx, frame_idx) in rep_indices:
                # Boxplots vs Median
                atom_centers = xyz_timeseries[frame_idx, :, :]
                atom_distances = np.zeros((num_in_cluster, num_atoms))
                for atom_idx in range(num_atoms):
                    for other_relative_idx, other_frame_idx in enumerate(frame_indices):
                        atom_distances[other_relative_idx, atom_idx] = np.sqrt(np.square(xyz_timeseries[other_frame_idx, atom_idx, :] - atom_centers[atom_idx, :]).sum())
                fig = pylab.figure()
                result = pylab.boxplot(atom_distances, whis=opts.whisker)
                pylab.axhline(y=0, linestyle='--', color='b')
                pylab.ylabel('Cartesian Distance (A)')
                if opts.min:
                    pylab.ylim(ymin=opts.min)
                if opts.max:
                    pylab.ylim(ymax=opts.max)
                title = f"Individual Cartesian Distances for Cluster{cluster_idx:.0f} Rep{rank:d} Frame{frame_idx:d}"
                basename = '_'.join((opts.jobname,
                                     opts.method,
                                     f"cluster{cluster_idx:.0f}",
                                     f"size{num_in_cluster:04d}",
                                     f"rep{rank:02d}",
                                     f"frame{frame_idx:04d}",
                                     "cartesian"))
                fig_fname = basename + '_boxplot.png'
                pylab.title(title)
                pylab.tight_layout()
                pylab.savefig(fig_fname)
                pylab.close()
                self.addOutputFile(fig_fname)

                # Boxplots vs Reference
                if opts.reference:
                    atom_distances = np.zeros((num_in_cluster, num_atoms))
                    for atom_idx in range(num_atoms):
                        for other_relative_idx, other_frame_idx in enumerate(frame_indices):
                            atom_distances[other_relative_idx, atom_idx] = np.sqrt(np.square(xyz_timeseries[other_frame_idx, atom_idx, :] - ref_atom_centers[atom_idx, :]).sum())
                    fig = pylab.figure()
                    result = pylab.boxplot(atom_distances, whis=opts.whisker)
                    pylab.axhline(y=0, linestyle='--', color='b')
                    pylab.ylabel('Cartesian Distance (A)')
                    if opts.min:
                        pylab.ylim(ymin=opts.min)
                    if opts.max:
                        pylab.ylim(ymax=opts.max)
                    title = f"Individual Cartesian Distances for Cluster{cluster_idx:.0f} to Reference"
                    basename = '_'.join((opts.jobname,
                                         opts.method,
                                         f"cluster{cluster_idx:.0f}",
                                         f"size{num_in_cluster:04d}",
                                         "to_reference_cartesian"))
                    fig_fname = basename + '_boxplot.png'
                    pylab.title(title)
                    pylab.tight_layout()
                    pylab.savefig(fig_fname)
                    pylab.close()
                    self.addOutputFile(fig_fname)
                # Stacked pcolor
                min_val = opts.min if opts.min else np.min(atom_distances)
                max_val = opts.max if opts.max else np.max(atom_distances)
                bins = np.linspace(min_val, max_val, 20)
                Zs = np.zeros((num_atoms, len(bins)-1), dtype=float)
                for atom_idx in range(num_atoms):
                    hist, bin_edges = np.histogram(atom_distances[:, atom_idx], bins=bins)
                    Zs[atom_idx, :] = hist
                Xs, Ys = np.meshgrid(range(num_atoms+1), bins, indexing='ij')
                fig = pylab.figure()
                pylab.pcolor(Xs, Ys, Zs)
                pylab.colorbar()
                fig_fname = basename + '_pcolor.png'
                pylab.ylim(ymin=min_val)
                pylab.ylim(ymax=max_val)
                pylab.ylabel('Cartesian Distance (A)')
                pylab.title(title)
                pylab.tight_layout()
                pylab.savefig(fig_fname)
                pylab.close()
                self.addOutputFile(fig_fname)
                # Save the rest for later analysis
                cluster_rep_frames[frame_idx] = {'index': cluster_idx,
                                                 'num_in_cluster': num_in_cluster,
                                                 'rank': rank,
                                                 'boxplot': result,
                                                 'centers': atom_centers,
                                                }
                npz_fname = basename + '.npz'
                np.savez(npz_fname,
                        xyz_timeseries=xyz_timeseries[frame_indices, :, :],
                        cluster_rep=cluster_rep_frames[frame_idx],)
                self.addOutputFile(npz_fname)


        # Now lets go find our cluster_reps from the trajectory!
        # Find the traj indices
        trj_indices = np.concatenate(((0,), np.cumsum(trj_frames)))
        iframe = -1
        for trj, istart, num_frames_expected in zip(opts.trajectory, trj_indices, trj_frames):
            for iframe, (frame_ct, _) in enumerate(self._iterateFile(opts, trj), start=istart):
                if iframe not in cluster_rep_frames: continue
                solute_aids = analyze.evaluate_asl(frame_ct, "NOT (solvent OR ions)")
                cluster_rep_ct = frame_ct.extract(solute_aids, copy_props=True)
                # Copy and clean up the properties a bit
                cluster_rep = cluster_rep_frames[iframe]
                cluster_rep_title = opts.jobname + '_cluster{0:.0f}_size{1:04d}_rep{2:02d}_frame{3:04d}'.format(cluster_rep['index'], cluster_rep['num_in_cluster'], cluster_rep['rank'], iframe)
                cluster_rep_ct.property['s_m_title'] = cluster_rep_title
                cluster_rep_ct.property['s_m_original_cms_file'] = trj
                #FIXME
                # Add the microstate definitions
                #microstate_definition = self.createMicrostateDefinition(opts,
                #                                                        torsion_asls,
                #                                                        cluster_rep['centers'],
                #                                                        cluster_rep['boxplot'])
                #cluster_rep_ct.property['s_desmond_encoded_microstate'] = microstate_definition
                cluster_rep_frames[iframe]['ct'] = cluster_rep_ct
            num_frames = (iframe - istart) + 1
            if num_frames != num_frames_expected:
                raise RuntimeWarning(f"Issue processing {trj}, expected {num_frames_expected} frames but found {num_frames}")
        # Align the frames to the most representative frame if no reference provided
        if opts.fitasl and not opts.reference:
            most_populated_rep = sorted(sorted(cluster_rep_frames.values(), key=operator.itemgetter('num_in_cluster'), reverse=True),
                                        key=operator.itemgetter('rank'))[0]
            ref_st = most_populated_rep['ct']
            ref_fitasl_alist = analyze.evaluate_asl(ref_st, opts.fitasl)
            for frame in cluster_rep_frames.values():
                frame_ct = frame['ct']
                frame_fitasl_alist = analyze.evaluate_asl(frame_ct, opts.fitasl)
                rms_fit = rmsd.superimpose(ref_st, ref_fitasl_alist,
                        frame_ct, frame_fitasl_alist, move_which=rmsd.CT)
        # Finally export!
        for iframe, cluster_rep in cluster_rep_frames.items():
            cluster_rep_title = '_'.join((opts.jobname,
                                          opts.method,
                                          f"cluster{cluster_rep['index']:.0f}",
                                          f"size{cluster_rep['num_in_cluster']:04d}",
                                          f"rep{cluster_rep['rank']:02d}",
                                          f"frame{iframe:04d}"))
            cluster_rep_fname = cluster_rep_title + '.mae'
            with structure.StructureWriter(cluster_rep_fname) as writer:
                writer.append(cluster_rep['ct'])
            self.addOutputFile(cluster_rep_fname)

    def outputCompleteCluster(self, opts, cluster_indices, trj_frames):
        '''Create a file for each cluster containing all contained frames.
        '''
        # Construct the writers
        cluster_counts = Counter(cluster_indices)
        cluster_keys = tuple(map(operator.itemgetter(0),
                                 sorted(cluster_counts.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)))
        cluster_writers = {}
        for key in cluster_keys:
            cluster_fname = '_'.join((opts.jobname,
                                      opts.method,
                                      f"cluster{key:.0f}",
                                      f"size{cluster_counts[key]:04d}")) + '.mae'
            cluster_writers[key] = structure.StructureWriter(cluster_fname)
        # Iterate through the frames
        trj_indices = np.concatenate(((0,), np.cumsum(trj_frames)))
        iframe = -1
        for trj, istart, num_frames_expected in zip(opts.trajectory, trj_indices, trj_frames):
            for iframe, (frame_ct, _) in enumerate(self._iterateFile(opts, trj), start=istart):
                cluster_key = cluster_indices[iframe]
                solute_aids = analyze.evaluate_asl(frame_ct, "NOT (solvent OR ions)")
                solute_ct = frame_ct.extract(solute_aids, copy_props=True)
                # Copy and clean up the properties a bit
                solute_ct.title = f"{opts.jobname}_cluster{cluster_key:.0f}_size{cluster_counts[key]:04d}_frame{iframe:04d}"
                solute_ct.property['s_m_original_title'] = frame_ct.title
                solute_ct.property['s_m_original_cms_file'] = trj
                cluster_writer = cluster_writers[cluster_key]
                cluster_writer.append(solute_ct)
            num_frames = (iframe - istart) + 1
            if num_frames != num_frames_expected:
                raise RuntimeWarning(f"Issue processing {trj}, expected {num_frames_expected} frames but found {num_frames}")
        for writer in cluster_writers.values():
            writer.close()
            self.addOutputFile(writer.filename)

    def timeseriesClusters(self, opts, cluster_indices, distances, trj_frames):
        '''Plot the timeseries of cluster assignment.'''
        # Remap cluster indices to sequential to make plotting cleaner
        # Sort required since a 0-index is a valid identifier of non-clustered
        old_indices = tuple(sorted(set(cluster_indices)))
        remapped_indices = np.zeros(cluster_indices.shape)
        for new_idx, old_idx in enumerate(old_indices):
            remapped_indices[np.where(cluster_indices == old_idx)[0]] = new_idx
        pylab.figure()
        pylab.scatter(range(len(remapped_indices)), remapped_indices, c=remapped_indices)
        for idx in np.cumsum(trj_frames)[:-1]:
            pylab.axvline(idx, color='k', linestyle='--', zorder=1)
        pylab.xlabel('Frame')
        pylab.ylabel('Cluster')
        pylab.yticks(range(len(old_indices)), map(int, old_indices))
        fig_fname = opts.jobname + f'_{opts.method}_cluster_timeseries.png'
        pylab.savefig(fig_fname)
        pylab.close()
        self.addOutputFile(fig_fname)
        dat_fname = opts.jobname + f'_{opts.method}_cluster_timeseries.txt'
        np.savetxt(dat_fname, np.vstack((np.arange(len(cluster_indices)), cluster_indices)))
        self.addOutputFile(dat_fname)
           
    def tabulateClusterComposition(self, opts, cluster_indices, trj_frames):
        '''Make a CSV tabulation of cluster composition.'''
        # Find the centroid indices
        clusterComposition = {}
        # Get them in sorted order for convenience
        clusterCount = Counter(cluster_indices)
        clusterUniqueIndices = tuple(map(operator.itemgetter(0),
                                         sorted(clusterCount.items(),
                                                key=operator.itemgetter(1),
                                                reverse=True)))
        for idx in clusterUniqueIndices:
            clusterComposition[idx] = {}
            for trjIdx in range(len(opts.trajectory)):
                clusterComposition[idx][trjIdx] = 0
        # Find the traj indices
        trjIndices = np.concatenate(((0,), np.cumsum(trj_frames)))
        # Assign each frame to a cluster/traj
        for frameIdx, clusterIdx in enumerate(cluster_indices):
            composition = clusterComposition[clusterIdx]
            for trjIdx, (firstFrame, lastFrame) in enumerate(zip(trjIndices[:-1], trjIndices[1:])):
                if firstFrame <= frameIdx < lastFrame: break
            else:
                raise UserWarning("Should never reach this")
            composition[trjIdx] = composition.get(trjIdx, 0) + 1
        fname = opts.jobname + f'_{opts.method}_cluster_composition.csv'
        with open(fname, 'w', newline='') as fh:
            writer = csv.writer(fh)
            trj_labels = opts.label or [f'trj{trjIdx:02d}' for trjIdx in range(len(opts.trajectory))]
            writer.writerow(itertools.chain(('cluster_index',), trj_labels))
            for clusterIdx, trjComposition in clusterComposition.items():
                row = [trjComposition.get(trjIdx, 0) for trjIdx in range(len(opts.trajectory))]
                writer.writerow(itertools.chain((f'c{int(clusterIdx)}',), row))
        self.addOutputFile(fname)
        return clusterComposition

    # FIXME
    def createMicrostateDefinition(self, opts, torsion_asls, centers, result):
        '''Create a list of dictionaries defining the microstate by torsions.

        {'fbhw': [{label: string,
                   atom_asls: list of strings (4 ASLs)
                   phi0: floating point number, single precision
                   sigma: floating point number, single precision
                  },
                  ...
                 ]
        }
        '''
        definition = {}
        torsions = []
        for torsion_idx, (torsion_label, torsion_atoms) in enumerate(torsion_asls):
            lower_whisker = result['whiskers'][torsion_idx*2].get_ydata().min()
            upper_whisker = result['whiskers'][torsion_idx*2+1].get_ydata().max()
            sigma = (upper_whisker - lower_whisker) / 2.0
            mean = (upper_whisker + lower_whisker) / 2.0
            assert(sigma <= 180.0)
            center = angles.Angle(centers[torsion_idx]) + mean
            specific_torsion = {}
            specific_torsion['label'] = torsion_label
            specific_torsion['atom_asls'] = torsion_atoms
            specific_torsion['phi0'] = f'{center:.1f}'
            specific_torsion['sigma'] = f'{sigma:.1f}'
            torsions.append(specific_torsion)
        definition['fbhw'] = torsions
        encoded = base64.b64encode(json.dumps(definition).encode()).decode()
        return encoded

def linkage_up_to_cutoff(linkage_matrix, cutoff, min_size=1):
    '''Break a linkage_matrix up into clusters before some cutoff value.
    Clusters below a minimum size will be lumped together in a 0-cluster.'''
    num_frames = linkage_matrix.shape[0]+1
    clusters = {}
    # Set up initial frame-per-cluster
    for frame_idx in range(num_frames):
        clusters[frame_idx] = [frame_idx,]
    # Step through the linkage
    clusters_above_cutoff = set()
    for linkage_idx, (left_cluster, right_cluster, dist, _) in enumerate(linkage_matrix):
        new_cluster_idx = linkage_idx + num_frames
        if ((dist >= cutoff) or
            (left_cluster in clusters_above_cutoff) or
            (right_cluster in clusters_above_cutoff)):
            # Skip all leaves once a branch is greater than the cutoff
            clusters_above_cutoff.add(new_cluster_idx)
        else:
            clusters[new_cluster_idx] = clusters.pop(int(left_cluster))
            clusters[new_cluster_idx].extend(clusters.pop(int(right_cluster)))
    cluster_indices = np.zeros(num_frames, dtype=int)
    for cluster_idx, frame_indices in clusters.items():
        if len(frame_indices) < min_size: continue
        for frame_idx in frame_indices:
            cluster_indices[frame_idx] = cluster_idx
    return cluster_indices, len(clusters_above_cutoff)


if __name__ == "__main__":
    CartesianCustomDistanceClusteringToFBHW(sys.argv[0]).launch(sys.argv[1:])
