import argparse
import enum
import os.path
import shutil
from collections import namedtuple
from typing import Tuple, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from Video2mesh.io import VTMDataset
from Video2mesh.options import COLMAPOptions
from Video2mesh.utils import tqdm_imap, log


class FrameSamplingMode(enum.Enum):
    # All unique pairs of frames
    EXHAUSTIVE = enum.auto()
    # All consecutive pairs
    CONSECUTIVE = enum.auto()
    # Consecutive pairs + increasingly distant pairs.
    HIERARCHICAL = enum.auto()


FramePair = Tuple[int, int]
FramePairs = List[FramePair]

"""Encapsulates the frame index, the coordinates of the image features and the depth at those points."""
FeatureData = namedtuple('FeatureData', ['index', 'points', 'depth'])

"""FeatureData for a pair of frames."""
FeatureSet = namedtuple('FeatureSet', ['frame_i', 'frame_j'])


class FeatureExtractor:
    """Extracts correspondences between image pairs and the depth at those correspondences."""

    def __init__(self, dataset: VTMDataset, frame_pairs: FramePairs, min_features: int, ignore_dynamic_objects: bool):
        """

        :param dataset: The RGB-D dataset to use.
        :param frame_pairs: The pairs of frames to extract matching image features from.
        :param min_features: The minimum number of matched features required per frame pair.
        :param ignore_dynamic_objects: Whether to ignore dynamic objects when extracting image features.
        """
        self.dataset = dataset
        self.frame_pairs = frame_pairs
        self.min_features = min_features
        self.ignore_dynamic_objects = ignore_dynamic_objects

        self.debug_path: Optional[str] = None

        self.frames: Optional[List[np.ndarray]] = None
        self.depth_maps: Optional[List[np.ndarray]] = None
        self.masks: Optional[List[np.ndarray]] = None

        self.sift = cv2.SIFT_create(nfeatures=512)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_feature_points(self) -> FeatureSet:
        """
        Extract a set of features from the dataset.

        :return: A FeatureSet contained the paired feature data (frame index, match coordinates, depth at match
        coordinates) for each frame pair.
        """
        log(f"Extracting image feature matches...")
        self.setup_debug_folder()
        self.frames, self.depth_maps, self.masks = self.get_frame_data()

        index_i = []
        points_i = []
        depth_i = []
        index_j = []
        points_j = []
        depth_j = []
        num_good_frame_pairs = 0

        log(f"Extracting matching image feature info...")

        for feature_set in tqdm_imap(self.get_image_features, self.frame_pairs):
            if feature_set is None:
                continue

            num_points = len(feature_set.frame_i.points)

            index_i += [feature_set.frame_i.index] * num_points
            points_i += feature_set.frame_i.points
            depth_i += feature_set.frame_i.depth

            index_j += [feature_set.frame_j.index] * num_points
            points_j += feature_set.frame_j.points
            depth_j += feature_set.frame_j.depth

            num_good_frame_pairs += 1

        coverage = len(set(index_i + index_j)) / self.dataset.num_frames
        log(f"Found {num_good_frame_pairs} good frame pairs ({num_good_frame_pairs}/{len(self.frame_pairs)})")
        log(f"Frame pairs cover {100 * coverage:.2f}% of the frames.")

        return FeatureSet(FeatureData(index_i, points_i, depth_i), FeatureData(index_j, points_j, depth_j))

    def setup_debug_folder(self):
        """
        Create a folder to save visualisations of the matched image features to.
        """
        feature_point_debug_path = os.path.join(self.dataset.base_path, 'image_feature_matches')

        if os.path.isdir(feature_point_debug_path):
            shutil.rmtree(feature_point_debug_path)

        os.makedirs(feature_point_debug_path)

        self.debug_path = feature_point_debug_path

    def get_frame_data(self) -> Tuple[list, list, Optional[list]]:
        """
        Load the video frames, depth maps and masks from the dataset into memory.
        :return: A 3-tuple of the RGB frames, depth maps and masks.
        """
        log(f"Loading frames...")

        def load_frame(index):
            return cv2.cvtColor(self.dataset.rgb_dataset[index], cv2.COLOR_RGB2GRAY)

        frames = tqdm_imap(load_frame, range(self.dataset.num_frames))

        log(f"Loading depth maps...")
        depth_maps = tqdm_imap(self.dataset.depth_dataset.__getitem__, range(self.dataset.num_frames))

        if self.ignore_dynamic_objects:
            log(f"Loading masks...")

            def get_mask(index):
                mask = self.dataset.mask_dataset[index]
                mask[mask > 0] = 255
                # Flip mask so that dynamic objects are set to 0, i.e. tell the SIFT detector to ignore dynamic objects.
                mask = ~mask

                return mask

            masks = tqdm_imap(get_mask, range(self.dataset.num_frames))
        else:
            masks = None

        return frames, depth_maps, masks

    def get_image_features(self, frame_pair: FramePair) -> Optional[FeatureSet]:
        """
        Extract the image features from a frame pair.
        :param frame_pair: The indices of the frames.
        :return: The paired feature data (frame index, match coordinates, depth at match coordinates),
            None if there is an insufficient number of matches.
        """
        i, j = frame_pair

        key_points_i, descriptors_i = self.get_key_points_and_descriptors(i)
        key_points_j, descriptors_j = self.get_key_points_and_descriptors(j)

        if min(len(key_points_i), len(key_points_j)) < self.min_features:
            return None

        matches = self.matcher.knnMatch(descriptors_i, descriptors_j, k=2)

        points_i, points_j, depth_i, depth_j, matches_mask = \
            self.filter_matches(i, j, key_points_i, key_points_j, matches)

        if self.debug_path:
            self.save_matches_visualisation(i, j, key_points_i, key_points_j, matches, matches_mask)

        if len(points_i) < self.min_features:
            return None

        depth_i, depth_j, points_i, points_j = \
            self.filter_matches_ransac(points_i, points_j, depth_i, depth_j, matches_mask)

        if len(points_i) < self.min_features:
            return None

        return FeatureSet(FeatureData(i, points_i, depth_i), FeatureData(j, points_j, depth_j))

    def get_key_points_and_descriptors(self, index) -> Tuple[tuple, tuple]:
        """
        Get the SIFT key points and descriptors for a frame.
        :param index: The index of the frame to process.
        :return: The key points and SIFT descriptors
        """
        if self.ignore_dynamic_objects:
            mask = self.masks[index]
        else:
            mask = None

        key_points, descriptors = self.sift.detectAndCompute(self.frames[index], mask)

        return key_points, descriptors

    def filter_matches(self, i, j, key_points_i, key_points_j, matches):
        """
        Filter candidate matches with Lowe's ratio test.

        :param i: The index of the first frame.
        :param j: The index of the second frame.
        :param key_points_i: The key points for frame i.
        :param key_points_j: The key points for frame j.
        :param matches: The candidate matches from the KNN matcher.
        :return: The filtered points of each frame, depth for these points, and the mask of accepted matches.
        """
        matches_mask = [[0, 0] for _ in range(len(matches))]
        points_i = []
        points_j = []
        depth_i = []
        depth_j = []

        for k, (m, n) in enumerate(matches):
            if m.distance > 0.7 * n.distance:
                continue

            point_i = key_points_i[m.queryIdx].pt
            the_depth_i = self.depth_maps[i][round(point_i[1]), round(point_i[0])]

            point_j = key_points_j[m.trainIdx].pt
            the_depth_j = self.depth_maps[j][round(point_j[1]), round(point_j[0])]

            if the_depth_i == 0.0 or the_depth_j == 0.0:
                continue

            # Mark match as good match for viz.
            matches_mask[k][0] = 1

            points_i.append(point_i)
            points_j.append(point_j)

            depth_i.append(the_depth_i)
            depth_j.append(the_depth_j)

        return points_i, points_j, depth_i, depth_j, matches_mask

    def filter_matches_ransac(self, points_i, points_j, depth_i, depth_j, matches_mask):
        """
        Filter candidate matches with RANSAC.

        :param points_i: The points of the first frames of the frame pairs.
        :param points_j: The points of the second frames of the frame pairs.
        :param depth_i: The depth of the points of the first frames of the frame pairs.
        :param depth_j: The depth of the points of the second frames of the frame pairs.
        :param matches_mask: The mask of accepted and rejected candidate matches.
        :return: 4-tuple of the filtered points of each frame and the depth for these points.
        """
        points_i = np.asarray(points_i)
        points_j = np.asarray(points_j)
        depth_i = np.asarray(depth_i)
        depth_j = np.asarray(depth_j)

        _, mask = cv2.findHomography(points_i, points_j, cv2.RANSAC)

        is_inlier = mask.flatten() > 0
        # Need to undo the matchesMask for good matches that were not inliers to ensure the viz is correct.
        matches_mask = np.asarray(matches_mask)
        matches_mask[np.argwhere((matches_mask == [1, 0]).all(axis=1))[~is_inlier]] = [0, 0]

        points_i = points_i[is_inlier].tolist()
        points_j = points_j[is_inlier].tolist()
        depth_i = depth_i[is_inlier].tolist()
        depth_j = depth_j[is_inlier].tolist()

        return points_i, points_j, depth_i, depth_j

    def save_matches_visualisation(self, i, j, key_points_i, key_points_j, matches, matches_mask):
        """
        Save a visualisation of the accepted and rejected matches.
        :param i: The index of the first frame in the pair.
        :param j: The index of the second frame in the pair.
        :param key_points_i: The matched points in the first frame.
        :param key_points_j: The matched points in the second frame.
        :param matches: The match data from the KNN matcher.
        :param matches_mask: The mask of accepted and rejected candidate matches.
        """
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        kp_matches_viz = cv2.drawMatchesKnn(self.frames[i], key_points_i, self.frames[j], key_points_j, matches, None,
                                            **draw_params)
        kp_matches_viz = cv2.cvtColor(kp_matches_viz, cv2.COLOR_BGR2RGB)
        plt.imsave(os.path.join(self.debug_path, f"{i:06d}-{j:06d}.jpg"), kp_matches_viz)


class PoseOptimiser:
    def __init__(self, dataset: VTMDataset):
        self.dataset = dataset

    def run(self, frame_sampling=FrameSamplingMode.HIERARCHICAL, mask_features=True, min_features_per_frame=20):
        frame_pairs = self.sample_frame_pairs(frame_sampling)
        feature_set = self.extract_feature_points(frame_pairs, min_features_per_frame, mask_features)
        fixed_parameters = self.get_fixed_parameters(feature_set)
        self.get_optimisation_parameters()
        self.optimise_pose_data()

        return self.dataset.camera_matrix, self.dataset.camera_trajectory

    def sample_frame_pairs(self, frame_sampling_mode: FrameSamplingMode) -> FramePairs:
        """
        Select frame pairs for image feature extraction.

        :param frame_sampling_mode: The strategy to use for sampling frame pairs.
        :return: A list of pairs of frame indices.
        """
        frame_pairs = []

        if frame_sampling_mode == FrameSamplingMode.EXHAUSTIVE:
            for i in range(self.dataset.num_frames):
                for j in range(i + 1, self.dataset.num_frames):
                    frame_pairs.append((i, j))

        elif frame_sampling_mode == FrameSamplingMode.CONSECUTIVE:
            for i in range(self.dataset.num_frames - 1):
                frame_pairs.append((i, i + 1))

        elif frame_sampling_mode == FrameSamplingMode.HIERARCHICAL:
            # Adapted from https://github.com/facebookresearch/consistent_depth/blob/e2c9b724d3221aa7c0bf89aa9449ae33b418d943/utils/frame_sampling.py#L78
            max_level = int(np.floor(np.log2(self.dataset.num_frames - 1)))

            for level in range(max_level + 1):
                step = 1 << level

                for start in range(0, self.dataset.num_frames, step):
                    end = start + step

                    if end >= self.dataset.num_frames:
                        continue

                    frame_pairs.append((start, end))
        else:
            raise RuntimeError(f"Unsupported frame sampling mode: {frame_sampling_mode}.")

        return frame_pairs

    def extract_feature_points(self, frame_pairs: FramePairs, min_features: int, mask_features: bool):
        feature_extractor = FeatureExtractor(self.dataset, frame_pairs, min_features, mask_features)

        return feature_extractor.extract_feature_points()

    def get_fixed_parameters(self, feature_set: FeatureSet):
        # TODO: write the rest of this function :)
        pass

    def get_optimisation_parameters(self):
        pass

    def optimise_pose_data(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to the VTM formatted dataset.')
    args = parser.parse_args()

    if not VTMDataset.is_valid_folder_structure(args.dataset_path):
        raise RuntimeError(f"The path {args.dataset_path} does not point to a valid dataset.")

    dataset = VTMDataset(args.dataset_path, overwrite_ok=False)

    dataset.use_estimated_camera_parameters(COLMAPOptions())
    dataset.create_or_find_masks()

    optimiser = PoseOptimiser(dataset)
    camera_matrix, camera_trajectory = optimiser.run()


if __name__ == '__main__':
    main()
