import argparse
import enum
import os.path
import shutil
import warnings
from os.path import join as pjoin
from typing import Tuple, List, Optional, Iterable, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
from tqdm import tqdm

from video2mesh.fusion import tsdf_fusion
from video2mesh.geometry import Quaternion, invert_trajectory, subtract_pose, add_pose, get_identity_pose, \
    point_cloud_from_depth
from video2mesh.io import VTMDataset
from video2mesh.options import StaticMeshOptions, MeshReconstructionMethod
from video2mesh.utils import tqdm_imap, log, temp_seed, Domain, check_domain


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    :param tensor: The tensor to convert.
    :return: The NumPy array.
    """
    return tensor.detach().cpu().numpy().copy()


def clone_param(param: torch.nn.Parameter) -> torch.Tensor:
    """
    Convert a PyTorch Parameter to a Tensor and detach it from the computation graph.

    :param param: The Parameter object to clone.
    :return: The copied Parameter object as a Tensor.
    """
    return param.detach().clone()


# noinspection PyArgumentList
class FrameSamplingMode(enum.Enum):
    """Method for sampling frame pairs from a video sequence."""

    # All unique pairs of frames, e.g. [(0, 1), (0, 1), ..., (0, n - 1), (1, 2), (1, 3), ..., (n - 2, n - 1)]
    Exhaustive = enum.auto()
    # All consecutive pairs, e.g. [(0, 1), (1, 2), (2, 3)]
    Consecutive = enum.auto()
    # Consecutive pairs that do not overlap, e.g. [(0, 1), (2, 3), (4, 5)]
    ConsecutiveNoOverlap = enum.auto()
    # Consecutive pairs that do not overlap and start at 1 (compare CONSECUTIVE_NO_OVERLAP),
    # e.g. [(1, 2), (3, 4), (5, 6)]
    ConsecutiveNoOverlapOffset = enum.auto()
    # Consecutive pairs + increasingly distant pairs, e.g. [(0, 1), (0, 2), (0, 4), ..., (1, 2)]
    Hierarchical = enum.auto()


"""A pair of frame indices."""
FramePair = Tuple[int, int]

"""A list of pairs of frame indices."""
FramePairs = List[FramePair]


class FeatureData(torch.nn.Module):
    """Encapsulates the frame index, the coordinates of the image features and the depth at those points."""

    def __init__(self, index=torch.empty(0, dtype=torch.long), points=torch.empty(0, 2, dtype=torch.float32),
                 depth=torch.empty(0, dtype=torch.float32)):
        """
        :param index: The frame indices of the points and their corresponding depth values.
        :param points: The (N, 2) array of 2D image coordinates of the matched image features.
        :param depth: The list of depth values for the matched points.
        """
        super().__init__()

        self.index = torch.nn.Parameter(index.to(torch.long), requires_grad=False)
        self.points = torch.nn.Parameter(points.to(torch.float32), requires_grad=False)
        self.depth = torch.nn.Parameter(depth.to(torch.float32), requires_grad=False)

    def __len__(self):
        return len(self.index)

    def sample_at(self, frame_indices: Union[torch.LongTensor, torch.BoolTensor]) -> 'FeatureData':
        """
        Get a copy of the FeatureDataTorch object at the given frame indices.
        :param frame_indices: A 1-dimensional array of either frame indices, or a boolean mask.
        :return: The copy of the FeatureDataTorch object.
        """
        return FeatureData(
            self.index[frame_indices].clone(),
            self.points[frame_indices].clone(),
            self.depth[frame_indices].clone()
        )


class FeatureSet(torch.nn.Module):
    """The parameters used in the optimisation process that do not change, e.g. matched feature points, depth."""

    def __init__(self, camera_matrix=torch.eye(3, dtype=torch.float32),
                 frame_i=FeatureData(), frame_j=FeatureData()):
        """
        :param camera_matrix: The (3, 3) camera intrinsics matrix.
        :param frame_i: The feature data for the left frames of the frame pairs.
        :param frame_j: The feature data for the right frames of the frame pairs.
        """
        super().__init__()

        self.camera_matrix = torch.nn.Parameter(camera_matrix, requires_grad=False)

        self.frame_i = frame_i
        self.frame_j = frame_j

    @property
    def device(self):
        """Get the device that the feature data is on (generally, either 'cpu' or 'cuda')."""
        return self.camera_matrix.device

    def __len__(self) -> int:
        """Get the number of correspondences across all frame pairs in the feature set."""
        return len(self.frame_i)

    @classmethod
    def load(cls, f) -> 'FeatureSet':
        """
        Load a FeatureSet from disk.
        :param f: Either the file object or path to the saved state_dict.
        :return: A new FeatureSet object using the loaded values.
        """
        state_dict = torch.load(f)

        camera_matrix = state_dict['camera_matrix']

        frame_i = FeatureData(
            index=state_dict['frame_i.index'],
            points=state_dict['frame_i.points'],
            depth=state_dict['frame_i.depth'],
        )

        frame_j = FeatureData(
            index=state_dict['frame_j.index'],
            points=state_dict['frame_j.points'],
            depth=state_dict['frame_j.depth'],
        )

        return cls(camera_matrix, frame_i, frame_j)

    def save(self, f):
        """
        Save the FeatureSet (more precisely it's state_dict) to disk.
        :param f: The file path or object to write to.
        """
        torch.save(self.state_dict(), f)

    def sample_at(self, frame_indices: Iterable[int]) -> 'FeatureSet':
        """
        Sample the fixed parameters at the given frames.
        :param frame_indices: The frames to include.
        :return: The FixedParameters object that only contains the data for the specified frames.
        """
        frame_set = set(frame_indices)
        device = self.device

        def get_matching_indices_mask(feature_data: FeatureData) -> torch.BoolTensor:
            """
            Get a binary mask of the FeatureData object where True indicates the row is a frame in `frame_indices`.
            :param feature_data: The FeatureData object to check.
            :return: A binary mask with a boolean value for each correspondence in FeatureData.
            """
            matches_mask = torch.zeros(len(feature_data), dtype=torch.bool, device=device)

            for index in frame_set:
                matches_mask |= feature_data.index == index

            return matches_mask

        matching_indices_mask = get_matching_indices_mask(self.frame_i) & get_matching_indices_mask(self.frame_j)

        frame_i = self.frame_i.sample_at(matching_indices_mask)
        frame_j = self.frame_j.sample_at(matching_indices_mask)

        return FeatureSet(self.camera_matrix.clone(), frame_i, frame_j)

    def subset_from(self, frame_pairs: FramePairs) -> 'FeatureSet':
        """
        An alternative form of `.sample_at(...)` that instead samples the FeatureSet object by frame pairs.
        :param frame_pairs: The frame pairs to keep.
        :return: A copy of this FeatureSet object that only contains the data for the given frame pairs.
        """
        indices = torch.vstack((self.frame_i.index, self.frame_j.index)).T
        indices = indices.cpu()
        mask = torch.zeros(len(self), dtype=torch.bool)
        frame_pairs = torch.from_numpy(np.asarray(frame_pairs))

        for frame_pair in frame_pairs:
            # noinspection PyTypeChecker
            mask |= torch.all(indices == frame_pair, dim=1)

        frame_i = self.frame_i.sample_at(mask)
        frame_j = self.frame_j.sample_at(mask)

        return FeatureSet(self.camera_matrix.clone(), frame_i, frame_j)


class FeatureExtractionOptions:
    """Options for the `FeatureExtractor` class."""

    def __init__(self, ignore_dynamic_objects=True, min_features=20, max_features: Optional[int] = 2048):
        """
        :param min_features: The minimum number of matched features required per frame pair.
        :param max_features: (optional) The maximum number of features to keep per frame pair. Setting this to None will
            mean that all features will be kept. This parameter affects two things:
            1) runtime - lower values = faster;
            and 2) number of matched features per frame pair - higher values generally means more matches per
            frame pair and a higher chance that all frames will be covered.
        :param ignore_dynamic_objects: Whether to ignore dynamic objects when extracting image features.
        """

        if not isinstance(min_features, int) or min_features < 5:
            raise ValueError(f"`min_features` must be a positive integer that is at least 5, but got {min_features}.")

        if max_features is not None and (not isinstance(max_features, int) or max_features <= min_features):
            raise ValueError(f"`max_features` must be a positive integer greater than `min_features` ({min_features}), "
                             f"but got {max_features}.")

        if min_features < 20:
            warnings.warn(f"`min_features` was set to {min_features}, however it is recommended to set `min_features` "
                          f"to at least 20. Anything lower than 20 generally leads to a low SNR and bad results.")

        if max_features is not None and max_features < 2 * min_features:
            warnings.warn(f"`max_features` was set to {max_features}, however it is recommended to set `max_features` "
                          f"to at least 2 * `min_features` ({2 * min_features}). This is to increase the likelihood "
                          f"that enough features will remain after filtering to meet the requirement of having "
                          f"at least `min_features` features.")

        self.ignore_dynamic_objects = ignore_dynamic_objects
        self.min_features = min_features
        self.max_features = max_features

    def __repr__(self):
        return f"{self.__class__.__name__}(ignore_dynamic_objects={self.ignore_dynamic_objects}, " \
               f"min_features={self.min_features}, " \
               f"max_features={self.max_features})"


class FeatureExtractor:
    """Extracts correspondences between image pairs and the depth at those correspondences."""

    def __init__(self, dataset: VTMDataset, frame_pairs: FramePairs,
                 feature_extraction_options=FeatureExtractionOptions(),
                 debug_path: Optional[str] = None):
        """
        :param dataset: The RGB-D dataset to extract matching image features from.
        :param frame_pairs: The pairs of frames to match image features.
        :param feature_extraction_options: Options for controlling certain aspects of the feature extraction process.
        :param debug_path: The path to save debug outputs and reuslts to.
            Setting this to `None` disables debug output and results caching.
        """

        self.dataset = dataset
        self.frame_pairs = frame_pairs
        self.options = feature_extraction_options

        self.debug_path: Optional[str] = debug_path
        self.match_viz_path: Optional[str] = None
        self.frame_pairs_path: Optional[str] = None
        self.feature_set_path: Optional[str] = None

        self.frames: Optional[List[np.ndarray]] = None
        self.depth_maps: Optional[List[np.ndarray]] = None
        self.masks: Optional[List[np.ndarray]] = None

        self.sift = cv2.SIFT_create(nfeatures=self.max_features)

        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    @property
    def min_features(self) -> int:
        """The minimum number of matches (after filtering) required for a frame pair to be kept."""
        return self.options.min_features

    @property
    def max_features(self) -> int:
        """The maximum number of SIFT descriptors to extract per frame."""
        return self.options.max_features

    @property
    def ignore_dynamic_objects(self):
        """Whether to ignore dynamic (i.e. moving) objects when extracting SIFT descriptors."""
        return self.options.ignore_dynamic_objects

    def extract_feature_points(self) -> FeatureSet:
        """
        Extract a set of features from the dataset.

        :return: A FeatureSet contained the paired feature data (frame index, match coordinates, depth at match
        coordinates) for each frame pair.
        """
        log(f"Extracting image feature matches...")
        self._setup_folders()

        if self.feature_set_path and os.path.isfile(self.feature_set_path):
            log(f"Found cached feature set at: {self.feature_set_path}")
            return FeatureSet.load(self.feature_set_path)

        self.frames, self.depth_maps, self.masks = self._get_frame_data()

        index_i = torch.empty(0)
        points_i = torch.empty(0, 2)
        depth_i = torch.empty(0)
        index_j = torch.empty(0)
        points_j = torch.empty(0, 2)
        depth_j = torch.empty(0)
        num_good_frame_pairs = 0

        log(f"Extracting matching image feature info...")

        for feature_set in tqdm_imap(self._get_image_features, self.frame_pairs):
            if feature_set is None:
                continue

            index_i = torch.hstack((index_i, feature_set.frame_i.index))
            points_i = torch.vstack((points_i, feature_set.frame_i.points))
            depth_i = torch.hstack((depth_i, feature_set.frame_i.depth))

            index_j = torch.hstack((index_j, feature_set.frame_j.index))
            points_j = torch.vstack((points_j, feature_set.frame_j.points))
            depth_j = torch.hstack((depth_j, feature_set.frame_j.depth))

            num_good_frame_pairs += 1

        self._print_results_stats(index_i, index_j, num_good_frame_pairs)

        full_feature_set = FeatureSet(torch.from_numpy(self.dataset.camera_matrix.copy()),
                                      FeatureData(index_i, points_i, depth_i),
                                      FeatureData(index_j, points_j, depth_j))

        if self.feature_set_path:
            full_feature_set.save(self.feature_set_path)

        return full_feature_set

    def _setup_folders(self):
        """
        Create a folder to save debug output.
        """
        if self.debug_path is None:
            return

        self.match_viz_path = pjoin(self.debug_path, 'match_viz')
        self.frame_pairs_path = pjoin(self.debug_path, 'frame_pairs.txt')
        self.feature_set_path = pjoin(self.debug_path, 'feature_set.pth')

        os.makedirs(self.debug_path, exist_ok=True)

        clear_cache = True

        if os.path.isfile(self.frame_pairs_path):
            cached_frame_pairs = np.loadtxt(self.frame_pairs_path)
            same_frame_pairs = len(cached_frame_pairs) == len(self.frame_pairs)
            same_frame_pairs = same_frame_pairs and np.all(cached_frame_pairs == self.frame_pairs)

            if same_frame_pairs:
                clear_cache = False

        if clear_cache:
            if os.path.isdir(self.match_viz_path):
                shutil.rmtree(self.match_viz_path)

            if os.path.isfile(self.frame_pairs_path):
                os.remove(self.frame_pairs_path)

            if os.path.isfile(self.feature_set_path):
                os.remove(self.feature_set_path)

            os.makedirs(self.match_viz_path)
            # noinspection PyTypeChecker
            np.savetxt(self.frame_pairs_path, self.frame_pairs)

    def _get_frame_data(self) -> Tuple[list, list, Optional[list]]:
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
                """Helper function to load a mask at a given index and perform some light processing."""
                mask = self.dataset.mask_dataset[index]
                mask[mask > 0] = 255
                # Flip mask so that dynamic objects are set to 0, i.e. tell the SIFT detector to ignore dynamic objects.
                mask = ~mask

                return mask

            masks = tqdm_imap(get_mask, range(self.dataset.num_frames))
        else:
            masks = None

        return frames, depth_maps, masks

    def _get_image_features(self, frame_pair: FramePair) -> Optional[FeatureSet]:
        """
        Extract the image features from a frame pair.
        :param frame_pair: The indices of the frames.
        :return: The paired feature data (frame index, match coordinates, depth at match coordinates),
            None if there is an insufficient number of matches.
        """
        i, j = frame_pair

        key_points_i, descriptors_i = self._get_key_points_and_descriptors(i)
        key_points_j, descriptors_j = self._get_key_points_and_descriptors(j)

        if min(len(key_points_i), len(key_points_j)) < self.min_features:
            return None

        matches = self.matcher.knnMatch(descriptors_i, descriptors_j, k=2)

        points_i, points_j, depth_i, depth_j, matches_mask = \
            self._filter_matches(i, j, key_points_i, key_points_j, matches)

        if len(points_i) < self.min_features:
            self._save_matches_visualisation(i, j, key_points_i, key_points_j, matches, matches_mask,
                                             frame_accepted=False)
            return None

        points_i, points_j, depth_i, depth_j, matches_mask = \
            self._filter_matches_ransac(points_i, points_j, depth_i, depth_j, matches_mask)

        if len(points_i) < self.min_features:
            self._save_matches_visualisation(i, j, key_points_i, key_points_j, matches, matches_mask,
                                             frame_accepted=False)
            return None

        self._save_matches_visualisation(i, j, key_points_i, key_points_j, matches, matches_mask, frame_accepted=True)

        return FeatureSet(
            camera_matrix=torch.from_numpy(self.dataset.camera_matrix.copy()),
            frame_i=FeatureData(
                index=torch.tensor([i] * len(points_i), dtype=torch.long),
                points=torch.tensor(points_i, dtype=torch.float32),
                depth=torch.tensor(depth_i, dtype=torch.float32)
            ),
            frame_j=FeatureData(
                index=torch.tensor([j] * len(points_j), dtype=torch.long),
                points=torch.tensor(points_j, dtype=torch.float32),
                depth=torch.tensor(depth_j, dtype=torch.float32)
            )
        )

    def _get_key_points_and_descriptors(self, index) -> Tuple[tuple, tuple]:
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

    def _filter_matches(self, i, j, key_points_i, key_points_j, matches):
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
            # Lowe's ratio test.
            if m.distance > 0.7 * n.distance:
                continue

            point_i = key_points_i[m.queryIdx].pt
            the_depth_i = self.depth_maps[i][round(point_i[1]), round(point_i[0])]

            point_j = key_points_j[m.trainIdx].pt
            the_depth_j = self.depth_maps[j][round(point_j[1]), round(point_j[0])]

            if the_depth_i == 0.0 or the_depth_j == 0.0:
                continue

            # Mark the match as good match for viz.
            matches_mask[k][0] = 1

            points_i.append(point_i)
            points_j.append(point_j)

            depth_i.append(the_depth_i)
            depth_j.append(the_depth_j)

        return points_i, points_j, depth_i, depth_j, matches_mask

    @staticmethod
    def _filter_matches_ransac(points_i, points_j, depth_i, depth_j, matches_mask) -> \
            Tuple[list, list, list, list, list]:
        """
        Filter candidate matches with RANSAC.

        :param points_i: The points of the first frames of the frame pairs.
        :param points_j: The points of the second frames of the frame pairs.
        :param depth_i: The depth of the points of the first frames of the frame pairs.
        :param depth_j: The depth of the points of the second frames of the frame pairs.
        :param matches_mask: The mask of accepted and rejected candidate matches.
        :return: 5-tuple of the filtered points of each frame and the depth for these points, and
            the updated matches mask.
        """
        points_i = np.asarray(points_i)
        points_j = np.asarray(points_j)
        depth_i = np.asarray(depth_i)
        depth_j = np.asarray(depth_j)

        _, mask = cv2.findHomography(points_i, points_j, cv2.USAC_MAGSAC)

        is_inlier = mask.flatten() > 0
        is_outlier = ~is_inlier
        # Need to undo the matchesMask for good matches that were found to be outliers to ensure the viz is correct.
        matches_mask = np.asarray(matches_mask)
        accepted_match_indices = np.argwhere((matches_mask == [1, 0]).all(axis=1))
        matches_mask[accepted_match_indices[is_outlier]] = [0, 0]
        matches_mask = matches_mask.tolist()

        points_i = points_i[is_inlier].tolist()
        points_j = points_j[is_inlier].tolist()
        depth_i = depth_i[is_inlier].tolist()
        depth_j = depth_j[is_inlier].tolist()

        # noinspection PyTypeChecker
        return points_i, points_j, depth_i, depth_j, matches_mask

    def _save_matches_visualisation(self, i, j, key_points_i, key_points_j, matches, matches_mask, frame_accepted):
        """
        Save a visualisation of the accepted and rejected matches.

        :param i: The index of the first frame in the pair.
        :param j: The index of the second frame in the pair.
        :param key_points_i: The matched points in the first frame.
        :param key_points_j: The matched points in the second frame.
        :param matches: The match data from the KNN matcher.
        :param matches_mask: The mask of accepted and rejected candidate matches.
        :param frame_accepted: Whether the frame was accepted (i.e. had number matches > `self.min_features`).
        """
        if not self.match_viz_path:
            return

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        kp_matches_viz = cv2.drawMatchesKnn(self.frames[i], key_points_i, self.frames[j], key_points_j, matches, None,
                                            **draw_params)

        def add_text(text, thickness, colour, position):
            cv2.putText(kp_matches_viz,
                        text=text,
                        org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=thickness,
                        color=colour)

        def add_text_with_shadow(text, thickness, colour, position):
            # shadow
            add_text(text, thickness + 1, colour=(0, 0, 0), position=position)
            # main text
            add_text(text, thickness, colour=colour, position=position)

        # status of frame pair
        add_text_with_shadow(text="accepted" if frame_accepted else "rejected", thickness=4,
                             colour=(0, 255, 0) if frame_accepted else (0, 0, 255), position=(8, 32))

        # statistics
        feature_was_matched = np.any(matches_mask, axis=1)
        num_matches = feature_was_matched.sum()
        percent_matched = 100 * feature_was_matched.mean()
        stat_string = f"{num_matches:>5,d}/{len(matches_mask):>5,d} ({percent_matched:>4.1f}%)"
        add_text_with_shadow(stat_string, thickness=4, colour=(255, 255, 255), position=(196, 32))

        kp_matches_viz = cv2.cvtColor(kp_matches_viz, cv2.COLOR_BGR2RGB)

        plt.imsave(pjoin(self.match_viz_path, f"{i:06d}-{j:06d}.jpg"), kp_matches_viz)

    def _print_results_stats(self, index_i: torch.Tensor, index_j: torch.Tensor, num_good_frame_pairs: int):
        """
        Print some statistics about the matched image features.
        :param index_i: The indices of the matched frames for the first frame of the pair.
        :param index_j: The indices of the matched frames for the second frame of the pair.
        :param num_good_frame_pairs: The number of frame pairs kept.
        """
        frames_with_matched_features = set(torch.hstack((index_i, index_j)).tolist())
        coverage = len(frames_with_matched_features) / self.dataset.num_frames

        log(f"Found {num_good_frame_pairs} good frame pairs ({num_good_frame_pairs}/{len(self.frame_pairs)})")
        log(f"Frame pairs cover {100 * coverage:.2f}% of the frames.")

        chunks = []
        chunk = []

        for frame_index in range(self.dataset.num_frames):
            if frame_index in frames_with_matched_features:
                chunk.append(frame_index)
            elif chunk:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)

        log(f"Found {len(chunks)} group(s) of consecutive frames.")


# noinspection PyArgumentList
class AlignmentType(enum.Enum):
    """How depth maps should be scaled and shifted (if at all) during pose optimisation."""

    """Align RGB-D frames with just the camera pose."""
    Rigid = enum.auto()

    """Align RGB-D frames with the camera pose and a scale and shift per depth map."""
    Affine = enum.auto()

    """Align RGB-D frames with the camera pose and a scale and shift per depth map."""
    Deformable = enum.auto()


class OptimisationParameters(torch.nn.Module):
    """The parameters subject to optimisation (camera poses)."""

    def __init__(self, initial_camera_poses: torch.FloatTensor,
                 alignment_type: AlignmentType,
                 scale_parameters: Optional[torch.FloatTensor] = None,
                 shift_parameters: Optional[torch.FloatTensor] = None,
                 dtype=torch.float32):
        """
        :param initial_camera_poses: The initial guess for the camera poses where each row is in the
            format [r_x, r_y, r_z, r_w, t_x, t_y, t_z] where r is a quaternion and t is a 3D position vector.
        :param alignment_type: The method for aligning the frames and whether to add additional parameters for
            pose optimisation.
        :param dtype: The data type to convert the parameters to.
        """
        super().__init__()

        def to_param(a: torch.Tensor):
            a_copy = a.clone().to(dtype)
            return torch.nn.Parameter(a_copy, requires_grad=True)

        r, t = initial_camera_poses[:, :4], initial_camera_poses[:, 4:]
        self.rotation_quaternions = to_param(r)
        self.translation_vectors = to_param(t)

        self.alignment_type = alignment_type

        num_frames = len(r)

        if alignment_type == AlignmentType.Rigid:
            shift_parameters = torch.empty(0)
            scale_parameters = torch.empty(0)
        elif alignment_type == AlignmentType.Affine:
            shift_parameters = shift_parameters if shift_parameters is not None else torch.zeros(num_frames)
            scale_parameters = scale_parameters if scale_parameters is not None else torch.ones(num_frames)
        elif alignment_type == AlignmentType.Deformable:
            shift_parameters = shift_parameters if shift_parameters is not None else torch.zeros((num_frames, 3, 3))
            scale_parameters = scale_parameters if scale_parameters is not None else torch.ones((num_frames, 3, 3))
        else:
            raise RuntimeError(f"Unsupported alignment type {alignment_type}.")

        self.shift = to_param(shift_parameters)
        self.scale = to_param(scale_parameters)

    @property
    def dtype(self):
        """The data type of the optimisation parameters."""
        return self.rotation_quaternions.dtype

    @property
    def device(self):
        """The device ('cpu' or 'cuda') that the optimisation parameters reside on."""
        return self.rotation_quaternions.device

    def __len__(self) -> int:
        """
        :return: The number of optimisation parameters.
        """
        return sum([p.nelement() for p in self.parameters()])

    def normalise_rotations(self):
        """
        Scale rotation quaternions to unit vectors.
        Note that this is done in place.
        """
        with torch.no_grad():
            r = self.rotation_quaternions
            r = r / torch.linalg.norm(r, ord=2, dim=1).reshape((-1, 1))
            self.rotation_quaternions = torch.nn.Parameter(r, requires_grad=True)

    def get_trajectory(self) -> np.ndarray:
        """Get the camera trajectory from the optimisation parameters as a (N, 7) NumPy array."""
        r = to_numpy(self.rotation_quaternions)
        t = to_numpy(self.translation_vectors)

        r = r / np.linalg.norm(r, ord=2, axis=1).reshape((-1, 1))

        return np.hstack((r, t))

    def sample_at(self, frame_indices: List[int]) -> 'OptimisationParameters':
        """
        Sample the optimisation parameters at the given frames.
        :param frame_indices: The frames to include.
        :return: The OptimisationParameters object that only contains the data for the specified frames.
        """
        r = clone_param(self.rotation_quaternions[frame_indices])
        t = clone_param(self.translation_vectors[frame_indices])
        trajectory = torch.hstack((r, t))

        if self.shift.numel() > 0:
            shift_parameters = clone_param(self.shift[frame_indices])
        else:
            shift_parameters = None
        if self.shift.numel() > 0:
            scale_parameters = clone_param(self.scale[frame_indices])
        else:
            scale_parameters = None

        return OptimisationParameters(initial_camera_poses=trajectory, alignment_type=self.alignment_type,
                                      scale_parameters=scale_parameters, shift_parameters=shift_parameters,
                                      dtype=self.dtype)

    def clone(self) -> 'OptimisationParameters':
        """Get a copy of the optimisation parameters object."""

        trajectory = torch.hstack((
            clone_param(self.rotation_quaternions), clone_param(self.translation_vectors)
        ))

        shift_parameters = clone_param(self.shift)
        scale_parameters = clone_param(self.scale)

        return OptimisationParameters(initial_camera_poses=trajectory, alignment_type=self.alignment_type,
                                      shift_parameters=shift_parameters, scale_parameters=scale_parameters,
                                      dtype=self.dtype)


class EarlyStopping:
    """A callback to keep track whether training has stagnated."""

    def __init__(self, patience=10, min_difference=0.0):
        """
        :param patience: The number of steps where the loss has not decreased more than `min_difference` before the
            flag `should_stop` is set.
        :param min_difference: The smallest change between the current loss and the best loss in the previous `patience`
            number of steps where training is not considered to have stagnated.
        """
        self.patience = patience
        self.min_difference = min_difference

        self.best_loss = float('inf')
        self.calls_since_last_best = 0

        self.should_stop = False

    def step(self, loss) -> bool:
        """
        Update the `should_stop` flag based on the current loss.
        :param loss: The loss for the current training step.
        :return: Whether training should stop.
        """
        loss = loss.detach().item()

        if loss < self.best_loss and abs(loss - self.best_loss) > self.min_difference:
            self.best_loss = loss
            self.calls_since_last_best = 0
        else:
            self.calls_since_last_best += 1

        if self.calls_since_last_best > self.patience:
            self.should_stop = True

        return self.should_stop


# noinspection PyArgumentList
class ResidualType(enum.Enum):
    """Different ways for calculating points and residuals in the optimisation step."""

    """
    Residuals are calculated on points from both frames projected into the world coordinate system.
    Faster but generally less accurate than `Image2D`.
    """
    World3D = enum.auto()

    """
    Residuals are calculated on points from one frame projected into the other frame in pixel coordinates.
    Slower but generally more accurate than `World3D`.
    """
    Image2D = enum.auto()


# noinspection PyArgumentList
class OptimisationStep(enum.Enum):
    """Represents steps in the pose optimisation pipeline."""

    """Local, pair-wise, frame alignment using residuals calculated over 3D correspondences."""
    PairWise3D = enum.auto()

    """Global (all frames) frame alignment using residuals calculated over 3D correspondences."""
    Global3D = enum.auto()
    # If using the frame sampling method `HIERARCHICAL`, running this step after `PairWise` will introduce
    # new constraints from non-adjacent frames. These new constraints should help improve the robustness of the
    # estimated poses.

    # The below steps are same as their 3D counterparts except a different residual type is used for calculating the
    # loss - the `Image2D` residuals. This projects the points from frame A into the 3D world coordinate system,
    # and then onto the image plane of frame B and measures the pixel distance between correspondences.
    # This step takes much longer than the `Global` and `PairWise` steps that use the `World3D` residuals, but also
    # produces slightly better alignments.

    """Local, pair-wise, frame alignment using residuals calculated over 2D correspondences."""
    PairWise2D = enum.auto()

    """Global (all frames) frame alignment using residuals calculated over 2D correspondences."""
    Global2D = enum.auto()


class OptimisationOptions:
    """Configuration for the `PoseOptimiser` class."""

    default_pipeline = (OptimisationStep.PairWise3D, OptimisationStep.Global3D)

    def __init__(self, num_epochs=4000, learning_rate=1e-2, l2_regularisation=0.5, min_loss_delta=1e-4,
                 lr_scheduler_patience=50, early_stopping_patience=75, alignment_type=AlignmentType.Rigid,
                 steps=default_pipeline, position_only=False, fine_tune=True, pose_t_reg=0.5, pose_r_reg=1.0,
                 trajectory_smoothing: Optional[float] = None, clip_distance: Optional[float] = 1.0):
        """
        :param num_epochs: The maximum number of iterations to run the optimisation algorithm for.
        :param learning_rate: How much the optimisation parameters are adjusted each step. Higher values (>0.1) may
            lead to instability and the algorithm not finding a good solution.
        :param l2_regularisation: Controls how much the scale and shift parameters influence the loss
            (if alignment type is not `Rigid`). The default value is taken from the paper 'Instant 3D Photography'.
        :param min_loss_delta: The minimum change in the loss before the learning rate scheduler and early stopping
            objects start counting down to either lower the learning rate or exit early.
        :param lr_scheduler_patience: How many epochs where the loss has changed less than `min_loss_delta` before
            lowering the learning rate.
        :param early_stopping_patience: How many epochs where the loss has changed less than `min_loss_delta` before
            exiting the optimisation loop early.
        :param alignment_type: The method for aligning the frames and whether to add additional parameters for
            pose optimisation.
        :param position_only: Whether to optimise the position (translation vector) of the pose only.
        :param steps: Use to configure the optimisation pipeline (which steps/operations to perform and in what order).
        :param fine_tune: Whether to add a step at the end that optimises the poses without the smoothing loss terms.
            This can remove some blurriness from a 3D reconstruction using the optimised poses.
        :param pose_t_reg: The regularisation weight for the distance between translation vectors or adjacent frames.
        :param pose_r_reg: The regularisation weight for the distance between quaternions or adjacent frames.
        :param trajectory_smoothing: (optional) How much smoothing [0, 1] to apply to the trajectory (positions) after
            the final optimisation step. Setting this to `None` disables smoothing.
        :param clip_distance: (optional) The maximum distance allowed between frames.
            Set to `None` to disable hard constraint on frame distance.
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.l2_regularisation = l2_regularisation
        self.min_loss_delta = min_loss_delta
        self.lr_scheduler_patience = lr_scheduler_patience
        self.early_stopping_patience = early_stopping_patience
        self.alignment_type = alignment_type
        self.steps = steps
        self.position_only = position_only
        self.pose_t_reg = pose_t_reg
        self.pose_r_reg = pose_r_reg
        self.fine_tune = fine_tune
        self.trajectory_smoothing = trajectory_smoothing
        self.clip_distance = clip_distance

        check_domain(num_epochs, 'num_epochs', int, Domain.Positive)
        check_domain(learning_rate, 'learning_rate', float, Domain.Positive)
        check_domain(l2_regularisation, 'l2_regularisation', float, Domain.NonNegative)
        check_domain(min_loss_delta, 'min_loss_delta', float, Domain.Positive)
        check_domain(lr_scheduler_patience, 'lr_scheduler_patience', int, Domain.Positive)
        check_domain(early_stopping_patience, 'early_stopping_patience', int, Domain.Positive)
        check_domain(pose_t_reg, 'pose_t_reg', float, Domain.NonNegative)
        check_domain(pose_r_reg, 'pose_r_reg', float, Domain.NonNegative)
        check_domain(trajectory_smoothing, 'trajectory_smoothing', float, Domain.NonNegative, nullable=True)
        check_domain(clip_distance, 'clip_distance', float, Domain.NonNegative, nullable=True)

        if not isinstance(steps, (tuple, list)) or len(steps) == 0:
            for step in steps:
                if not isinstance(step, OptimisationStep):
                    raise ValueError(f"steps must only contain values of type {OptimisationStep}, "
                                     f"but found a value of type {type(step)}")

            raise ValueError(f"steps must a tuple or list with at least one element, got {type(steps)} instead.")

    def __repr__(self):
        return f"{self.__class__.__name__}(num_epochs={self.num_epochs}, " \
               f"learning_rate={self.learning_rate}, " \
               f"l2_regularisation={self.l2_regularisation}, " \
               f"min_loss_delta={self.min_loss_delta}, " \
               f"lr_scheduler_patience={self.lr_scheduler_patience}, " \
               f"early_stopping_patience={self.early_stopping_patience}, " \
               f"alignment_type={self.alignment_type}, " \
               f"steps={self.steps}, " \
               f"position_only={self.position_only}, " \
               f"pose_t_reg={self.pose_t_reg}, " \
               f"pose_r_reg={self.pose_r_reg}, " \
               f"fine_tune={self.fine_tune}, " \
               f"trajectory_smoothing={self.trajectory_smoothing}, " \
               f"clip_distance={self.clip_distance})"

    def copy(self) -> 'OptimisationOptions':
        """Make a copy of the optimisation options."""
        return OptimisationOptions(num_epochs=self.num_epochs, learning_rate=self.learning_rate,
                                   l2_regularisation=self.l2_regularisation, min_loss_delta=self.min_loss_delta,
                                   lr_scheduler_patience=self.lr_scheduler_patience,
                                   early_stopping_patience=self.early_stopping_patience,
                                   alignment_type=self.alignment_type, steps=self.steps,
                                   position_only=self.position_only, fine_tune=self.fine_tune,
                                   pose_t_reg=self.pose_t_reg, pose_r_reg=self.pose_r_reg,
                                   trajectory_smoothing=self.trajectory_smoothing)


class PoseOptimiser:
    """Algorithm for optimising the camera trajectory of a RGB-D video sequence."""

    DEBUG_FOLDER = 'pose_optim'

    def __init__(self, dataset: VTMDataset, frame_sampling=FrameSamplingMode.Hierarchical,
                 feature_extraction_options=FeatureExtractionOptions(),
                 optimisation_options=OptimisationOptions(), debug=True):
        """
        :param dataset: The dataset to optimise over. The pose data present in the dataset will be used as the initial
            estimate.
        :param frame_sampling: The method to use for sampling frame pairs.
        :param feature_extraction_options: The options for the FeatureExtractor object.
        :param optimisation_options: The options for the pose optimisation algorithm.
        :param debug: Whether to run in debug mode (enables debug output and results caching).
        """
        self.dataset = dataset
        self.frame_sampling = frame_sampling
        self.feature_extraction_options = feature_extraction_options
        self.optimisation_options = optimisation_options
        self.debug = debug
        self.debug_path: Optional[str] = None

    def run(self, num_frames=-1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimise the pose data.

        :param num_frames: (optional) a limit on the number of frames to use. Defaults to -1 which will use all frames.
        :return: the (N, 7) optimised camera trajectory where each row is a quaternion (scalar last) and a
            translation vector; the per frame scale factor; and the per frame shift factor.
        """
        if num_frames == -1:
            num_frames = self.dataset.num_frames

        truncated_camera_trajectory = torch.from_numpy(self.dataset.camera_trajectory[:num_frames])

        self._setup_debug_folder()
        frame_pairs = self._sample_frame_pairs(self.frame_sampling)
        fixed_parameters = self._extract_feature_points(frame_pairs, self.feature_extraction_options)
        optimisation_parameters = OptimisationParameters(truncated_camera_trajectory,
                                                         alignment_type=self.optimisation_options.alignment_type)
        optimised_parameters = self._optimise_pose(fixed_parameters, optimisation_parameters,
                                                   optimisation_options=self.optimisation_options,
                                                   num_frames=num_frames)
        interpolated_trajectory = self._interpolate_poses_without_matches(fixed_parameters,
                                                                          optimised_parameters.get_trajectory())
        if self.optimisation_options.trajectory_smoothing:
            interpolated_trajectory = self._smooth_trajectory(trajectory=interpolated_trajectory,
                                                              weight=self.optimisation_options.trajectory_smoothing)

        # TODO: Why does this step need to be done for TSDFFusion reconstruction to interpret the pose data accurately?
        final_camera_trajectory = invert_trajectory(interpolated_trajectory)

        scale = to_numpy(optimised_parameters.scale)
        shift = to_numpy(optimised_parameters.shift)

        if self.debug:
            # noinspection PyTypeChecker
            np.savetxt(pjoin(self.debug_path, 'optimised_camera_trajectory.txt'), final_camera_trajectory)

            should_reshape = self.optimisation_options.alignment_type == AlignmentType.Deformable

            np.savetxt(pjoin(self.debug_path, 'scale.txt'),
                       scale.reshape((num_frames, -1)) if should_reshape else scale)

            np.savetxt(pjoin(self.debug_path, 'shift.txt'),
                       shift.reshape((num_frames, -1)) if should_reshape else shift)

        return final_camera_trajectory, scale, shift

    def _setup_debug_folder(self):
        """Create the debug folder if in debug mode and the folder doesn't already exist."""
        if self.debug:
            self.debug_path = pjoin(self.dataset.base_path, self.DEBUG_FOLDER)

            os.makedirs(self.debug_path, exist_ok=True)

    def _sample_frame_pairs(self, frame_sampling_mode: FrameSamplingMode, num_frames=-1) -> FramePairs:
        """
        Select frame pairs for image feature extraction.

        :param frame_sampling_mode: The strategy to use for sampling frame pairs.
        :param num_frames: (optional) The frame index to stop at.
            If set to -1, this value is set to the length of the dataset.
        :return: A list of pairs of frame indices.
        """
        num_frames = self.dataset.num_frames if num_frames == -1 else num_frames
        frame_pairs = []

        if frame_sampling_mode == FrameSamplingMode.Exhaustive:
            for i in range(num_frames):
                for j in range(i + 1, num_frames):
                    frame_pairs.append((i, j))

        elif frame_sampling_mode in (FrameSamplingMode.Consecutive, FrameSamplingMode.ConsecutiveNoOverlap,
                                     FrameSamplingMode.ConsecutiveNoOverlapOffset):
            if frame_sampling_mode == FrameSamplingMode.ConsecutiveNoOverlapOffset:
                start = 1
            else:
                start = 0

            if frame_sampling_mode in (FrameSamplingMode.ConsecutiveNoOverlap,
                                       FrameSamplingMode.ConsecutiveNoOverlapOffset):
                step = 2
            else:
                step = 1

            for i in range(start, num_frames - 1, step):
                frame_pairs.append((i, i + 1))

        elif frame_sampling_mode == FrameSamplingMode.Hierarchical:
            # Adapted from https://github.com/facebookresearch/consistent_depth/blob/e2c9b724d3221aa7c0bf89aa9449ae33b418d943/utils/frame_sampling.py#L78
            max_level = int(np.floor(np.log2(num_frames - 1)))

            for level in range(max_level + 1):
                step = 1 << level

                for start in range(0, num_frames, step):
                    end = start + step

                    if end >= num_frames:
                        continue

                    frame_pairs.append((start, end))
        else:
            raise RuntimeError(f"Unsupported frame sampling mode: {frame_sampling_mode}.")

        return frame_pairs

    def _extract_feature_points(self, frame_pairs: FramePairs,
                                feature_extraction_options=FeatureExtractionOptions()) -> FeatureSet:
        """
        Run the feature extractor.
        :param frame_pairs: The pairs of frames to try to find correspondences between.
        :param feature_extraction_options: The options for the feature extraction object.
        :return: A FeatureSet object which contains the paired correspondence data objects and the camera matrix K.
        """
        feature_extractor = FeatureExtractor(self.dataset, frame_pairs, feature_extraction_options,
                                             debug_path=self.debug_path)

        feature_set = feature_extractor.extract_feature_points()
        feature_set = feature_set.subset_from(frame_pairs)

        return feature_set

    def _optimise_pose(self, fixed_parameters: FeatureSet,
                       optimisation_parameters: OptimisationParameters,
                       optimisation_options=OptimisationOptions(),
                       num_frames=-1) -> OptimisationParameters:
        """
        Optimises the given poses.

        :param fixed_parameters: Object containing the paired sets of correspondence data and the
            camera intrinsics matrix K.
        :param optimisation_parameters: The pose parameters to optimise.
        :param optimisation_options: The options for the optimisation algorithm.
        :param num_frames: (optional) The frame index to stop at.
            If set to -1, this value is set to the length of the dataset.
        :return: the optimised parameters.
        """
        if num_frames != self.dataset.num_frames:
            fixed_parameters = fixed_parameters.sample_at(range(num_frames))
            optimisation_parameters = optimisation_parameters.sample_at(list(range(num_frames)))

        if torch.cuda.is_available():
            # TODO: Why is this not in-place like optimisation_parameters.cuda()?
            fixed_parameters = fixed_parameters.cuda()
            optimisation_parameters.cuda()

        device = fixed_parameters.device
        num_steps = len(self.optimisation_options.steps)

        if self.optimisation_options.fine_tune:
            num_steps += 1

        for i, step in enumerate(self.optimisation_options.steps):
            log(f"Step {i + 1}/{num_steps}: {step.name} Alignment...")

            if step == OptimisationStep.PairWise2D or step == OptimisationStep.PairWise3D:
                residual_type = ResidualType.Image2D if step == OptimisationStep.PairWise2D else ResidualType.World3D

                optimisation_parameters = self._optimise_pairwise(fixed_parameters=fixed_parameters,
                                                                  optimisation_parameters=optimisation_parameters,
                                                                  optimisation_options=optimisation_options,
                                                                  residual_type=residual_type,
                                                                  num_frames=num_frames)
            elif step == OptimisationStep.Global2D or step == OptimisationStep.Global3D:
                residual_type = ResidualType.Image2D if step == OptimisationStep.Global2D else ResidualType.World3D

                optimisation_parameters = self._optimisation_loop(fixed_parameters=fixed_parameters,
                                                                  optimisation_parameters=optimisation_parameters,
                                                                  optimisation_options=optimisation_options,
                                                                  residual_type=residual_type)
            else:
                raise RuntimeError(f"Unsupported optimisation step: {step}.")

            if self.debug:
                step_name = f"{i}_{step.name}"
                self.visualise_solution(optimisation_parameters, step_name)

            optimisation_parameters = optimisation_parameters.to(device)

        if self.optimisation_options.fine_tune:
            log(f"Step {num_steps}/{num_steps}: Fine tuning...")

            optimisation_parameters = self._optimisation_loop(fixed_parameters=fixed_parameters,
                                                              optimisation_parameters=optimisation_parameters,
                                                              optimisation_options=optimisation_options,
                                                              residual_type=ResidualType.World3D,
                                                              smooth_trajectory=False)

            if self.debug:
                step_name = f"{num_steps}_FineTune"
                self.visualise_solution(optimisation_parameters, step_name)

        return optimisation_parameters.cpu()

    def _optimise_pairwise(self, fixed_parameters: FeatureSet,
                           optimisation_parameters: OptimisationParameters,
                           optimisation_options=OptimisationOptions(),
                           residual_type=ResidualType.World3D,
                           num_frames=-1) -> OptimisationParameters:
        """
        Optimise poses over each unique consecutive frame pair in isolation and merge the resulting pose data
         to get locally optimal alignments.
        Note: Rigid alignment is used regardless of what is specified in `optimisation_options`.

        :param fixed_parameters: Object containing the paired sets of correspondence data and the
            camera intrinsics matrix K.
        :param optimisation_parameters: The pose and depth scaling parameters to optimise.
        :param optimisation_options: The options for the optimisation algorithm.
        :param residual_type: The type of residuals to use. Either: `World3D` which projects both frames of a frame pair
            into the world coordinate system and measures the geometric distance between correspondences in 3D space;
            or `Image2D` which projects the points of one frame onto the other and measures the pixel distance between
            the correspondences.
        :param num_frames: (optional) The frame index to stop at.
            If set to -1, this value is set to the length of the dataset.
        :return: the optimised parameters.
        """
        if num_frames == -1:
            num_frames = self.dataset.num_frames

        options = optimisation_options.copy()
        # First align pose data alone and not scale+shift data, so force `Rigid` alignment.
        # This avoids the problem of how to combine scale+shift for frames shared between frame pairs.
        options.alignment_type = AlignmentType.Rigid

        def optimise_frame_pairs(frame_sampling_mode: FrameSamplingMode):
            frame_pairs = self._sample_frame_pairs(frame_sampling_mode, num_frames)
            feature_set = fixed_parameters.subset_from(frame_pairs)
            pair_wise_parameters = self._optimisation_loop(feature_set, optimisation_parameters, options,
                                                           residual_type=residual_type)
            trajectory = pair_wise_parameters.get_trajectory()

            for frame_pair in frame_pairs:
                key = tuple(frame_pair)
                indices = list(frame_pair)

                pose_data[key] = trajectory[indices]

        # First we run the optimisation over the frame pairs (0, 1), (2, 3), ..., (n - 2, n - 1) where n is the number
        # of frames in the sequence. Since they don't overlap, they should not affect each other during
        # back-propagation.
        # The second run uses the frame pairs (1, 2), (3, 4), ..., (n - 3, n - 2).
        # Combining the optimised poses from both runs gives us overlapping frames (0, 1) -> (1, 2) -> (2, 3) which can
        # be chained to give us the absolute pose for all the frames (unless some frame pairs could not be matched).
        pose_data = dict()
        optimise_frame_pairs(FrameSamplingMode.ConsecutiveNoOverlap)
        optimise_frame_pairs(FrameSamplingMode.ConsecutiveNoOverlapOffset)

        merged_pose_data = [get_identity_pose()]
        previous_pose = merged_pose_data[0]

        for i, j in sorted(pose_data.keys()):
            pose_i, pose_j = pose_data[i, j]
            j_rel_to_i = subtract_pose(pose_i, pose_j)
            next_pose = add_pose(previous_pose, j_rel_to_i)
            merged_pose_data.append(next_pose)
            previous_pose = next_pose

        camera_trajectory = torch.from_numpy(np.asarray(merged_pose_data))

        return OptimisationParameters(camera_trajectory,
                                      optimisation_parameters.alignment_type,
                                      optimisation_parameters.scale,
                                      optimisation_parameters.shift,
                                      optimisation_parameters.dtype)

    def _optimisation_loop(self, fixed_parameters: FeatureSet,
                           optimisation_parameters: OptimisationParameters,
                           optimisation_options=OptimisationOptions(),
                           residual_type=ResidualType.World3D,
                           smooth_trajectory=True) -> OptimisationParameters:
        """
        Run the full optimisation loop.

        :param fixed_parameters: Object containing the paired sets of correspondence data and the
            camera intrinsics matrix K.
        :param optimisation_parameters: The pose and depth map scaling parameters to optimise.
        :param optimisation_options: The options for the optimisation algorithm.
        :param residual_type: The type of residuals to use. Either: `World3D` which projects both frames of a frame pair
            into the world coordinate system and measures the geometric distance between correspondences in 3D space;
            or `Image2D` which projects the points of one frame onto the other and measures the pixel distance between
            the correspondences.
        :param smooth_trajectory: Whether to add loss terms to smooth the trajectory and also encourage smaller
            changes in pose. While this helps discourage 'clustering' and noisy trajectories, it may lead to blurrier
            reconstructions.
        :return: the optimised parameters object.
        """
        options = optimisation_options

        # Clone parameters to avoid side effects from this method.
        optimisation_parameters = optimisation_parameters.clone().to(optimisation_parameters.device)

        optimiser = torch.optim.Adam(optimisation_parameters.parameters(), lr=options.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=options.lr_scheduler_patience,
                                                                  mode='min', threshold=options.min_loss_delta,
                                                                  threshold_mode='abs')
        early_stopping = EarlyStopping(patience=options.early_stopping_patience, min_difference=options.min_loss_delta)

        progress_bar = tqdm(range(options.num_epochs))

        for _ in progress_bar:
            with torch.no_grad():
                # TODO: Any way of speeding this step up?
                # This step helps prevent quaternion rotations' norms becoming zero.
                r = optimisation_parameters.rotation_quaternions
                r = r / torch.linalg.norm(r, ord=2, dim=1).reshape((-1, 1))
                optimisation_parameters.rotation_quaternions = torch.nn.Parameter(r, requires_grad=True)

                # This puts a hard limit on the distance between frames.
                if self.optimisation_options.clip_distance is not None:
                    position_vectors = self._clip_distance_between_frames(
                        optimisation_parameters,
                        max_displacement_per_sec=self.optimisation_options.clip_distance
                    )
                    optimisation_parameters.translation_vectors = torch.nn.Parameter(position_vectors,
                                                                                     requires_grad=True)

            optimiser.zero_grad()
            residuals = self._calculate_residuals(fixed_parameters, optimisation_parameters, residual_type,
                                                  optimisation_options.alignment_type)
            loss = self._calculate_loss(residuals, optimisation_parameters, smooth_trajectory)
            loss.backward()

            # Pin first frames at origin or at least whatever they were initialised to.
            optimisation_parameters.translation_vectors.grad[0] *= 0.0

            if self.optimisation_options.position_only:
                optimisation_parameters.rotation_quaternions.grad[:] *= 0.0
            else:
                optimisation_parameters.rotation_quaternions.grad[0] *= 0.0

            optimiser.step()
            lr_scheduler.step(loss)
            early_stopping.step(loss)

            loss_value = loss.detach().cpu().item()

            lr = float('nan')

            for param_group in optimiser.param_groups:
                lr = param_group['lr']
                break

            progress_bar.set_postfix_str(f"Loss: {loss_value:<7.4f} - LR: {lr:,.2e}")

            if early_stopping.should_stop:
                break

        return optimisation_parameters

    def _clip_distance_between_frames(self, parameters: OptimisationParameters, max_displacement_per_sec=1.0) \
            -> torch.FloatTensor:
        """
        Enforce a maximum distance (Euclidean) between all pairs of adjacent frames in a trajectory.

        **Note:** Make sure this method is called within a `torch.no_grad()` context to ensure
        no side effects (gradients) and the best performance.

        :param parameters: The optimisation parameters with the trajectory to clip.
        :param max_displacement_per_sec: The maximum distance (meters) the camera position is allowed to change in
            one millisecond.
        :return: The clipped trajectory.
        """
        position_vectors = parameters.translation_vectors.clone()
        num_frames = len(position_vectors)
        max_frame_distance = max_displacement_per_sec * (1.0 / self.dataset.fps)

        # TODO: Speed this up. Try vectorising the calculations?
        for i in range(1, num_frames):
            pos_curr = position_vectors[i]
            pos_prev = position_vectors[i - 1]

            pos_diff = pos_curr - pos_prev
            distance = torch.linalg.norm(pos_diff)

            if distance > max_frame_distance:
                direction = pos_diff / distance
                new_pos = pos_prev + max_frame_distance * direction

                position_vectors[i] = new_pos

                final_diff = position_vectors[i] - pos_curr

                next_i = i + 1

                if next_i < num_frames:
                    position_vectors[i + 1:] += final_diff

        return position_vectors

    def _calculate_loss(self, residuals: torch.Tensor, optimisation_parameters: OptimisationParameters,
                        smooth_trajectory: bool):
        loss = torch.mean(torch.linalg.norm(residuals, ord=2, dim=0))

        if smooth_trajectory:
            t = optimisation_parameters.translation_vectors

            order_one_grad = t[:-1] - t[1:]
            order_two_grad = t[:-2] - 2 * t[1:-1] + t[2:]
            order_three_grad = order_two_grad[:-1] - order_two_grad[1:]
            loss += self.optimisation_options.pose_t_reg * torch.mean(torch.sum(torch.square(order_one_grad), dim=1))
            loss += self.optimisation_options.pose_t_reg * torch.mean(torch.sum(torch.square(order_two_grad), dim=1))
            loss += self.optimisation_options.pose_t_reg * torch.mean(torch.sum(torch.square(order_three_grad), dim=1))

            r = optimisation_parameters.rotation_quaternions
            rotational_distance = torch.mean(1 - torch.square(torch.einsum('ij,ij->i', r[:-1], r[1:])))
            loss += self.optimisation_options.pose_r_reg * rotational_distance

        if self.optimisation_options.alignment_type != AlignmentType.Rigid:
            l2_reg = self.optimisation_options.l2_regularisation
            # noinspection PyTypeChecker
            loss += l2_reg * torch.mean(torch.square(1. / optimisation_parameters.scale - 1.))
            loss += 2 * l2_reg * torch.mean(torch.square(optimisation_parameters.shift))

        return loss

    def _calculate_residuals(self, fixed_parameters: FeatureSet,
                             optimisation_parameters: OptimisationParameters,
                             residual_type: ResidualType,
                             alignment_type=AlignmentType.Rigid) -> torch.Tensor:
        """
        Calculate the distance (residuals) between correspondences given the camera poses.

        :param fixed_parameters: Object containing the paired sets of correspondence data and the
            camera intrinsics matrix K.
        :param optimisation_parameters: The pose parameters to optimise.
        :param residual_type: The type of residuals to use. Either: `World3D` which projects both frames of a frame pair
            into the world coordinate system and measures the geometric distance between correspondences in 3D space;
            or `Image2D` which projects the points of one frame onto the other and measures the pixel distance between
            the correspondences.
        :param alignment_type: (optional) How depth maps should be scaled and shifted (if at all).
        :return: Given N correspondences, returns N residuals.
        """
        p = self._project_to_world_coords(fixed_parameters.frame_i,
                                          fixed_parameters.camera_matrix,
                                          optimisation_parameters,
                                          alignment_type)

        if residual_type == ResidualType.World3D:
            q = self._project_to_world_coords(fixed_parameters.frame_j,
                                              fixed_parameters.camera_matrix,
                                              optimisation_parameters,
                                              alignment_type)

            return p - q
        elif residual_type == ResidualType.Image2D:
            q_ = self._project_to_image_coords(p, fixed_parameters.frame_j, fixed_parameters.camera_matrix,
                                               optimisation_parameters)

            return fixed_parameters.frame_j.points.T - q_
        else:
            raise RuntimeError(f"calculate_residuals got give an unsupported residuals type "
                               f"{residual_type}.")

    def _project_to_world_coords(self, frame_data: FeatureData, camera_matrix: torch.Tensor,
                                 params: OptimisationParameters, alignment_type=AlignmentType.Rigid) -> torch.Tensor:
        """
        Project correspondences from 2D image coordinates to 3D world coordinates.

        :param frame_data: The correspondence data to project.
        :param camera_matrix: The camera matrix for projecting points into camera space.
        :param params: The optimisation parameters including the camera poses.
        :param alignment_type: (optional) How depth maps should be scaled and shifted (if at all).
        :return: A (3, N) tensor of the projected points.
        """
        indices = frame_data.index
        points = frame_data.points
        depth = frame_data.depth

        if alignment_type == AlignmentType.Affine:
            depth = 1. / (params.scale[indices] * (1. / depth) + params.shift[indices])
        elif alignment_type == AlignmentType.Deformable:
            u, v = torch.round(points).to(torch.long).T
            scale = self._interpolate_field(params.scale)[indices, u, v]
            shift = self._interpolate_field(params.shift)[indices, u, v]
            depth = 1. / (scale * (1. / depth) + shift)

        u, v = points.T
        f_x, f_y, c_x, c_y = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

        points_camera_space = torch.vstack(
            (
                (u - c_x) * depth / f_x,
                (v - c_y) * depth / f_y,
                depth
            )
        )

        r = params.rotation_quaternions[indices].T
        t = params.translation_vectors[indices].T
        # Note: The conjugate of a quaternion is the equivalent of an inverse.
        points_world_space = Quaternion(r).normalise().conjugate().apply(points_camera_space - t)

        return points_world_space

    def _interpolate_field(self, fields: torch.FloatTensor) -> torch.FloatTensor:
        """
        Take a (N, H, W) tensor of (H, W) grids and scale them up to the resolution of the dataset.

        :param fields: a (N, H, W) tensor of (H, W) grids.
        :return: The (N, frame_width, frame_height) tensor of grids with interpolated values.
        """
        return torch.nn.functional.interpolate(fields.unsqueeze(1),
                                               size=(self.dataset.frame_width, self.dataset.frame_height),
                                               align_corners=True,
                                               mode='bilinear').squeeze(1)

    @staticmethod
    def _project_to_image_coords(points_world_space: torch.Tensor, frame_data: FeatureData,
                                 camera_matrix: torch.Tensor,
                                 params: OptimisationParameters) -> torch.Tensor:
        """
        Project points from 3D world coordinates to 2D image coordinates.
        :param points_world_space:
        :param frame_data: The correspondence data to project.
        :param camera_matrix: The camera matrix for projecting points into camera space.
        :param params: The optimisation parameters including the camera poses.
        :return: A (2, N) tensor of the projected points.
        """
        indices = frame_data.index

        r = params.rotation_quaternions[indices].T
        t = params.translation_vectors[indices].T
        points_camera_space = Quaternion(r).normalise().apply(points_world_space) + t

        x, y, z = points_camera_space
        f_x, f_y, c_x, c_y = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
        points_image_space = torch.vstack((f_x * x + c_x * z, f_y * y + c_y * z)) / z

        return points_image_space

    @staticmethod
    def _interpolate_poses_without_matches(feature_set: FeatureSet,
                                           optimised_camera_trajectory: np.ndarray) -> np.ndarray:
        """
        Fill in pose data for frames that were not in a frame pair that had enough matching image features.

        :param feature_set: The correspondence data (this function wants the frame indices).
        :param optimised_camera_trajectory: The optimised pose data.
        :return: The optimised pose data with any gaps filled in.
        """
        num_frames = len(optimised_camera_trajectory)

        all_indices = torch.hstack((feature_set.frame_i.index, feature_set.frame_j.index)).tolist()
        frames_with_matched_features = set(index for index in all_indices if index < num_frames)

        chunks = []
        chunk = []

        for frame_index in range(num_frames):
            if frame_index not in frames_with_matched_features:
                chunk.append(frame_index)
            elif chunk:
                chunks.append(chunk)
                chunk = []

        if chunk:
            chunks.append(chunk)

        interpolated_poses = optimised_camera_trajectory.copy()

        for chunk in chunks:
            start = max(0, chunk[0] - 1)
            end = min(chunk[-1] + 1, num_frames - 1)

            start_rotation = optimised_camera_trajectory[start, :4]
            end_rotation = optimised_camera_trajectory[end, :4]
            start_position = optimised_camera_trajectory[start, 4:]
            end_position = optimised_camera_trajectory[end, 4:]

            key_frame_times = [0, 1]
            times_to_interpolate = np.linspace(0, 1, num=end - start + 1)

            slerp = Slerp(times=key_frame_times, rotations=Rotation.from_quat([start_rotation, end_rotation]))
            lerp = interp1d(key_frame_times, [start_position, end_position], axis=0)

            interpolated_poses[start:end + 1, 4:] = lerp(times_to_interpolate)
            interpolated_poses[start:end + 1, :4] = slerp(times_to_interpolate).as_quat()

        return interpolated_poses

    @staticmethod
    def _smooth_trajectory(trajectory: np.ndarray, weight=0.9) -> np.ndarray:
        """
        Smooths the translation vectors of the given trajectory.
        This is done by calculating the exponential moving averages (EMA) for the translation vectors.

        :param trajectory: The Nx7 camera trajectory to smooth.
        :param weight: The weight alpha for the EMA.
        :return: The smoothed trajectory.
        """
        smoothed_positions = np.zeros_like(trajectory[:, 4:])

        smoothed_positions[0] = trajectory[0, 4:]

        for i in range(1, len(smoothed_positions)):
            smoothed_positions[i] = weight * trajectory[i, 4:] + (1 - weight) * smoothed_positions[i - 1]

        smoothed_trajectory = np.hstack((trajectory[:, :4], smoothed_positions))

        return smoothed_trajectory

    def visualise_solution(self, solution: OptimisationParameters, label: str):
        # The inverse transform is needed so that the trajectories are comparable to the ground truth ones and
        # the final trajectories.
        trajectory = invert_trajectory(solution.get_trajectory())[:, 4:]
        output_path = pjoin(self.debug_path, f"{label}.png")

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

        def plot_trajectory(plot_axis, secondary_axis='y'):
            if secondary_axis == 'y':
                axis = 1
            elif secondary_axis == 'z':
                axis = 2
            else:
                raise RuntimeError(f"secondary_axis must be one of ('y', 'z').")

            plot_axis.plot(trajectory[:, 0], trajectory[:, axis], '-', color="black", label="trajectory")
            plot_axis.legend()

            plot_axis.set_xlabel("x [m]")
            plot_axis.set_ylabel(f"{secondary_axis} [m]")
            plot_axis.set_title(f"Trajectory on X{secondary_axis.upper()} Plane")

        plot_trajectory(ax1, secondary_axis='y')
        plot_trajectory(ax2, secondary_axis='z')

        plt.tight_layout()
        plt.savefig(output_path, dpi=90)


class ForegroundPoseOptimiser:
    def __init__(self, dataset: VTMDataset, learning_rate=1e-5, num_epochs=100):
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def run(self) -> np.ndarray:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_frames = self.dataset.num_frames

        def get_point_cloud(index):
            depth = self.dataset.depth_dataset[index]
            mask = self.dataset.mask_dataset[index]
            # TODO: Track instances between frames and calculate loss per instance
            mask = mask > 0

            point_cloud = point_cloud_from_depth(depth, mask, self.dataset.camera_matrix)

            return point_cloud

        # (N, M, 3) tensor where: N is the number of frames;
        #  and M is the number of pixels belonging to dynamic objects in each depth map.
        point_clouds = tqdm_imap(get_point_cloud, list(range(num_frames)))
        trajectory_torch = torch.from_numpy(self.dataset.camera_trajectory.copy())
        params = OptimisationParameters(trajectory_torch, AlignmentType.Rigid).to(device)

        optimiser = torch.optim.Adam(params.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        chunks = []
        chunk = []
        min_chunk_size = 3

        centroids = np.zeros((num_frames, 3), dtype=float)

        for i, point_cloud in enumerate(point_clouds):
            if len(point_cloud) > 0:
                chunk.append(i)
                centroids[i] = np.mean(point_cloud, axis=0)
            else:
                if len(chunk) >= min_chunk_size:
                    chunks.append(chunk)

                chunk = []

        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        centroids_camera_space = torch.from_numpy(centroids).to(device)

        with torch.no_grad():
            centroids_world_space_gt = Quaternion(params.rotation_quaternions.T)\
                .normalise()\
                .conjugate()\
                .apply((centroids_camera_space - params.translation_vectors).T).T

        with tqdm(range(self.num_epochs), total=self.num_epochs) as progress_bar:
            for _ in progress_bar:
                optimiser.zero_grad()

                loss = torch.tensor(0.0, device=device)

                for chunk in chunks:
                    r = params.rotation_quaternions[chunk]
                    t = params.translation_vectors[chunk]
                    centroids_chunk = centroids_camera_space[chunk]

                    centroids_world_space = Quaternion(r.T) \
                        .normalise() \
                        .conjugate() \
                        .apply((centroids_chunk - t).T).T

                    error_geom = torch.mean(torch.norm(centroids_world_space_gt[chunk] - centroids_world_space, dim=1))
                    error_temp = torch.mean(torch.norm(t[:-2] - 2 * t[1:-1] + t[2:]))
                    error_vel = torch.mean(torch.norm(t[:-1] - t[1:]))

                    w_geom = 0.01
                    w_temp = 0.1
                    w_vel = 0.1
                    loss += w_geom * error_geom + w_temp * error_temp + w_vel * error_vel

                loss.backward()
                optimiser.step()

                loss_value = loss.detach().cpu().item()

                lr = float('nan')

                for param_group in optimiser.param_groups:
                    lr = param_group['lr']
                    break

                progress_bar.set_postfix_str(f"Loss: {loss_value:<7.4f} - LR: {lr:,.2e}")

        return params.get_trajectory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to the VTM formatted dataset.')
    parser.add_argument('--num_frames', type=int, default=-1,
                        help='Number of frames to optimise. If set to -1 (default), use all frames.')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to perform an additional fine tuning step.')
    parser.add_argument('--params_init', type=str, choices=['gt', 'random'], default='gt',
                        help='How to initialise the camera trajectory.')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed to use when initialising camera trajectory with random data.')
    args = parser.parse_args()

    if not VTMDataset.is_valid_folder_structure(args.dataset_path):
        raise RuntimeError(f"The path {args.dataset_path} does not point to a valid dataset.")

    dataset = VTMDataset(args.dataset_path, overwrite_ok=False)

    num_frames = args.num_frames

    if num_frames == -1:
        num_frames = dataset.num_frames
    elif num_frames < 2:
        raise RuntimeError(f"--num_frames must at least 2, but got {num_frames}.")

    if args.params_init == 'random':
        with temp_seed(args.random_seed):
            dataset.camera_trajectory[:, :4] = Rotation.random(len(dataset), random_state=args.random_seed).as_quat()
            dataset.camera_trajectory[:, 4:] = np.random.normal(loc=0., scale=.1, size=(len(dataset), 3))

    optimiser = PoseOptimiser(
        dataset,
        feature_extraction_options=FeatureExtractionOptions(
            min_features=40,
            max_features=2048
        ),
        optimisation_options=OptimisationOptions(num_epochs=20000, learning_rate=1e-2, lr_scheduler_patience=50)
    )
    camera_trajectory, _, _ = optimiser.run(num_frames)

    if optimiser.debug_path:
        reconstruction_options = StaticMeshOptions(reconstruction_method=MeshReconstructionMethod.TSDFFusion,
                                                   sdf_num_voxels=80000000)
        log("Running TSDFFusion on initial pose data...")
        mesh_before = tsdf_fusion(dataset, options=reconstruction_options, num_frames=num_frames)
        mesh_before.export(pjoin(optimiser.debug_path, 'before.ply'))

        dataset.camera_trajectory = camera_trajectory
        log("Running TSDFFusion on final pose data..")
        mesh_after = tsdf_fusion(dataset, options=reconstruction_options, num_frames=num_frames)
        mesh_after.export(pjoin(optimiser.debug_path, 'after.ply'))


if __name__ == '__main__':
    main()
