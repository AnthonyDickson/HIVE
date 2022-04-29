import argparse
import enum
import json
import os.path
import shutil
import warnings
from typing import Tuple, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
from tqdm import tqdm

from Video2mesh.geometry import Quaternion
from Video2mesh.io import VTMDataset
from Video2mesh.options import StaticMeshOptions, MeshReconstructionMethod, COLMAPOptions
from Video2mesh.utils import tqdm_imap, log
from Video2mesh.video2mesh import Video2Mesh


# noinspection PyArgumentList
class FrameSamplingMode(enum.Enum):
    # All unique pairs of frames
    EXHAUSTIVE = enum.auto()
    # All consecutive pairs
    CONSECUTIVE = enum.auto()
    # Consecutive pairs + increasingly distant pairs.
    HIERARCHICAL = enum.auto()


FramePair = Tuple[int, int]
FramePairs = List[FramePair]


class FeatureData:
    """Encapsulates the frame index, the coordinates of the image features and the depth at those points."""

    def __init__(self, index, points, depth):
        self.index = index
        self.points = points
        self.depth = depth

    @classmethod
    def from_json(cls, data):
        index = data['index']
        points = data['points']
        depth = data['depth']

        return cls(index, points, depth)

    def to_json(self):
        return dict(index=self.index, points=self.points, depth=self.depth)


class FeatureDataTorch(torch.nn.Module, FeatureData):
    def __init__(self, index: torch.Tensor, points: torch.Tensor, depth: torch.Tensor):
        torch.nn.Module.__init__(self)
        FeatureData.__init__(self, index, points, depth)

        self.index = torch.nn.Parameter(index, requires_grad=False)
        self.points = torch.nn.Parameter(points, requires_grad=False)
        self.depth = torch.nn.Parameter(depth, requires_grad=False)

    @classmethod
    def from_json(cls, data):
        index = torch.tensor(data['index'])
        points = torch.tensor(data['points'])
        depth = torch.tensor(data['depth'])

        return cls(index, points, depth)

    def to_json(self):
        return dict(index=self.index.tolist(), points=self.points.tolist(), depth=self.depth.tolist())


class FeatureSet:
    """FeatureData for a pair of frames."""

    def __init__(self, frame_i: FeatureData, frame_j: FeatureData):
        self.frame_i = frame_i
        self.frame_j = frame_j

    @classmethod
    def load(cls, f):
        if isinstance(f, str):
            with open(f, 'r') as file:
                data = json.load(file)
        else:
            data = json.load(f)

        frame_i = FeatureData.from_json(data['frame_i'])
        frame_j = FeatureData.from_json(data['frame_j'])

        return cls(frame_i, frame_j)

    def save(self, f):
        frame_i = self.frame_i.to_json()
        frame_j = self.frame_j.to_json()

        if isinstance(f, str):
            with open(f, 'w') as file:
                json.dump(dict(frame_i=frame_i, frame_j=frame_j), file)
        else:
            json.dump(dict(frame_i=frame_i, frame_j=frame_j), f)


class FeatureExtractor:
    """Extracts correspondences between image pairs and the depth at those correspondences."""

    def __init__(self, dataset: VTMDataset, frame_pairs: FramePairs,
                 min_features: int = 20, max_features: Optional[int] = None,
                 ignore_dynamic_objects: bool = True):
        """

        :param dataset: The RGB-D dataset to use.
        :param frame_pairs: The pairs of frames to extract matching image features from.
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

        self.dataset = dataset
        self.frame_pairs = frame_pairs
        self.min_features = min_features
        self.max_features = max_features
        self.ignore_dynamic_objects = ignore_dynamic_objects

        self.debug_path: Optional[str] = None
        self.match_viz_path: Optional[str] = None
        self.frame_pairs_path: Optional[str] = None
        self.feature_set_path: Optional[str] = None

        self.frames: Optional[List[np.ndarray]] = None
        self.depth_maps: Optional[List[np.ndarray]] = None
        self.masks: Optional[List[np.ndarray]] = None

        self.sift = cv2.SIFT_create(nfeatures=max_features)

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
        self._setup_folders()

        if os.path.isfile(self.feature_set_path):
            log(f"Found cached feature set at: {self.feature_set_path}")
            return FeatureSet.load(self.feature_set_path)

        self.frames, self.depth_maps, self.masks = self._get_frame_data()

        index_i = []
        points_i = []
        depth_i = []
        index_j = []
        points_j = []
        depth_j = []
        num_good_frame_pairs = 0

        log(f"Extracting matching image feature info...")

        for feature_set in tqdm_imap(self._get_image_features, self.frame_pairs):
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

        self._print_results_stats(index_i, index_j, num_good_frame_pairs)

        full_feature_set = FeatureSet(FeatureData(index_i, points_i, depth_i), FeatureData(index_j, points_j, depth_j))
        full_feature_set.save(self.feature_set_path)

        return full_feature_set

    def _setup_folders(self):
        """
        Create a folder to save debug output.
        """
        self.debug_path = os.path.join(self.dataset.base_path, 'pose_optim')
        self.match_viz_path = os.path.join(self.debug_path, 'match_viz')
        self.frame_pairs_path = os.path.join(self.debug_path, 'frame_pairs.txt')
        self.feature_set_path = os.path.join(self.debug_path, 'feature_set.json')

        os.makedirs(self.debug_path, exist_ok=True)

        if os.path.isfile(self.frame_pairs_path):
            cached_frame_pairs = np.loadtxt(self.frame_pairs_path)
            same_frame_pairs = len(cached_frame_pairs) == len(self.frame_pairs) and \
                               np.all(cached_frame_pairs == self.frame_pairs)

            if not same_frame_pairs:
                shutil.rmtree(self.debug_path)
                os.makedirs(self.debug_path)
                # noinspection PyTypeChecker
                np.savetxt(self.frame_pairs_path, self.frame_pairs)

        if os.path.isdir(self.match_viz_path):
            shutil.rmtree(self.match_viz_path)

        os.makedirs(self.match_viz_path)

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

        return FeatureSet(FeatureData(i, points_i, depth_i), FeatureData(j, points_j, depth_j))

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

    def _filter_matches_ransac(self, points_i, points_j, depth_i, depth_j, matches_mask) -> \
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

        _, mask = cv2.findHomography(points_i, points_j, cv2.RANSAC)

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

        plt.imsave(os.path.join(self.match_viz_path, f"{i:06d}-{j:06d}.jpg"), kp_matches_viz)

    def _print_results_stats(self, index_i, index_j, num_good_frame_pairs):
        """
        Print some statistics about the matched image features.
        :param index_i: The indices of the matched frames for the first frame of the pair.
        :param index_j: The indices of the matched frames for the second frame of the pair.
        :param num_good_frame_pairs: The number of frame pairs kept.
        """
        frames_with_matched_features = set(index_i + index_j)
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

        log(f"Found {len(chunks)} groups consecutive of frames.")


class FixedParameters(torch.nn.Module):
    """The parameters used in the optimisation process that do not change, e.g. matched feature points, depth."""

    def __init__(self, camera_matrix: np.ndarray, feature_set: FeatureSet, dtype=torch.float32):
        """
        :param camera_matrix: The 3x3 camera intrinsics matrix.
        :param feature_set: The paired set of image feature data (frame index, match coordinates, depth).
        :param dtype: The data type to convert the parameters to.
        """
        super().__init__()

        def to_tensor(x, data_type=dtype):
            x = np.asarray(x)
            x = torch.from_numpy(x)
            x = x.to(data_type)

            return x

        def to_tensor_feature_data(feature_data: FeatureData) -> FeatureDataTorch:
            return FeatureDataTorch(index=to_tensor(feature_data.index, data_type=torch.long),
                                    points=to_tensor(feature_data.points),
                                    depth=to_tensor(feature_data.depth))

        self.camera_matrix = to_tensor(camera_matrix)

        self.frame_i = to_tensor_feature_data(feature_set.frame_i)
        self.frame_j = to_tensor_feature_data(feature_set.frame_j)


class OptimisationParameters(torch.nn.Module):
    """The parameters subject to optimisation (camera poses)."""

    def __init__(self, initial_camera_poses: np.ndarray, dtype=torch.float32):
        """
        :param initial_camera_poses: The initial guess for the camera poses where each row is in the
            format [r_x, r_y, r_z, r_w, t_x, t_y, t_z] where r is a quaternion and t is a 3D position vector.
        :param dtype: The data type to convert the parameters to.
        """
        super().__init__()

        def to_param(a):
            a = torch.tensor(a, dtype=dtype)
            return torch.nn.Parameter(a, requires_grad=True)

        r, t = initial_camera_poses[:, :4], initial_camera_poses[:, 4:]
        self.rotation_quaternions = to_param(r)
        self.translation_vectors = to_param(t)

    def __len__(self):
        """
        :return: The number of optimisation parameters.
        """
        return sum([p.nelement() for p in self.parameters()])

    def to_trajectory(self) -> np.ndarray:
        """Convert the optimisation parameters to a (N, 7) NumPy array."""
        r = self.rotation_quaternions.detach().cpu().numpy()
        t = self.translation_vectors.detach().cpu().numpy()

        r = r / np.linalg.norm(r, ord=2, axis=1).reshape((-1, 1))

        return np.hstack((r, t))


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



class PoseOptimiser:
    def __init__(self, dataset: VTMDataset):
        self.dataset = dataset

    def run(self, frame_sampling=FrameSamplingMode.HIERARCHICAL, mask_features=True,
            min_features_per_frame=20, max_features_per_frame: Optional[int] = 2048, num_epochs=10000):
        frame_pairs = self.sample_frame_pairs(frame_sampling)
        feature_set = self.extract_feature_points(frame_pairs, min_features_per_frame, max_features_per_frame,
                                                  mask_features)
        fixed_parameters = FixedParameters(self.dataset.camera_matrix.copy(), feature_set)
        optimisation_parameters = OptimisationParameters(self.dataset.camera_trajectory.copy())
        optimised_camera_trajectory = self.optimise_pose_data(fixed_parameters, optimisation_parameters,
                                                              learning_rate=1e-1, num_epochs=num_epochs)
        interpolated_trajectory = self.interpolate_poses_without_matches(feature_set, optimised_camera_trajectory)

        return interpolated_trajectory

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

    def extract_feature_points(self, frame_pairs: FramePairs, min_features: int, max_features: int,
                               mask_features: bool):
        feature_extractor = FeatureExtractor(self.dataset, frame_pairs, min_features, max_features, mask_features)

        return feature_extractor.extract_feature_points()

    def optimise_pose_data(self, fixed_parameters: FixedParameters,
                           optimisation_parameters: OptimisationParameters,
                           num_epochs=10000,
                           learning_rate=1e-4, min_loss_delta=1e-4,
                           lr_scheduler_patience=250, early_stopping_patience=500) -> np.ndarray:
        fixed_parameters.cuda()
        optimisation_parameters.cuda()

        optimiser = torch.optim.Adam(optimisation_parameters.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=lr_scheduler_patience, mode='min',
                                                                  threshold=min_loss_delta, threshold_mode='abs')
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_difference=min_loss_delta)

        progress_bar = tqdm(range(num_epochs))

        for _ in progress_bar:
            optimiser.zero_grad()
            residuals = self.calculate_residuals(fixed_parameters, optimisation_parameters)
            # noinspection PyTypeChecker
            loss = torch.mean(torch.linalg.norm(residuals, ord=2, dim=0))
            loss.backward()

            # TODO: Focus training on aligning pairs initially, later expanding it to all frames.
            # Can mask gradients as such: optimisation_parameters.translation_vectors.grad[row_to_mask, :] = 0.0

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

        return optimisation_parameters.to_trajectory()

    def calculate_residuals(self, fixed_parameters: FixedParameters,
                            optimisation_parameters: OptimisationParameters) -> torch.Tensor:
        p = self.project_to_world_coords(fixed_parameters.frame_i,
                                         fixed_parameters.camera_matrix,
                                         optimisation_parameters)
        q = self.project_to_world_coords(fixed_parameters.frame_j,
                                         fixed_parameters.camera_matrix,
                                         optimisation_parameters)

        return p - q
        # q_ = self.project_to_image_coords(p, fixed_parameters.frame_j, fixed_parameters.camera_matrix,
        #                                   optimisation_parameters)
        #
        # return fixed_parameters.frame_j.points.T - q_

    def project_to_world_coords(self, frame_data: FeatureDataTorch, camera_matrix: torch.Tensor,
                                pose_data: OptimisationParameters) -> torch.Tensor:
        indices = frame_data.index
        points = frame_data.points
        depth = frame_data.depth

        u, v = points.T
        f_x, f_y, c_x, c_y = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

        points_camera_space = torch.vstack(
            (
                (u - c_x) * depth / f_x,
                (v - c_y) * depth / f_y,
                depth
            )
        )

        r = pose_data.rotation_quaternions[indices].T
        t = pose_data.translation_vectors[indices].T
        points_world_space = Quaternion(r).normalise().conjugate().apply(points_camera_space - t)

        return points_world_space

    def project_to_image_coords(self, points_world_space: torch.Tensor, frame_data: FeatureDataTorch, camera_matrix: torch.Tensor,
                                pose_data: OptimisationParameters) -> torch.Tensor:
        indices = frame_data.index

        r = pose_data.rotation_quaternions[indices].T
        t = pose_data.translation_vectors[indices].T
        points_camera_space = Quaternion(r).normalise().conjugate().apply(points_world_space + t)

        x, y, z = points_camera_space
        f_x, f_y, c_x, c_y = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
        points_image_space = torch.vstack((f_x * x + c_x * z, f_y * y + c_y * z)) / z

        return points_image_space

    def interpolate_poses_without_matches(self, feature_set: FeatureSet, optimised_camera_trajectory: np.ndarray):
        frames_with_matched_features = set(feature_set.frame_i.index + feature_set.frame_j.index)

        chunks = []
        chunk = []

        for frame_index in range(self.dataset.num_frames):
            if frame_index not in frames_with_matched_features:
                chunk.append(frame_index)
            elif chunk:
                chunks.append(chunk)
                chunk = []

        if chunk:
            chunks.append(chunk)

        interpolated_poses = optimised_camera_trajectory.copy()

        for chunk in chunks:
            start = chunk[0] - 1
            end = chunk[-1] + 1

            start_rotation = optimised_camera_trajectory[start, :4]
            end_rotation = optimised_camera_trajectory[end, :4]
            start_position = optimised_camera_trajectory[start, 4:]
            end_position = optimised_camera_trajectory[end, 4:]

            key_frame_times = [0, 1]
            times_to_interpolate = np.linspace(0, 1, num=len(chunk) + 2)

            slerp = Slerp(times=key_frame_times, rotations=Rotation.from_quat([start_rotation, end_rotation]))
            lerp = interp1d(key_frame_times, [start_position, end_position], axis=0)

            interpolated_poses[start:end + 1, 4:] = lerp(times_to_interpolate)
            interpolated_poses[start:end + 1, :4] = slerp(times_to_interpolate).as_quat()

        return interpolated_poses


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
    camera_trajectory = optimiser.run(min_features_per_frame=40, max_features_per_frame=2048, num_epochs=100000)
    # TODO: Complete docstrings.
    # TODO: Cache image feature data and frame pairs used to generate the data
    # TODO: Saved optimised trajectory
    # TODO: Create mesh with TSDFFusion using the original trajectory
    # TODO: Create mesh with TSDFFusion using the optimised trajectory
    reconstruction_options = StaticMeshOptions(reconstruction_method=MeshReconstructionMethod.TSDF_FUSION,
                                               sdf_num_voxels=4000000)
    mesh_before = Video2Mesh._create_static_mesh(dataset, options=reconstruction_options)

    camera_trajectory_backup = dataset.camera_trajectory.copy()
    dataset.camera_trajectory = camera_trajectory

    mesh_after = Video2Mesh._create_static_mesh(dataset, options=reconstruction_options)

    dataset.camera_trajectory = camera_trajectory_backup

    mesh_before.export('before.ply')
    mesh_after.export('after.ply')


if __name__ == '__main__':
    main()