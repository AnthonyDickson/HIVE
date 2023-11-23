#  HIVE, creates 3D mesh videos.
#  Copyright (C) 2023 Anthony Dickson anthony.dickson9656@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest

import numpy as np
from numpy import asarray as to_npy
from scipy.spatial.transform import Rotation
from torch import tensor as to_tensor

from hive.geometric import Quaternion


def quat_to_scipy(quat: Quaternion) -> Rotation:
    return Rotation.from_quat(to_npy(quat.values.T))


def scipy_to_quat(rotation: Rotation) -> Quaternion:
    return Quaternion(to_tensor(rotation.as_quat().T))


class QuaternionTests(unittest.TestCase):
    def test_normalise(self):
        q = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True)
        # SciPy normalises rotations by default.
        q_norm = q.as_rotvec()

        quat = scipy_to_quat(q)
        quat_norm = quat_to_scipy(quat.normalise()).as_rotvec()

        np.testing.assert_allclose(q_norm, quat_norm)

    def test_conjugate(self):
        q = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True)
        q_conjugate = q.inv()

        quat = scipy_to_quat(q)
        quat_conjugate = quat_to_scipy(quat.conjugate())

        # Have to use .as_rotvec(...) since SciPy's Rotation inverts quaternions by negating the scalar w, instead of
        # the xyz components.
        np.testing.assert_allclose(q_conjugate.as_rotvec(), quat_conjugate.as_rotvec())

    def test_multiplying_by_conjugate_gives_identity(self):
        q = Rotation.from_euler('xyz', [[90, 0, 0]], degrees=True)
        quat = scipy_to_quat(q)

        quat_by_conj = to_npy((quat * quat.conjugate()).values)

        np.testing.assert_allclose(np.array([[0.], [0.], [0.], [1.]]), quat_by_conj)

    def test_multiplication(self):
        r1 = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True)
        r2 = Rotation.from_euler('xyz', [[45, 0, 0], [0, 45, 0], [0, 0, 45]], degrees=True)

        result_scipy = (r1 * r2).as_rotvec()

        result = scipy_to_quat(r1) * scipy_to_quat(r2)
        result = quat_to_scipy(result).as_rotvec()

        np.testing.assert_allclose(result_scipy, result)

    def test_rotating_vector(self):
        r1 = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True)
        v1 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        v2 = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ])
        v3 = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ])

        np.testing.assert_allclose(r1.apply(v1.T),
                                   to_npy(scipy_to_quat(r1).apply(to_tensor(v1))).T)
        np.testing.assert_allclose(r1.apply(v2.T),
                                   to_npy(scipy_to_quat(r1).apply(to_tensor(v2))).T)
        np.testing.assert_allclose(r1.apply(v3.T),
                                   to_npy(scipy_to_quat(r1).apply(to_tensor(v3))).T)


if __name__ == '__main__':
    unittest.main()
