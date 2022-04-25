import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from Video2mesh.geometry import Quaternion


class QuaternionTests(unittest.TestCase):
    def test_normalise(self):
        q = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True)
        quat = Quaternion(q.as_quat().T)

        # SciPy normalises rotations by default.
        q_norm = q.as_rotvec()
        quat_norm = quat.normalize()
        quat_norm = Rotation.from_quat(quat_norm.values.T).as_rotvec()

        np.testing.assert_allclose(q_norm, quat_norm)

    def test_conjugate(self):
        q = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True)
        q_conjugate = q.inv()

        quat = Quaternion(q.as_quat().T)
        quat_conjugate = quat.conjugate().values.T
        quat_conjugate = Rotation.from_quat(quat_conjugate)

        # Have to use .as_rotvec(...) since SciPy's Rotation inverts quaternions by negating the scalar w, instead of
        # the xyz components.
        np.testing.assert_allclose(q_conjugate.as_rotvec(), quat_conjugate.as_rotvec())

    def test_multiplying_by_conjugate_gives_identity(self):
        q = Rotation.from_euler('xyz', [[90, 0, 0]], degrees=True)
        quat = Quaternion(q.as_quat().T)

        quat_by_conj = (quat * quat.conjugate()).values

        np.testing.assert_allclose(np.array([[0.], [0.], [0.], [1.]]), quat_by_conj)

    def test_multiplication(self):
        q1 = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True)
        q2 = Rotation.from_euler('xyz', [[45, 0, 0], [0, 45, 0], [0, 0, 45]], degrees=True)

        result_scipy = (q1 * q2).as_rotvec()

        result = Quaternion(q1.as_quat().T) * Quaternion(q2.as_quat().T)
        result = Rotation.from_quat(result.values.T).as_rotvec()

        np.testing.assert_allclose(result_scipy, result)

    def test_rotating_vector(self):
        q1 = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, 90]], degrees=True).as_quat().T
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

        np.testing.assert_allclose(Rotation.from_quat(q1.T).apply(v1.T), Quaternion(q1).apply(v1).T)
        np.testing.assert_allclose(Rotation.from_quat(q1.T).apply(v2.T), Quaternion(q1).apply(v2).T)
        np.testing.assert_allclose(Rotation.from_quat(q1.T).apply(v3.T), Quaternion(q1).apply(v3).T)


if __name__ == '__main__':
    unittest.main()
