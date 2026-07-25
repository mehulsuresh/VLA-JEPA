import torch

from starVLA.dataloader.gr00t_lerobot.transform import rotation_ops
from starVLA.dataloader.gr00t_lerobot.transform.state_action import RotationTransform


def test_axis_angle_round_trip_preserves_matrix():
    axis_angle = torch.tensor([[0.2, -0.3, 0.5], [0.0, 0.0, 0.0]], dtype=torch.float64)

    matrix = rotation_ops.axis_angle_to_matrix(axis_angle)
    restored = rotation_ops.axis_angle_to_matrix(rotation_ops.matrix_to_axis_angle(matrix))

    assert torch.allclose(restored, matrix, atol=1e-10)


def test_quaternion_round_trip_preserves_matrix():
    quaternion = torch.tensor([[1.0, 0.2, -0.3, 0.4]], dtype=torch.float64)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)

    matrix = rotation_ops.quaternion_to_matrix(quaternion)
    restored = rotation_ops.quaternion_to_matrix(rotation_ops.matrix_to_quaternion(matrix))

    assert torch.allclose(restored, matrix, atol=1e-10)


def test_rotation_6d_round_trip_preserves_matrix():
    quaternion = torch.tensor([[1.0, -0.1, 0.5, 0.2]], dtype=torch.float64)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    matrix = rotation_ops.quaternion_to_matrix(quaternion)

    restored = rotation_ops.rotation_6d_to_matrix(rotation_ops.matrix_to_rotation_6d(matrix))

    assert torch.allclose(restored, matrix, atol=1e-10)


def test_euler_round_trip_preserves_matrix():
    euler = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float64)

    matrix = rotation_ops.euler_angles_to_matrix(euler, "XYZ")
    restored = rotation_ops.euler_angles_to_matrix(
        rotation_ops.matrix_to_euler_angles(matrix, "XYZ"),
        "XYZ",
    )

    assert torch.allclose(restored, matrix, atol=1e-10)


def test_rotation_transform_works_without_pytorch3d_dependency():
    quaternion = torch.tensor([[1.0, 0.2, -0.3, 0.4]], dtype=torch.float32)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)

    transform = RotationTransform("quaternion", "rotation_6d")

    assert transform.forward(quaternion).shape == (1, 6)
