from __future__ import annotations

import torch
import torch.nn.functional as F


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return torch.stack(flat, dim=-1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.shape[-1] != 3:
        raise ValueError(f"Invalid euler angle shape: {euler_angles.shape}")
    if len(convention) != 3 or len(set(convention)) != 3:
        raise ValueError(f"Invalid euler convention: {convention}")
    matrices = [
        _axis_angle_rotation(axis, angle)
        for axis, angle in zip(convention, torch.unbind(euler_angles, dim=-1), strict=True)
    ]
    return matrices[0] @ matrices[1] @ matrices[2]


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape: {matrix.shape}")
    if len(convention) != 3 or len(set(convention)) != 3:
        raise ValueError(f"Invalid euler convention: {convention}")

    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise ImportError("matrix_to_euler_angles fallback requires scipy.") from exc

    original_shape = matrix.shape[:-2]
    matrix_np = matrix.detach().cpu().reshape(-1, 3, 3).numpy()
    angles = Rotation.from_matrix(matrix_np).as_euler(convention)
    return torch.as_tensor(angles, dtype=matrix.dtype, device=matrix.device).reshape(original_shape + (3,))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape: {quaternions.shape}")
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / quaternions.square().sum(-1).clamp_min(torch.finfo(quaternions.dtype).eps)

    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive = x > 0
    ret[positive] = torch.sqrt(x[positive])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape: {matrix.shape}")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    flr = torch.tensor(0.1, dtype=matrix.dtype, device=matrix.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    quat = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(dtype=torch.bool),
        :,
    ].reshape(matrix.shape[:-2] + (4,))
    return torch.where(quat[..., :1] < 0, -quat, quat)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return torch.cat((torch.cos(half_angles), axis_angle * sin_half_angles_over_angles), dim=-1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    if d6.shape[-1] != 6:
        raise ValueError(f"Invalid rotation_6d shape: {d6.shape}")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape: {matrix.shape}")
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))
