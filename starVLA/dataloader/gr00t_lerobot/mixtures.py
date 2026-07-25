"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],

    "libero_plus": [
        ("libero_plus", 1.0, "libero_franka", "v3.0"),
    ],

    "libero_plus_4suite": [
        ("libero_plus_10", 1.0, "libero_franka"),
        ("libero_plus_goal", 1.0, "libero_franka"),
        ("libero_plus_object", 1.0, "libero_franka"),
        ("libero_plus_spatial", 1.0, "libero_franka"),
    ],

    "droid": [
        ("", 1.0, "libero_franka"),
    ],

    "fr3_realworld": [
        ("", 1.0, "fr3_real_world"),
    ],

    "trossen_subtask_combined": [
        ("", 1.0, "trossen_ai_stationary"),
    ],

    "ogrealman_canonical_v3": [
        ("", 1.0, "realman_bimanual", "v3.0"),
    ],

    "ogrealman_source_v3": [
        ("", 1.0, "realman_bimanual_source", "v3.0"),
    ],

    "ogrealman_source_no_base_v3": [
        ("", 1.0, "realman_bimanual_source_no_base", "v3.0"),
    ],

    "ogrealman_source_no_base_human_labelled_cloud_v3": [
        ("", 1.0, "realman_bimanual_source_no_base", "v3.0"),
    ],

    "magna_source_no_base_interventions_v3": [
        ("", 1.0, "realman_bimanual_source_no_base", "v3.0"),
    ],

    "magna_source_no_base_no_lift_interventions_v3": [
        ("", 1.0, "realman_bimanual_source_no_base_no_lift", "v3.0"),
    ],

    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_90": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
        # ("libero_90_no_noops_lerobot", 1.0, "libero_ur5"),
    ],

    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],

    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


}
