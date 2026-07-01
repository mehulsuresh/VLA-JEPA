import collections
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
import re
from pathlib import Path
import requests
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.model2libero_interface import M1Inference


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_libero_gripper_command(gripper_val: np.ndarray | float) -> np.ndarray:
    """LIBERO / robosuite convention: -1 opens, +1 closes."""
    arr = np.asarray(gripper_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 if v > 0.0 else -1.0
    return np.asarray([bin_val], dtype=np.float32)

def _strip_libero_plus_task_metadata(task_description: str) -> str:
    task_description = task_description.strip()
    metadata_patterns = (
        r"\s+view\s+.*$",
        r"\s+noise\s+\d+\s*$",
        r"\s+(?:tb|table|light)\s+\d+\s*$",
        r"\s+level\d+\s+sample\d+\s*$",
        r"\s+add\s+\d+\s*$",
    )
    cleaned = task_description
    for pattern in metadata_patterns:
        cleaned = re.sub(pattern, "", cleaned).strip()
    return cleaned

import hashlib

def short_name(text, max_len=80):
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    clean = text.replace(" ", "_")[:max_len]
    return f"{clean}_{h}"

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    category_value: str = "Background Textures"
            #Background Textures
        #Camera Viewpoints
        #Language Instructions
        #Light Conditions
        #Objects Layout
        #Robot Initial States
        #Sensor Noise

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    with_state: str = "true"

    job_name: str = "test"

    max_tasks: int | None = None
    task_start: int = 0
    max_steps_override: int | None = None
    task_language_filter: str | None = None
    num_ddim_steps: int = 10
    action_execution_mode: str = "receding"
    action_ensemble: bool = True
    action_ensemble_horizon: int | None = None
    adaptive_ensemble_alpha: float = 0.1
    skip_existing: bool = False


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[args.task_suite_name](category_value=args.category_value)
    except TypeError:
        task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 250  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10" or args.task_suite_name == "libero_mix":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.max_steps_override is not None:
        max_steps = int(args.max_steps_override)

    model = M1Inference(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        num_ddim_steps=args.num_ddim_steps,
        action_execution_mode=args.action_execution_mode,
        action_ensemble=args.action_ensemble,
        action_ensemble_horizon=args.action_ensemble_horizon,
        adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    )


    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_ids = list(range(num_tasks_in_suite))
    if args.task_language_filter:
        task_language_filter = args.task_language_filter.strip()
        task_ids = [
            task_id
            for task_id in task_ids
            if _strip_libero_plus_task_metadata(task_suite.get_task(task_id).language) == task_language_filter
        ]
        if not task_ids:
            raise ValueError(
                f"No tasks in suite {args.task_suite_name!r} matched canonical task "
                f"language {task_language_filter!r}."
            )
    if args.max_tasks is not None:
        start = max(0, int(args.task_start))
        end = min(num_tasks_in_suite, start + max(0, int(args.max_tasks)))
        task_ids = task_ids[start:end]

    for task_id in tqdm.tqdm(task_ids):
        episode_indices = list(range(args.num_trials_per_task))
        if args.skip_existing:
            episode_indices = [
                episode_idx
                for episode_idx in episode_indices
                if not _rollout_video_exists(args.video_out_path, task_id, episode_idx)
            ]
            if not episode_indices:
                logging.info("Skipping task %04d because all rollout videos already exist.", task_id)
                continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
            horizon=max_steps + args.num_steps_wait + 1,
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(episode_indices):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan = collections.deque()

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0
            
            # full_actions = np.load("./debug/action.npy")
            
            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = { # 
                    "observation.primary": np.expand_dims(
                        img, axis=0
                    ),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API
                obs_input = {
                    "images": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "task_description": observation["instruction"][0],  
                    "step": step,
                }

                if args.with_state == "true":
                    obs_input["state"] = observation["observation.state"]

                start_time = time.time()
                
                response = model.step(**obs_input) 
                
                end_time = time.time()
                # print(f"time: {end_time - start_time}")
                
                # # 
                raw_action = response["raw_action"]
                
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_libero_gripper_command(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(f"Unexpected action sizes: "
                                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                    f"Falling back to LIBERO_DUMMY_ACTION.")
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)
                
                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                try:
                    obs, reward, done, info = env.step(delta_action.tolist())
                except ValueError as exc:
                    if "terminated episode" not in str(exc):
                        raise
                    logging.warning(
                        "Environment was already terminated before the next action; "
                        "ending rollout as a failure and saving replay."
                    )
                    done = False
                    break
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = short_name(task_description.replace(" ", "_"))
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_task{task_id:04d}_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)
            
            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


def _rollout_video_exists(video_out_path: str, task_id: int, episode_idx: int) -> bool:
    out = pathlib.Path(video_out_path)
    pattern = f"rollout_task{task_id:04d}_*_episode{episode_idx}_*.mp4"
    return any(out.glob(pattern))


def _get_libero_env(task, resolution, seed, horizon: int | None = None):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = _strip_libero_plus_task_metadata(task.language)
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if not task_bddl_file.exists():
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / "libero_mix"
            / task.bddl_file
        )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if horizon is not None:
        env_args["horizon"] = int(horizon)
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    if os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        start_debugpy_once()
    tyro.cli(eval_libero)
