# VLA-JEPA Architecture

This diagram reflects the current training pipeline in [`VLA_JEPA.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/framework/VLA_JEPA.py) with the active single-GPU training config in [`vlajepa_robot_ft_trossen_vjepa21_small_5090_lerobot.yaml`](/home/mehul/work/vjepa/VLA-JEPA/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_5090_lerobot.yaml).

## Executive Summary

- The model has two trainable heads on top of frozen perceptual backbones.
- `Qwen 3.5 VL` converts images + language into multimodal token features.
- `V-JEPA encoder` converts multi-view videos into latent video tokens.
- `VJ predictor` learns a world-model objective by predicting future video latents.
- `GR00T action head` learns a policy objective by predicting future robot actions.
- Training jointly optimizes:
  - `action_loss`
  - `wm_loss`

## Training Diagram

```mermaid
flowchart LR
    classDef frozen fill:#e8f0ff,stroke:#4b6cb7,stroke-width:1.5px,color:#102040;
    classDef trainable fill:#eaf8ea,stroke:#2f7d32,stroke-width:1.5px,color:#103010;
    classDef input fill:#fff6dd,stroke:#b78900,stroke-width:1.2px,color:#4d3b00;
    classDef loss fill:#fdeaea,stroke:#b23b3b,stroke-width:1.5px,color:#5a1010;
    classDef output fill:#f4ecff,stroke:#7d4db3,stroke-width:1.2px,color:#30104d;

    subgraph Inputs["Inputs Per Sample"]
        I1["3 camera views for Qwen<br/>RGB images + prompt text"]:::input
        I2["3 camera views x 8 frames<br/>384x384 video clip"]:::input
        I3["Robot state"]:::input
        I4["Target future action chunk"]:::input
        I5["RA-BC metadata<br/>progress / mistake labels"]:::input
    end

    subgraph QwenPath["Language + Image Path"]
        Q1["Qwen 3.5 VL Interface<br/>multimodal feature extractor"]:::frozen
        Q2["Extract action token embeddings"]:::output
        Q3["Extract embodied action embeddings"]:::output
    end

    subgraph VideoPath["Video World-Model Path"]
        V1["Frozen V-JEPA Encoder<br/>multi-view video latent tokens"]:::frozen
        V2["Split latent tokens into:<br/>past context states + future target states"]:::output
        V3["VJ Predictor<br/>VisionTransformerPredictorAC"]:::trainable
        L1["wm_loss<br/>L1(predicted future states,<br/>ground-truth future states)"]:::loss
    end

    subgraph ActionPath["Policy Path"]
        A1["Repeat target chunk N times<br/>repeated_diffusion_steps = 4"]:::output
        A2["GR00T Action Head<br/>state encoder + action encoder + DiT"]:::trainable
        A3["Per-sample flow-matching loss"]:::output
        A4["Optional RA-BC weighting"]:::output
        L2["action_loss"]:::loss
    end

    subgraph Final["Joint Optimization"]
        F1["total_loss =<br/>1.0 * action_loss + 0.1 * wm_loss"]:::loss
    end

    I1 --> Q1
    Q1 --> Q2
    Q1 --> Q3

    I2 --> V1
    V1 --> V2
    V2 -->|"past latent states"| V3
    Q2 -->|"action token conditioning"| V3
    V3 -->|"predicted future latent states"| L1
    V2 -->|"future latent states"| L1

    I4 --> A1
    A1 --> A2
    I3 --> A2
    Q3 -->|"policy conditioning"| A2
    A2 --> A3
    I5 --> A4
    A3 --> A4
    A4 --> L2

    L1 --> F1
    L2 --> F1
```

## Inference Diagram

```mermaid
flowchart LR
    classDef frozen fill:#e8f0ff,stroke:#4b6cb7,stroke-width:1.5px,color:#102040;
    classDef trainable fill:#eaf8ea,stroke:#2f7d32,stroke-width:1.5px,color:#103010;
    classDef input fill:#fff6dd,stroke:#b78900,stroke-width:1.2px,color:#4d3b00;
    classDef output fill:#f4ecff,stroke:#7d4db3,stroke-width:1.2px,color:#30104d;

    I1["Current images + instruction"]:::input --> Q1["Qwen 3.5 VL"]:::frozen
    I2["Current robot state"]:::input --> A1["GR00T Action Head"]:::trainable
    Q1 --> Q2["Embodied action embeddings"]:::output
    Q2 --> A1
    A1 --> O1["4 denoising / Euler steps<br/>from Gaussian noise to action chunk"]:::output
    O1 --> O2["Predicted future action sequence"]:::output
```

## Current Freeze / Train Status

- Frozen:
  - `qwen_vl_interface.model`
  - `vj_encoder`
- Trainable:
  - `vj_predictor`
  - `action_model`

## Main Data Shapes

- Qwen inputs:
  - multi-image prompt per sample
- Video input:
  - `B x V x T x C x H x W`
  - current config uses `V=3`, `T=8`
- Action head target:
  - future action window of `7` steps
- Inference denoising:
  - `num_inference_timesteps = 4`
