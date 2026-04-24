import argparse

import torch.nn as nn
from stable_baselines3 import SAC

from env.panda_env import PandaEnv


def _get_activation(name: str):
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Choose from: {', '.join(activations.keys())}")
    return activations[key]


def _build_parser():
    parser = argparse.ArgumentParser(description="Train SAC on PandaEnv with customizable MLP architecture")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer during training")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument(
        "--pi-layers",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Actor network hidden layers, e.g. --pi-layers 256 256 128",
    )
    parser.add_argument(
        "--qf-layers",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Critic network hidden layers, e.g. --qf-layers 512 512",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "elu", "gelu", "leaky_relu"],
        help="Activation function for both actor and critic MLPs",
    )
    parser.add_argument("--save-path", type=str, default="panda_sac", help="Path prefix for saved model")
    return parser


def main():
    args = _build_parser().parse_args()

    # ===== Create environment =====
    env = PandaEnv(render=args.render)

    # ===== Network structure =====
    policy_kwargs = dict(
        net_arch=dict(
            pi=args.pi_layers,
            qf=args.qf_layers,
        ),
        activation_fn=_get_activation(args.activation),
    )

    # ===== SAC model =====
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=50000,
        batch_size=128,
        learning_rate=3e-4,
        train_freq=1,
        gradient_steps=1,
        gamma=0.98,
    )

    # ===== Train =====
    model.learn(total_timesteps=args.timesteps)

    # ===== Save =====
    model.save(args.save_path)


if __name__ == "__main__":
    main()