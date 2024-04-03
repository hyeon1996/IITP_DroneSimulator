import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument(
        '--project',
        type=str,
        default='Cooperative',
        help='The project which env is in')
    parser.add_argument(
        '--scenario', type=str, default='3_robots', help='the map of the game')
    parser.add_argument(
        '--difficulty',
        type=str,
        default='7',
        help='Difficulty of the environment.')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument(
        '--total_steps', type=int, default=10000000, help='total episode')
    parser.add_argument(
        '--replay_buffer_size',
        type=int,
        default=5000,
        help='Max number of episodes stored in the replay buffer.')
    parser.add_argument(
        '--memory_warmup_size',
        type=int,
        default=32,
        help="Learning start until replay_buffer_size >= 'memory_warmup_size'")
    parser.add_argument(
        '--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./cooperative_marl_ros/src/robot_control/scripts/work_dirs',
        help='result directory of the policy',
    )
    parser.add_argument(
        '--logger',
        type=str,
        default='wandb',
        help='the logger for the experiment')
    parser.add_argument(
        '--train_log_interval',
        type=int,
        default=5,
        help='Log interval(Eposide) for training')
    parser.add_argument(
        '--test_log_interval',
        type=int,
        default=20,
        help='Log interval for testing.')
    parser.add_argument(
        '--test_steps',
        type=int,
        default=100,
        help="Evaluate the model every 'test_steps' steps.")
    parser.add_argument(
        '--load_model',
        type=bool,
        default=False,
        help='whether to load the pretrained model',
    )
    parser.add_argument(
        '--stats',
        type=str,
        default='',
        help='the stats file for data normalization')
    parser.add_argument(
        '--delta_time', type=float, default=1, help='delta time per step')
    parser.add_argument(
        '--step_mul',
        type=int,
        default=8,
        help='how many steps to make an action')
    parser.add_argument(
        '--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    return args
