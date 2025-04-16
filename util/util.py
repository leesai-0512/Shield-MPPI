import math
import numpy as np
import json
import torch
import os

def get_log_dir(env_name: str, sample_num: int) -> str:
    base_dir = "data"
    folder_name = f"{env_name}_K={sample_num}"
    full_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def opt_traj_txt_data(frame, optimal_traj, log_dir):
    file_path = os.path.join(log_dir, "optimal_traj_log.txt")
    with open(file_path, "a") as f_opt:
        np.savetxt(f_opt, optimal_traj.cpu().numpy(), delimiter=",", fmt="%.6f", header=f"Frame {frame}", comments="")


def clustered_trajs_txt_data(frame, clustered_trajs, log_dir):
    file_path = os.path.join(log_dir, "clustered_traj_log.txt")
    with open(file_path, "a") as f_clu:
        clustered_traj_dict = {
            str(label): traj.cpu().numpy().tolist()
            for label, traj in clustered_trajs
        } if clustered_trajs else {}

        f_clu.write(json.dumps({
            "TimeStep": frame,
            "Clustered_Trajectory": clustered_traj_dict
        }) + "\n")


def sampled_trajs_txt_data(frame, sampled_trajs, log_dir):
    file_path = os.path.join(log_dir, "sampled_traj_log.txt")
    with open(file_path, "a") as f_samp:
        sampled_array = sampled_trajs.cpu().numpy().tolist()
        json.dump({"TimeStep": frame, "Sampled_Trajectory": sampled_array}, f_samp)
        f_samp.write("\n")



def frame_count_txt_data(frame_count, log_dir):
    with open(os.path.join(log_dir, "goal_reached.txt"), "a") as file:
        file.write(f"{frame_count}\n")

def path_length_txt_data(distance, log_dir):
    with open(os.path.join(log_dir, "total_distance_travelled.txt"), "a") as file:
        file.write(f"{distance}\n")

def total_time_txt_data(time, log_dir):
    with open(os.path.join(log_dir, "total_time.txt"), "a") as file:
        file.write(f"{time:.6f}\n")

def dbscan_time_txt_data(time, log_dir):
    with open(os.path.join(log_dir, "dbscan_time.txt"), "a") as file:
        file.write(f"{time:.6f}\n")

def optimal_input_txt_data(optimal_input, log_dir):
    with open(os.path.join(log_dir, "optimal_input.txt"), "a") as file:
        file.write(f"{optimal_input:.6f}\n")

def collision_num_txt_data(collision_cnt, log_dir):
    with open(os.path.join(log_dir, "collision_cnt.txt"), "a") as file:
        file.write(f"{collision_cnt:.6f}\n")

def error_txt_data(error, log_dir):
    with open(os.path.join(log_dir, "error.txt"), "a") as file:
        file.write(f"{error:.6f}\n")


def compute_next_state(x: torch.Tensor, u: torch.Tensor, dt) -> torch.Tensor:
    """
    로봇의 다음 상태 x_{k+1}을 계산하는 함수: x_{k+1} = f(x_k, u_k)

    Args:
        x (torch.Tensor): 현재 상태 벡터 (3,) - [x, y, theta]
        u (torch.Tensor): 제어 입력 벡터 (2,) - [v, omega]

    Returns:
        torch.Tensor: 다음 상태 벡터 (3,) - [x_next, y_next, theta_next]
    """
    # 상태 변수 파싱
    theta = x[2]  # 스칼라 값
    v, omega = u[0], u[1]  # 스칼라 값

    # 다음 상태 계산
    x_next = x[0] + v * torch.cos(theta) * dt
    y_next = x[1] + v * torch.sin(theta) * dt
    theta_next = theta + omega * dt

    # theta를 0~2*pi로 정규화
    theta_next = (theta_next + 2 * torch.pi) % (2 * torch.pi)

    # 다음 상태를 하나의 텐서로 결합
    next_state = torch.tensor([x_next, y_next, theta_next], device=u.device)  # (3,)
    return next_state

def goal_check(state,goal,tolerance):
    # 📝 목표 지점 도달 여부 체크
    distance_to_goal = torch.norm(state[:2] - goal[:2])
    theta_to_goal = torch.norm(state[2] - goal[2])

    return distance_to_goal < tolerance[0] and theta_to_goal < tolerance[1]


def compute_distance(new_state,previous_state):

    distance_travelled = torch.norm(new_state[:2] - previous_state[:2])  # 2D 거리 계산
    return distance_travelled.item()