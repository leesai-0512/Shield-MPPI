#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
import torch
import cuml
from cuml.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from plot.simenv1 import SIMULATIONENV1 as SIM
from solver.shieldmppi import MPPIController
import util.util as util
import torch.nn.functional as F
import time

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import animation


if __name__ == "__main__":
    # Matplotlib Figure and Axes 설정
    # Figure 초기화
    fig, ax = plt.subplots(figsize=(15, 15))
    

    # 로봇 및 MPPI 컨트롤러 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    env_name = "shield_mppi_wo_second_stage"
    batch_size = 30
    dt = 0.03
    state = torch.tensor([0.0, 0.0, -math.pi/6], device=device)
    radius  = 0.3
    log_dir = util.get_log_dir(env_name, batch_size)
    sim = SIM(state=state, radius=radius,dt=dt, device=device, ax=ax,fig=fig ,obstacles = None ,obstacles_types=None)
    mppi = MPPIController(
        delta_t=dt,  # [s]
        max_vx=10.0,  # [m/s]
        max_wz=5.0,  # [rad/s]
        time_step=30,  # 예측 타임스텝
        batch_size=batch_size,  # 배치 크기 (샘플 수)
        sigma=torch.tensor([[0.3, 0.0], [0.0, 1.0]], dtype=torch.float32),  # torch 텐서로 변환
        ref_track_path=sim.ref_path,  # torch 텐서로 변환 (사이즈: num_of_waypoints x 4)
        param_lambda=0.01,  # MPPI 람다 파라미터
        param_alpha=0.98,  # MPPI 알파 파라미터
        param_exploration=0.0,  # 탐험 파라미터
        stage_cost_weight=torch.tensor([50.0, 50.0, 0.0, 1], dtype=torch.float32),  # 스테이지 코스트 가중치
        terminal_cost_weight=torch.tensor([50.0, 50.0, 0.0, 1], dtype=torch.float32),  # 터미널 코스트 가중치
        cbf_cost_weight=10000,
        obstacles=None,
        obstacles_types=None,
        w_stepsize=None,
        robot_radius = radius,
        rho_nu_max = torch.tensor([0.4,0.4], dtype=torch.float32),
        rho_nu_min = torch.tensor([0.4,0.4], dtype=torch.float32),
        eta = 0.6,
        eps=10.0,
        min_samples=2,
        beta=0.1,
        w_T=2.0,
        visualize_optimal_traj=True,  # 최적 경로 시각화
        visualize_sampled_trajs=True,  # 샘플 경로 시각화
        execute_dbscan=None,
        device=device
    )


    # 📝 프레임(반복 횟수) 추적 변수 추가
    frame_count = 0
    total_distance = 0.0  # 로봇이 이동한 총 거리

    # 📝 이전 상태를 저장할 변수 (초기값은 로봇의 초기 위치)
    previous_state = state.clone()



    def update(frame):
        """
        Animation 업데이트 함수. 매 프레임마다 호출되어 로봇 상태와 경로를 업데이트합니다.
        """

        global frame_count, total_distance, previous_state
        frame_count += 1  # 프레임 카운트 증가

        mppi.set_state(sim.state)
        mppi.check_collision()
        start = time.perf_counter()
        
        optimal_input, optimal_input_sequence,sampled_trajectory = mppi.compute_control_input()
        total_elapsed_time = time.perf_counter() - start
        optimal_traj, opt_clustered_traj,clustered_trajs,before_optimal_traj = mppi.compute_plot_data()
        mppi.set_zero()
        new_state = util.compute_next_state(sim.state, optimal_input,dt)
        total_distance += util.compute_distance(new_state,previous_state)
        
        
        # 이전 상태를 현재 상태로 업데이트
        previous_state = new_state.clone()
        # 로봇 상태 업데이트
        sim.state = new_state
    
        sim.update_plot(optimal_traj=optimal_traj, predicted_traj=sampled_trajectory,opt_clustered_traj = None,
                        best_trajectory = before_optimal_traj,
                        clustered_trajs=dict((str(label), traj) for label, traj in clustered_trajs))
        
        util.optimal_input_txt_data(optimal_input[0],log_dir)
        util.error_txt_data(mppi.dist,log_dir)
        if frame_count != 1:
            util.total_time_txt_data(total_elapsed_time,log_dir)
            
        
        if mppi.lab == 30:
            util.path_length_txt_data(total_distance,log_dir)
            util.collision_num_txt_data(mppi.collision_cnt,log_dir)
            print("end")
            plt.close()

    # Animation 설정
    ani = FuncAnimation(fig, update, frames=None, interval=100, repeat=False, cache_frame_data=False)

    # 그래프 표시
    plt.show()