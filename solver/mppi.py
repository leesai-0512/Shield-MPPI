#!/usr/bin/env python3
import math
import numpy as np
import torch
import cuml
from cuml.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.nn.functional as F
import time
import json
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import animation

class MPPIController():
    def __init__(
        self,
        delta_t: float = 0.05,
        max_vx: float = 0.523,  # [m/s]
        max_wz: float = 2.000,  # [rad/s]
        time_step: int = 30,
        batch_size: int = 1000,
        sigma: torch.Tensor = torch.tensor([[0.7, 0.0], [0.0, 0.5]]),  # torch tensor
        ref_track_path: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 1.0]),  # torch tensor
        param_lambda: float = 50.0,
        param_alpha: float = 1.0,
        param_exploration: float = 0.0,
        stage_cost_weight: torch.Tensor = torch.tensor([50.0, 50.0, 1.0, 20.0]),  # torch tensor
        terminal_cost_weight: torch.Tensor = torch.tensor([50.0, 50.0, 1.0, 20.0]),  # torch tensor
        cbf_cost_weight: float = 1000,
        obstacles: torch.Tensor = torch.tensor([], dtype=torch.float32),  # 장애물 정보 (x, y, r)
        obstacles_types = ["circle","circle","circle"],
        w_stepsize: torch.Tensor = torch.tensor([50.0, 50.0]),
        robot_radius: float = 0.1,
        eta: float = 0.1,
        eps: float = 10.0,
        min_samples: int = 2,
        rho_nu_max: torch.Tensor = torch.tensor([0.5, 0.5]),
        rho_nu_min: torch.Tensor = torch.tensor([0.5, 0.5]),
        beta: float = 0.5,
        w_T: float = 3.0,
        visualize_optimal_traj=True,
        visualize_sampled_trajs=False,
        execute_dbscan=False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # specify device
) -> None:
        # mppi parameters
        self.device = device  # Set device (e.g., CPU or GPU)
        self.dim_x = 3  # dimension of system state vector
        self.dim_u = 2  # dimension of control input vector
        self.time_step = time_step  # prediction horizon
        self.batch_size = batch_size  # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha  # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi

        # Convert all inputs to tensors and move them to the specified device
        self.sigma = sigma.to(self.device)  # deviation of noise
        self.sigma_diag = self.sigma.diag()
        self.sigma_matrix = self.sigma.expand(self.batch_size, self.time_step, -1, -1)
        self.sigma_inv = torch.linalg.inv(self.sigma)
        self.sigma_inv_diag = self.sigma_inv.diag()  # shape: (2,)
        self.ref_track_path = ref_track_path.to(self.device)  # reference path
        self.stage_cost_weight = stage_cost_weight.to(self.device)
        self.terminal_cost_weight = terminal_cost_weight.to(self.device)
        self.robot_radius = robot_radius
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualize_sampled_trajs = visualize_sampled_trajs
        # vehicle parameters
        self.dt = delta_t #[s]
        self.max_vx = max_vx # [rad]
        self.max_wz = max_wz # [m/s^2]
        self.min_v = torch.tensor([-self.max_vx, -self.max_wz], device = self.device)
        self.max_v = torch.tensor([self.max_vx, self.max_wz], device = self.device)

        self.constraints = []  # 빈 리스트로 초기화
        self.opt_clustered_traj = []
        self.clustered_trajs = []

        self.S = torch.zeros((self.batch_size), device=self.device)
        self.stage_cost = torch.zeros((self.batch_size), device=self.device)
        self.terminal_cost = torch.zeros((self.batch_size), device=self.device)

        # mppi variables
        self.state = torch.tensor([0.0, -2.0, math.pi / 2], device=self.device)
        # MPPI 변수 PyTorch로 변환
        self.u_prev = torch.zeros((self.time_step, self.dim_u), device=self.device)
        self.u = torch.zeros((self.dim_u), device=self.device)
        self.x0 = torch.zeros((self.dim_x), device=self.device)
        self.u_prev = torch.zeros((self.time_step, self.dim_u), device=self.device)
        self.u = torch.zeros((time_step, self.dim_u), device=self.device)
        self.v = torch.zeros((batch_size,time_step,self.dim_u), device=self.device)
        self.noise = torch.zeros((batch_size,time_step,self.dim_u), device=self.device)
        self.opt_u = torch.zeros((time_step, self.dim_u), device=self.device)
        self.standard_normal_noise = torch.zeros(batch_size, time_step, 2, device=device)
        self.trajectory = torch.zeros((batch_size,time_step,self.dim_x), device=device)


        self.collision_cnt = 0
        self.start_idx = 0
        self.lab = 0
        self.dist = 0.0
        self.lab_check_idx=0
        self.prev_collision = False
        self.lab_check = False
        # ref_path info
        self.prev_waypoints_idx = 0
        self.obstacle_direction = 1 
        self.dbscan_elapsed_time = 0.0


        self.total_iter = 0
        self.beta=beta
        self.alpha = 1- self.beta
        self.w_T = w_T
        self.cbf_cost_weight = cbf_cost_weight



    def generateNoiseAndSampling(self):

        self.standard_normal_noise[:,:,0].normal_(0,self.sigma_diag[0])
        self.standard_normal_noise[:,:,1].normal_(0,self.sigma_diag[1])

        return  self.standard_normal_noise

    
    def predict_trajectory(self, initial_state, v):
        """
        2D 모바일 로봇의 예측 시뮬레이션 (for문 없이 벡터화)
        v: (batch_size, time_step, 2)
        initial_state: (3,) → (batch_size, 3)로 확장하여 시뮬레이션 진행
        """
        num_samples, num_timesteps, _ = v.shape
        trajectory = torch.empty((num_samples, num_timesteps, 3), device=self.device)
        vx = v[:,:,0]
        wz = v[:,:,1]
        # 초기 상태 확장
        x0 = initial_state.unsqueeze(0).expand(num_samples, -1)  # (batch_size, 3)

        # (1) 각도 theta 계산 (cumsum 활용)
        dtheta = wz * self.dt  # 마지막 입력값을 제외한 `u` 차이값 계산
        theta_cumsum = torch.cumsum(dtheta, dim=1)  # 누적합 계산

        trajectory[:, :, 2].copy_(x0[:, 2].unsqueeze(1) + theta_cumsum)
        trajectory[:, :, 2] = (trajectory[:, :, 2] + torch.pi) % (2 * torch.pi) - torch.pi

        theta_calc = torch.cat([x0[:, 2].unsqueeze(1), trajectory[:, :-1, 2]], dim=1)

        # (2) cos(theta), sin(theta) 계 산
        cos_theta, sin_theta = torch.cos(theta_calc), torch.sin(theta_calc)

        # (3) dx, dy 계산
        dx = vx * cos_theta * self.dt  # (batch_size, time_step)
        dy = vx * sin_theta * self.dt  # (batch_size, time_step)

        # (5) 최종 trajectory 조합 (x, y, theta)
        trajectory[:, :, 0].copy_(x0[:, 0].unsqueeze(1) + torch.cumsum(dx, dim=1))
        trajectory[:, :, 1].copy_(x0[:, 1].unsqueeze(1) + torch.cumsum(dy, dim=1))


        return trajectory
    
    
    def compute_total_cost(self, trajectory: torch.Tensor, v: torch.Tensor,ref_path) -> torch.Tensor:
        """
        스테이지 코스트와 터미널 코스트를 한 번에 계산하여 연산 속도를 최적화

        Args:
            trajectory (torch.Tensor): (batch_size, time_step, 3) - 각 샘플의 상태 벡터 [x, y, yaw]
            v (torch.Tensor): (batch_size, time_step, 2) - 각 샘플의 제어 입력 [v, omega]

        Returns:
            torch.Tensor: (batch_size,) - 샘플당 총 코스트
        """
        

        # 참고 웨이포인트 가져오기 (1회 호출)
        ref_x   = ref_path[..., 0]  # (B, T)
        ref_y   = ref_path[..., 1]
        ref_yaw = ref_path[..., 2]
        ref_v   = ref_path[..., 3]


        # ✅ torch.square() 사용하여 중복 연산 제거
        stage_cost = (
            self.stage_cost_weight[0] * torch.square(trajectory[:, :-1, 0] - ref_x[:,:-1])
            + self.stage_cost_weight[1] * torch.square(trajectory[:, :-1, 1] - ref_y[:,:-1])
            + self.stage_cost_weight[3] * torch.square(v[:, :-1, 0] - ref_v[:,:-1])
            + self.stage_cost_weight[2] * torch.square(trajectory[:, :-1, 2] - ref_yaw[:,:-1])
        ).sum(dim=1)  # (batch_size,)


        # 🎯 터미널 코스트 계산 (마지막 타임스텝만 고려)

        terminal_cost = (
            self.terminal_cost_weight[0] * torch.square(trajectory[:, -1, 0] - ref_x[:,-1])
            + self.terminal_cost_weight[1] * torch.square(trajectory[:, -1, 1] - ref_y[:,-1])
            + self.terminal_cost_weight[3] * torch.square(v[:,-1,0] - ref_v[:,-1])
            + self.terminal_cost_weight[2] * torch.square(trajectory[:, -1, 2] - ref_yaw[:,-1])
        )
        # 🎯 충돌 페널티를 한 번만 계산하여 적용
        total_collision_penalty = self.compute_collision_penalty(trajectory[:,:,:2],ref_path)  # (batch_size,)
        total_cost = stage_cost + terminal_cost + total_collision_penalty
        return total_cost
    


    
    def compute_collision_penalty(self, x,ref_path) -> torch.Tensor:

        is_batched = ref_path.ndim == 3
        if not is_batched:
            ref_xy = ref_path[:,:2]
        else:
            ref_xy = ref_path[:,:,:2]

        diff = ref_xy - x 
        half = self.w_T/2
        e_y_square = torch.sum(torch.square(diff),dim=-1)
        collision = (half*half - e_y_square)<0
        collision_penalty = self.cbf_cost_weight*collision.sum(dim=-1)
        return collision_penalty
    

    
    def compute_weights(self, S: torch.Tensor) -> torch.Tensor:
        # Softmax는 상수 offset에 대해 불변이므로, rho를 빼는 과정을 생략할 수 있습니다.
        return torch.softmax(-S / self.param_lambda, dim=0)
    
    def apply_constraint(self, u: torch.Tensor) -> torch.Tensor:
        u.clamp_(
            min=self.min_v,  # ✅ 리스트 사용 → device 설정 필요 없음
            max=self.max_v  # ✅ 리스트 사용 → device 설정 필요 없음
        )
        return u  # ✅ 기존 메모리를 유지하면서 값만 변경
    

    
    def set_zero(self):
        self.S.zero_()
        self.u.zero_()
        self.v.zero_()
        self.trajectory.zero_()


    def get_nearest_waypoints(self, trajectory, SEARCH_IDX_LEN=200, update_prev_idx=True):
        """
        현재 state 기준 최근접 ref 지점에서부터 탐색 시작 (loop-aware).
        trajectory: (B, T, 3), return: (B, T, 4)
        """
        is_batched = trajectory.ndim == 3
        if not is_batched:
            trajectory = trajectory.unsqueeze(0)  # (1, T, 3
        batch, time_step, _ = trajectory.shape
        N = self.ref_track_path.shape[0]  # 전체 참조 경로 길이

        # 1️⃣ 현재 위치 기준 최근접 참조 인덱스
        current_pos = self.state[:2]
        ref_xy_all = self.ref_track_path[:, :2]
        dists = torch.norm(ref_xy_all - current_pos[None, :], dim=1)
        self.lab_check_idx = torch.argmin(dists).item()

        self.start_idx = self.prev_waypoints_idx

        if(self.start_idx >=950):
            self.lab_check = True

        if(self.lab_check and self.lab_check_idx >=0 and self.lab_check_idx < 20):
            self.lab +=1
            self.lab_check = False
            print(f"lab",self.lab)

        
        # 2️⃣ 탐색 범위 슬라이싱 (폐곡선 처리)
        end_idx = self.start_idx + SEARCH_IDX_LEN
        if end_idx <= N:
            search_slice = self.ref_track_path[self.start_idx:end_idx]  # (SEARCH_IDX_LEN, 4)
        else:
            tail = self.ref_track_path[self.start_idx:]               # (N - start_idx, 4)
            head = self.ref_track_path[:end_idx % N]             # wrap-around (modulo)
            search_slice = torch.cat([tail, head], dim=0)        # (SEARCH_IDX_LEN, 4)

        ref_xy = search_slice[:, :2]  # (SEARCH_IDX_LEN, 2)

        # 3️⃣ 거리 계산
        query_xy = trajectory[:, :, :2]  # (B, T, 2)
        diff = query_xy[:, :, None, :] - ref_xy[None, None, :, :]  # (B, T, S, 2)
        dists_squared = torch.sum(diff ** 2, dim=-1)  # (B, T, S)

        # 4️⃣ 최근접 로컬 인덱스 → 글로벌 인덱스로 환산 (wrap-around 포함)
        nearest_local_idx = torch.argmin(dists_squared, dim=-1)  # (B, T)
        nearest_global_idx = (self.start_idx + nearest_local_idx) % N  # (B, T)

        # 5️⃣ 인덱싱 및 reshape
        flat_idx = nearest_global_idx.view(-1)               # (B*T,)
        ref_selected = self.ref_track_path[flat_idx]         # (B*T, 4)
        ref_path = ref_selected.view(batch, time_step, 4)

        # 6️⃣ 업데이트
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_global_idx[:, 0].min().item()
            # self.prev_waypoints_idx = self.start_idx

        return ref_path
    
    
    def check_collision(self):
        near_ref_path = self.ref_track_path[self.lab_check_idx]
        near_ref_xy = near_ref_path[:2]
        diff = self.state[:2]-near_ref_xy
        dist = torch.sum(diff**2,dim=-1)
        half = self.w_T/2
        self.dist = torch.sqrt(dist)
        if dist > half*half:
            
            if(self.prev_collision != True):
                self.collision_cnt+=1
                print(f"collision:",self.collision_cnt)
                self.prev_collision = True
                self.u_prev.zero_()
            return True
        else:
            self.prev_collision = False
            return False

    def compute_plot_data(self):

        optimal_traj = torch.zeros((self.time_step, self.dim_x), device=self.device)
        if self.visualize_optimal_traj:
            optimal_traj = self.predict_trajectory(self.state, self.u_prev.unsqueeze(0)).squeeze(0)

        return optimal_traj, self.opt_clustered_traj, self.clustered_trajs


 
    def compute_control_input(self,noise=None):

        S = self.S
        x0 = self.x0
        u = self.u
        v = self.v
        noise=self.noise
        opt_u = self.opt_u
        
        x0.copy_(self.state) 
        u.copy_(self.u_prev)
        noise.copy_(self.generateNoiseAndSampling())
        v.copy_(noise + u)
        v = self.apply_constraint(v)
        noise.copy_(v-u)
        trajectory = self.predict_trajectory(x0, v)
        ref_path = self.get_nearest_waypoints(trajectory,150,True)
        S += self.compute_total_cost(trajectory = trajectory, v = v, ref_path=ref_path)
        quad_term = torch.sum(u.unsqueeze(0) * self.sigma_inv_diag * v, dim=-1)
        S += self.param_gamma * quad_term.sum(dim=1)  # (batch_size,)
        w = self.compute_weights(S)
        w_expanded = w.view(-1, 1, 1)
        w_epsilon = torch.sum(w_expanded * noise, dim=0)
        opt_u.copy_(u + w_epsilon)

        self.u_prev.copy_(opt_u)

        
        return opt_u[0], opt_u, trajectory
    
    
    def set_state(self,state):
        self.state = state

    def set_ref(self, ref_path):
        self.ref_path = ref_path





