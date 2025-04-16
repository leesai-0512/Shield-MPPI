import torch
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import splprep, splev
class SIMULATIONENV1():
    def __init__(self, state: torch.Tensor = None, radius = 0.2 ,dt = 0.05,device: str = 'cuda' if torch.cuda.is_available() else 'cpu', ax=None,fig = None ,obstacles_types=None, obstacles = None) -> None:
        """
        Robot 초기화 함수.
        
        Args:
            state (torch.Tensor, optional): 초기 상태 벡터. 기본값은 [0.0, -2.0, pi/2].
            device (str, optional): 연산에 사용할 디바이스 ('cuda' 또는 'cpu'). 기본값은 'cuda' (가능하면).
            ax (matplotlib.Axes, optional): 로봇을 플로팅할 Axes 객체.
        """
        self.device = device
        # 초기 상태를 외부 입력으로 설정하거나 기본값 사용
        self.state = state if state is not None else torch.tensor([0.0, -2.0, math.pi / 2], device=self.device)
        self.dt = dt
        self.radius = radius
        # 플롯 설정
        self.ax = ax
        self.fig =fig
        self.obstacles_types = obstacles_types
        self.obstacles = obstacles
        self.obstacle_direction = 1 

        self.ref_path = self.generate_complex_track(num_points=1000,device=device,velocity=7.0)
        self.track_left, self.track_right = self.compute_track_boundaries(self.ref_path)
        if self.ax is not None:
            # 로봇 본체
            self.robot_circle = Circle((self.state[0].item(), self.state[1].item()), self.radius, color='blue')
            self.ax.add_patch(self.robot_circle)
            
            # 바퀴 설정
            self.wheel_width = 0.05
            self.wheel_height = 0.02
            self.wheel_offset = 0.12

            self.left_wheel = Rectangle((-self.wheel_width / 2, -self.wheel_height / 2),
                                        self.wheel_width, self.wheel_height, color='black')
            self.right_wheel = Rectangle((-self.wheel_width / 2, -self.wheel_height / 2),
                                         self.wheel_width, self.wheel_height, color='black')

            self.ax.add_patch(self.left_wheel)
            self.ax.add_patch(self.right_wheel)

            # 로봇 경로
            self.path_line, = self.ax.plot([], [], 'g-')  # 로봇 경로 선
            self.x_data, self.y_data = [], []


        fig.subplots_adjust(bottom=0.15, left=0.12)
        # X, Y 축 설정
        ax.set_xlim(-3.0, 16.0)
        ax.set_ylim(-3.0, 20.0)
        ax.set_aspect('equal')

        ax.tick_params(axis='both', which='major', labelsize=80, width=5, length=25, direction='out')
        ax.tick_params(axis='both', which='minor', labelsize=80, width=5, length=20, direction='out')

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        ax.set_xlabel("X [m]", fontsize=80, fontweight='bold', labelpad=20)
        ax.set_ylabel("Y [m]", fontsize=80, fontweight='bold', labelpad=5)

        for spine in ax.spines.values():
            spine.set_linewidth(5)
            spine.set_color("black")

        ax.grid(True, linewidth=5, color='black', alpha=0.5)

    def generate_complex_track(self,num_points=500, device='cuda', velocity=1.5):
        """
        트랙 폭을 고려해 겹치지 않도록 (비교적) 크게 루프를 돌면서
        내부로 들어갔다 나오는 식으로 구성한 예시 키포인트입니다.
        필요시 점의 위치를 수정하여 더 다양한 트랙을 얻을 수 있습니다.
        """
        # 아래 key_points는 바깥쪽 큰 루프 -> 안쪽 루프 -> 바깥쪽으로 나가며
        # 다시 시작점 주변으로 돌아오는 형태로 구성.
        # 왼쪽 아래 겹침이 덜하도록 마지막 부분의 좌표를 약간 수정한 버전입니다.
        key_points = np.array([
            [0,    0],    # 시작점 (좌하단)
            [4,    0],
            [8,    2],
            [10,   4],    # 추가: 완만한 상승을 위한 점
            [14,   7],
            [13,   8.5],  # 추가: [13,7]과 [11,10] 사이 부드러운 전환
            [10,   10],
            [8,   10],   # 추가: [11,10]과 [9,15] 사이 부드러운 전환
            [6,    11],

            [6.5,   13],  # 추가: [13,7]과 [11,10] 사이 부드러운 전환
            [9,   14],
            [13,   15],   # 추가: [11,10]과 [9,15] 사이 부드러운 전환
            [13,    17],

            [6,    16],
            [2,    15],
            [1,    14],
            [-1,    13],
            [1,    9],
            [3,    9],
            [5,    9],
            [6.5,    6],
            [5,    5],
            [0,    4],
            [-2,   4],
        ])

        # B-spline 보간 (per=True로 폐곡선)
        tck, _ = splprep([key_points[:, 0], key_points[:, 1]], s=0.5, per=True)
        u = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u, tck)

        # 진행 방향(yaw) 계산
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        yaw = np.arctan2(dy, dx)

        # 선속도(velocity) 일정
        v = np.ones_like(yaw) * velocity

        # PyTorch 텐서로 변환
        ref_path = torch.tensor(
            np.stack([x_smooth, y_smooth, yaw, v], axis=1),
            dtype=torch.float32,
            device=device
        )
        return ref_path
    
    def compute_track_boundaries(self,ref_path: torch.Tensor, track_width: float = 2.0):
        """
        ref_path: (N, 4) tensor -> [x, y, yaw, v]
        track_width: 전체 트랙 폭 (좌우 경계 각각 track_width / 2 만큼 떨어짐)

        return: left_boundary, right_boundary: (N, 2)
        """
        half_width = track_width / 2.0
        x = ref_path[:, 0]
        y = ref_path[:, 1]
        yaw = ref_path[:, 2]

        # 왼쪽은 +90도 방향, 오른쪽은 -90도 방향
        dx = torch.cos(yaw + math.pi / 2) * half_width
        dy = torch.sin(yaw + math.pi / 2) * half_width

        left = torch.stack([x + dx, y + dy], dim=1)
        right = torch.stack([x - dx, y - dy], dim=1)
        return left, right

    def update_plot(self, optimal_traj: torch.Tensor = None, predicted_traj: torch.Tensor = None, 
                best_trajectory: torch.Tensor = None,
                opt_clustered_traj: dict = None, before_optimal_traj: torch.Tensor = None,
                clustered_trajs: dict = None, bound_traj: dict = None, before_filter_traj: dict = None):
        """
        로봇의 현재 상태와 경로를 업데이트하는 함수.
        지나간 경로를 화면에 표시하지 않고, 로봇의 현재 위치와 새로운 데이터를 플롯합니다.
        
        Args:
            optimal_traj (torch.Tensor, optional): 최적 경로 (time_step, 3)
            predicted_traj (torch.Tensor, optional): 예측 경로 (batch_size, time_step, 3)
            best_trajectory (torch.Tensor, optional): 최적 샘플 경로 (time_step, 3)
            obstacles (torch.Tensor, optional): 장애물 정보 (num_obstacles, 3) - [obs_x, obs_y, obs_r]
            clustered_trajectory (dict, optional): 클러스터별 경로. 
                예: {"cluster_0": (time_step, dim_x), "cluster_1": (time_step, dim_x), ...}
        """

        if self.ax is None:
            return
        
        # 기존 플롯된 요소 삭제
        for artist in reversed(self.ax.patches + self.ax.lines):
            artist.remove()

        # 현재 로봇 위치와 방향 플롯
        x, y, theta = self.state
        self.robot_circle = Circle((x.item(), y.item()), self.radius, color='black')
        self.ax.add_patch(self.robot_circle)

        # 로봇의 방향 표시 (선)
        line_length = self.radius * 1.0  # 선의 길이
        line_x = x + line_length * torch.cos(theta)  # 선의 끝점 X
        line_y = y + line_length * torch.sin(theta)  # 선의 끝점 Y
        self.ax.plot(
            [x.item(), line_x.item()],  # 선의 시작점과 끝점 X 좌표
            [y.item(), line_y.item()],  # 선의 시작점과 끝점 Y 좌표
            color = 'red', linestyle = '-', linewidth=2  # 빨간색 선으로 표시
        )

        # 예측 경로 플롯 (Sampled Trajectories → label 한 번만 추가)
        if predicted_traj is not None:
            added_predicted_label = False
            for batch in predicted_traj:
                label = "Sampled Trajectories" if not added_predicted_label else None
                self.ax.plot(
                    batch[:, 0].cpu().numpy(),
                    batch[:, 1].cpu().numpy(),
                    'b-', alpha=0.5, linewidth=10.0, label=label
                )
                added_predicted_label = True

        
        
        # 전체 배치 샘플에서 클러스터링된 경로 플롯 (실선 스타일)
        if clustered_trajs is not None:
            cluster_colors_traj = plt.cm.get_cmap("Set1", len(clustered_trajs))  # 새로운 컬러맵 사용
            added_labels = set()  # 이미 추가된 라벨을 추적

            for idx, (label, traj) in enumerate(clustered_trajs.items()):
                color = 'gray' if label == "noise" else cluster_colors_traj(idx)  # 다른 컬러맵 적용
                legend_label = f"Sampled trajectory of {label}"  # 라벨 텍스트

                for path in traj:
                    # 해당 라벨이 추가되지 않았다면 추가하고, 그렇지 않으면 None
                    if legend_label not in added_labels:
                        self.ax.plot(
                            path[:, 0].cpu().numpy(),
                            path[:, 1].cpu().numpy(),
                            linestyle='-', color=color, alpha=0.7, label=legend_label, linewidth=10.0
                        )
                        added_labels.add(legend_label)  # 라벨을 추가한 목록에 저장
                    else:
                        self.ax.plot(
                            path[:, 0].cpu().numpy(),
                            path[:, 1].cpu().numpy(),
                            linestyle='-', color=color, alpha=0.7, linewidth=10.0
                        )
        

        # before clipping traj plot
        if before_optimal_traj is not None:
            self.ax.plot(
                before_optimal_traj[:, 0].cpu().numpy(),
                before_optimal_traj[:, 1].cpu().numpy(),
                'g-', label="Before Optimal Trajectory", alpha=0.7, linewidth=10.0  # 선 굵기를 2.5로 설정
            )


        # 최적 경로 플롯
        if optimal_traj is not None:
            self.ax.plot(
                optimal_traj[:, 0].cpu().numpy(),
                optimal_traj[:, 1].cpu().numpy(),
                linestyle='-', color='cyan', label="After Constraint Recovery Stage", linewidth=10.0
            )

        # 최적 경로 플롯
        if before_filter_traj is not None:
            self.ax.plot(
                before_filter_traj[:, 0].cpu().numpy(),
                before_filter_traj[:, 1].cpu().numpy(),
                linestyle='--', color='black', label="before_filter Trajectory",linewidth=10.0
            )

        # 경계 경로 플롯
        # if bound_traj is not None:
        #     for batch in bound_traj:
        #         self.ax.plot(
        #             batch[:, 0].cpu().numpy(),
        #             batch[:, 1].cpu().numpy(),
        #             linestyle='-', color='gray', label="bound Trajectory"
        #         )
            
        

        # 최적 샘플 경로 플롯
        if best_trajectory is not None:
            self.ax.plot(
                best_trajectory[:, 0].cpu().numpy(),
                best_trajectory[:, 1].cpu().numpy(),
                linestyle='--', color='gray', label="After CBF-augmented Stage",linewidth=10.0
            )

        # 가중평균 계산 후 클러스터별 경로 플롯 (점선 스타일)
        if opt_clustered_traj is not None:
            cluster_colors_trajectory = plt.cm.get_cmap("Set2", len(opt_clustered_traj))  # 다른 컬러맵 사용
            for idx, (cluster_label, cluster_traj) in enumerate(opt_clustered_traj.items()):
                color = cluster_colors_trajectory(idx)  # 다른 컬러맵 적용
                self.ax.plot(
                    cluster_traj[:, 0].cpu().numpy(),
                    cluster_traj[:, 1].cpu().numpy(),
                    linestyle='--', color=color, label=f"Optimal trajectory of {cluster_label}", linewidth=10.0
                )

        # 중앙선
        self.ax.plot(
            self.ref_path[:, 0].cpu().numpy(),
            self.ref_path[:, 1].cpu().numpy(),
            color='black', linestyle='--', linewidth=3, #label='Ref Path'
        )

        # 트랙 좌우 경계
        self.ax.plot(
            self.track_left[:, 0].cpu().numpy(),
            self.track_left[:, 1].cpu().numpy(),
            color='gray', linestyle='-', linewidth=3, #label='Left Boundary'
        )
        self.ax.plot(
            self.track_right[:, 0].cpu().numpy(),
            self.track_right[:, 1].cpu().numpy(),
            color='gray', linestyle='-', linewidth=3, #label='Right Boundary'
        )

        
        self.ax.legend(
            fontsize=40,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=0.,
            frameon=True  # 테두리 없애기
        )
        self.ax.figure.canvas.draw()
