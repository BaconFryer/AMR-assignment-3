wind_active = True # Select whether you want to activate wind or not
group_number = 18 # Enter your group number here
base = 0.35
 
class PIDController:
    def __init__(self, kp, ki, kd,threshold=30):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.threshold=threshold
        self.prev_error = 0
        self.integral = 0
 
    def update(self, error, dt):
        self.integral += error * dt
        self.integral = max(min(self.integral, self.threshold), -self.threshold)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
 
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# 扰动观测器类
class DisturbanceObserver:
    def __init__(self, beta):
        # 初始化扰动观测器的参数
        self.beta = beta  # 滤波系数
        self.disturbance_estimate = 0  # 扰动估计值

    def update(self, control_input, system_output, model_output):
        # 更新扰动估计值
        self.disturbance_estimate = (1 - self.beta) * self.disturbance_estimate + self.beta * (system_output - model_output)
        return self.disturbance_estimate


class ControlSystem:
    def __init__(self):
        self.pid_x = PIDController(0.15, 0.01, 0.18)
        self.pid_y = PIDController(0.5, 0.01, 0.35)
        self.pid_phi = PIDController(0.98, 0.005, 0.32)

        # 初始化扰动观测器
        self.dob = DisturbanceObserver(0.1)
        self.base = 0.35  # 控制信号的基础值
        # 初始化前一状态
        self.prev_x = 0  # x位置
        self.prev_y = 0  # y位置
        self.prev_vx = 0  # x速度
        self.prev_vy = 0  # y速度
    def simulate_step(self, state, target, dt):
        # 模拟控制系统的一个步骤
        x, y, vx, vy, phi, ang_vel = state

        # 基于前一状态的模型预测（假设没有扰动）
        model_x = self.prev_x + self.prev_vx * dt
        model_y = self.prev_y + self.prev_vy * dt

        # 更新前一状态
        self.prev_x, self.prev_y, self.prev_vx, self.prev_vy = x, y, vx, vy

        # 估计扰动
        disturbance_x = self.dob.update(self.pid_x.update(target[0] - x, dt), x, model_x)
        disturbance_y = self.dob.update(self.pid_y.update(target[1] - y, dt), y, model_y)

        return disturbance_x,disturbance_y


# PID gains
# Y Position control
kp_pos_y = 0.5
ki_pos_y = 0.01
kd_pos_y = 0.35
# X Position control
kp_pos_x = 0.15
ki_pos_x = 0.01
kd_pos_x = 0.18
# Attitude control
kp_att = 0.98
ki_att = 0.005
kd_att = 0.32
 
# PID controllers - x, y, attitude
pid_pos_y = PIDController(kp_pos_y, ki_pos_y, kd_pos_y)
pid_pos_x = PIDController(kp_pos_x, ki_pos_x, kd_pos_x)
pid_phi = PIDController(kp_att, ki_att, kd_att)
# pid_ang_vel = PIDController(kp_ang_vel, ki_ang_vel, kd_ang_vel)
control_system = ControlSystem()

 
def controller(state, target, dt):
    # state format: [position_x (m), position_y (m), velocity_x (m/s), velocity_y (m/s), attitude(radians), angular_velocity (rad/s)]
    # target format: [x (m), y (m)]
    # dt: time step
    # return: action format: (u_1, u_2)
    # u_1 and u_2 are the throttle settings of the left and right motor
 
    # State unpacking
    x = state[0]
    y = state[1]
    phi = state[4]


    #max 60
    #use maxium control angle to pull back in the opposite direction
    if abs(phi)>1.0472: #math.radians(60)
        if phi>0:
            phi=-1*1.0472
        if phi<0:
            phi=1.0472

    # Error calculation
    x_err = target[0] - x
    y_err = -(target[1] - y)
    phi_err = - phi

    disturbance_x,disturbance_y=control_system.simulate_step(state, target, dt)
    print(disturbance_x)
    print(disturbance_y)
 

    # PID update
    y_control = pid_pos_y.update(y_err-disturbance_y, dt)
    x_control = pid_pos_x.update(x_err-disturbance_x, dt)
    phi_control = pid_phi.update(phi_err, dt)
 
    y_throttle = base + y_control
    x_throttle = x_control


    # Throttle calculations
    
    u_1 = y_throttle + x_throttle + phi_control
    u_2 = y_throttle - x_throttle - phi_control


    print(f"y: {round(y_throttle, 5)}, x: {round(x_throttle, 5)}, phi: {round(phi_control, 5)}, u_1: {round(u_1, 5)}, u_2: {round(u_2, 5)}", end="\r")
 
    u_1 = max(min(u_1, 1.0), 0.0)
    u_2 = max(min(u_2, 1.0), 0.0)
 
    action = (u_1, u_2)
    return action