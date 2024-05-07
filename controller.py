from stable_baselines3 import PPO

# Configuration
wind_active = True
group_number = 18
controller_type = "RL"
time = 0

# PID gains
PID_GAINS = {
    "pos_y": (0.55, 0.01, 0.35),
    "pos_x": (0.11, 0.004, 0.16),
    "att": (0.96, 0.0025, 0.4),
}
BASE_THROTTLE = 0.35

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class DisturbanceObserver:
    def __init__(self, beta):
        self.beta = beta
        self.disturbance_estimate = 0

    def update(self, control_input, system_output, model_output):
        self.disturbance_estimate = (1 - self.beta) * self.disturbance_estimate + self.beta * (system_output - model_output)
        return self.disturbance_estimate

class ObserverSystem:
    def __init__(self):
        self.pid_x = PIDController(*PID_GAINS["pos_x"])
        self.pid_y = PIDController(*PID_GAINS["pos_y"])
        self.pid_phi = PIDController(*PID_GAINS["att"])
        self.dob = DisturbanceObserver(0.1)
        self.prev_x = self.prev_y = self.prev_vx = self.prev_vy = 0

    def sim_step(self, state, target, dt):
        x, y, vx, vy, phi, ang_vel = state
        model_x = self.prev_x + self.prev_vx * dt
        model_y = self.prev_y + self.prev_vy * dt
        self.prev_x, self.prev_y, self.prev_vx, self.prev_vy = x, y, vx, vy
        disturb_x = self.dob.update(self.pid_x.update(target[0] - x, dt), x, model_x)
        disturb_y = self.dob.update(self.pid_y.update(target[1] - y, dt), y, model_y)
        return disturb_x, disturb_y

def pid_controller(state, target, dt):
    x, y, _, _, phi, _ = state
    x_err, y_err, phi_err = target[0] - x, -(target[1] - y), -phi
    y_control = pid_pos_y.update(y_err, dt)
    x_control = pid_pos_x.update(x_err, dt)
    phi_control = pid_phi.update(phi_err, dt)
    y_throttle = BASE_THROTTLE + y_control
    x_throttle = x_control
    u_1 = y_throttle + x_throttle + phi_control
    u_2 = y_throttle - x_throttle - phi_control
    return u_1, u_2

def rl_controller(state, target, dt):
    global model
    error_x, error_y = target[0] - state[0], target[1] - state[1]
    state = list(state)
    state[0], state[1] = error_x, error_y
    action = model.predict(tuple(state))[0]
    u_1 = action[0] / 160 if action[0] != 0 else 0
    u_2 = action[1] / 160 if action[1] != 0 else 0
    return u_1, u_2

def do_controller(state, target, dt):
    x, y, _, _, phi, _ = state
    phi = max(min(phi, 1.0472), -1.0472)
    x_err, y_err, phi_err = target[0] - x, -(target[1] - y), -phi
    disturb_x, disturb_y = control_system.sim_step(state, target, dt)
    y_control = pid_pos_y.update(y_err - disturb_y, dt)
    x_control = pid_pos_x.update(x_err - disturb_x, dt)
    phi_control = pid_phi.update(phi_err, dt)
    y_throttle = BASE_THROTTLE + y_control
    x_throttle = x_control
    u_1 = max(min(y_throttle + x_throttle + phi_control, 1.0), 0.0)
    u_2 = max(min(y_throttle - x_throttle - phi_control, 1.0), 0.0)
    return u_1, u_2

def controller(state, target, dt):
    global time
    if controller_type == "PID":
        u_1, u_2 = pid_controller(state, target, dt)
    elif controller_type == "RL":
        u_1, u_2 = rl_controller(state, target, dt)
    elif controller_type == "DO":
        u_1, u_2 = do_controller(state, target, dt)
    time += dt
    error_x, error_y = target[0] - state[0], target[1] - state[1]
    print(f"{controller_type}: u_1: {u_1:.3f}, u_2: {u_2:.3f}, time: {time:.3f}, error_x: {error_x:.3f}, error_y: {error_y:.3f}", end="\r")
    return u_1, u_2

# Initialize controllers and load RL model
pid_pos_y = PIDController(*PID_GAINS["pos_y"])
pid_pos_x = PIDController(*PID_GAINS["pos_x"])
pid_phi = PIDController(*PID_GAINS["att"])
control_system = ObserverSystem()

if controller_type == "RL":
    try:
        model = PPO.load("./models/PPO_Drone_Kris_170m_steps.zip")
    except FileNotFoundError:
        print("No model found.")