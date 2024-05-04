wind_active = True  # Select whether you want to activate wind or not
group_number = 18  # Enter your group number here
base = 0.35

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

# PID gains
# Y Position control
kp_pos_y = 0.55
ki_pos_y = 0.01
kd_pos_y = 0.35
# X Position control
kp_pos_x = 0.11
ki_pos_x = 0.004
kd_pos_x = 0.16
# Attitude control
kp_att = 0.96
ki_att = 0.0025
kd_att = 0.4 #0.3

# PID controllers - x, y, attitude
pid_pos_y = PIDController(kp_pos_y, ki_pos_y, kd_pos_y)
pid_pos_x = PIDController(kp_pos_x, ki_pos_x, kd_pos_x)
pid_phi = PIDController(kp_att, ki_att, kd_att)
# pid_ang_vel = PIDController(kp_ang_vel, ki_ang_vel, kd_ang_vel)

t = 0.0

def controller(state, target, dt):
    global t
    # state format: [position_x (m), position_y (m), velocity_x (m/s), velocity_y (m/s), attitude(radians), angular_velocity (rad/s)]
    # target format: [x (m), y (m)]
    # dt: time step
    # return: action format: (u_1, u_2)
    # u_1 and u_2 are the throttle settings of the left and right motor

    # State unpacking
    x = state[0]
    y = state[1]
    phi = state[4]

    # Error calculation
    x_err = target[0] - x
    y_err = -(target[1] - y)
    phi_err = -phi

    # PID update
    y_control = pid_pos_y.update(y_err, dt)
    x_control = pid_pos_x.update(x_err, dt)
    phi_control = pid_phi.update(phi_err, dt)
    
    y_throttle = base + y_control
    x_throttle = x_control

    # Throttle calculations
    u_1 = y_throttle + x_throttle + phi_control
    u_2 = y_throttle - x_throttle - phi_control
    
    t += dt
    print(f"u_1: {round(u_1, 5)}, u_2: {round(u_2, 5)}, t: {round(t, 2)}, x_err: {x_err}, y_err: {y_err}", end="\r")
    #y: {round(y_throttle, 5)}, x: {round(x_throttle, 5)}, phi: {round(phi_control, 5)}, 
    
    u_1 = max(min(u_1, 1.0), 0.0)
    u_2 = max(min(u_2, 1.0), 0.0)

    return (u_1, u_2)