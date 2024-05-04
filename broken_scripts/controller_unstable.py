wind_active = False  # Select whether you want to activate wind or not
group_number = 18  # Enter your group number here
G = 0.344264

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
        
        print(f"Error: {error}, Integral: {self.integral}, Derivative: {derivative}")
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def controller(state, target_pos, dt):
    # state format: [position_x (m), position_y (m), velocity_x (m/s), velocity_y (m/s), attitude(radians), angular_velocity (rad/s)]
    # target_pos format: [x (m), y (m)]
    # dt: time step
    # return: action format: (u_1, u_2)
    # u_1 and u_2 are the throttle settings of the left and right motor

    x_pos = state[0]
    y_pos = state[1]
    x_vel = state[2]
    y_vel = state[3]
    att = state[4]
    ang_vel = state[5]

    x_pos_err = target_pos[0] - x_pos
    y_pos_err = target_pos[1] - y_pos
    
    # Position control PID gains
    kp_pos = 0.5
    ki_pos = 0.05
    kd_pos = 0.025

    # Velocity control PID gains
    kp_vel = 0.5
    ki_vel = 0.05
    kd_vel = 0.025
    
    # Attitude control PID gains
    kp_att = 0.5
    ki_att = 0.01
    kd_att = 0.01
    
    # Angular velocity control PID gains
    kp_ang_vel = 2.0
    ki_ang_vel = 0.1
    kd_ang_vel = 0.025

    pid_x_pos = PIDController(kp_pos, ki_pos, kd_pos)
    pid_y_pos = PIDController(kp_vel, ki_vel, kd_vel)
    pid_x_vel = PIDController(kp_vel, ki_vel, kd_vel)
    pid_y_vel = PIDController(kp_vel, ki_vel, kd_vel)
    pid_att = PIDController(kp_att, ki_att, kd_att)
    pid_ang_vel = PIDController(kp_ang_vel, ki_ang_vel, kd_ang_vel)

    base = G

    # Calculate the desired velocities based on position errors
    desired_x_vel = pid_x_pos.update(x_pos_err, dt)
    desired_y_vel = pid_y_pos.update(y_pos_err, dt)

    # Calculate the velocity errors
    x_vel_err = desired_x_vel - x_vel
    y_vel_err = desired_y_vel - y_vel

    # Apply velocity feedback
    x_throttle = pid_x_vel.update(x_vel_err, dt)
    y_throttle = pid_y_vel.update(y_vel_err, dt)
    
    # Calculate the desired attitude based on x-velocity error
    desired_att = pid_x_vel.update(x_vel_err, dt)
    
    # Limit the desired attitude to a reasonable range
    desired_att = max(min(desired_att, 0.2), -0.2)
    att_err = desired_att - att
    desired_ang_vel = pid_att.update(att_err, dt)
    desired_ang_vel = max(min(desired_ang_vel, 1.0), -1.0)
    
    # Calculate the attitude error
    ang_vel_err = desired_ang_vel - ang_vel
    
    # Apply attitude feedback
    ang_throttle = pid_ang_vel.update(ang_vel_err, dt)

    # Combine the throttle values
    u_1 = base - y_throttle + ang_throttle
    u_2 = base - y_throttle - ang_throttle
    
    # Limit the throttle values to a valid range
    u_1 = max(min(u_1, 1.0), 0.0)
    u_2 = max(min(u_2, 1.0), 0.0)

    action = (u_1, u_2)
    return action