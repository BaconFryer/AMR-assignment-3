wind_active = False  # Select whether you want to activate wind or not
group_number = 27  # Enter your group number here


# Implement a controller
#def controller(state, target_pos, dt):
    # state format: [position_x (m), position_y (m), velocity_x (m/s), velocity_y (m/s), attitude(radians), angular_velocity (rad/s)]
    # target_pos format: [x (m), y (m)]
    # dt: time step
    # return: action format: (u_1, u_2)
    # u_1 and u_2 are the throttle settings of the left and right motor

    #action = (0, 0)
    #return action

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

def controller(state, target_pos, dt):
    # PID parameters for position control
    kp, ki, kd = 0.98, 0.03, 0.4  # Tune these values as per your requirements

    # PID parameters for velocity control
    kv_y = 0  # Velocity damping coefficient for y, tune this value
    kv_x = 0  # Velocity damping coefficient for x, tune this value

    # Initialize static PID controllers if not already done
    global pid_x, pid_y, pid_attitude
    try:
        pid_x
        pid_y
        pid_attitude
    except NameError:
        pid_x = PIDController(kp, ki, kd)
        pid_y = PIDController(kp, ki, kd)
        pid_attitude = PIDController(6, 10, 3.3) # (kp*16, ki*35, kd*8) (12, 7, 6.4)

    # Calculate errors
    error_x = target_pos[0] - state[0]  # Horizontal position error
    error_y = target_pos[1] - state[1]  # Vertical position error
    error_attitude = -state[4]  # Attitude error (maintain level flight, target = 0)

    # Use PID controllers to compute control actions
    control_x = pid_x.compute(error_x, dt)
    control_y = -pid_y.compute(error_y, dt)  # Invert control_y for correct throttle response
    control_attitude = pid_attitude.compute(error_attitude, dt)

    # Throttle calculations
    velocity_damping_y = kv_y * -state[3]  # Damping proportional to the vertical velocity
    velocity_damping_x = kv_x * -state[2]  # Damping proportional to the horizontal velocity
    
    base_throttle = max(0, min(1, 0.5 + control_y - velocity_damping_y))  # Adjust base throttle to maintain y position and dampen as necessary

# Adjust throttles based on attitude correction, horizontal movement, and x-velocity damping
    u_1 = base_throttle + control_attitude + control_x - velocity_damping_x
    u_2 = base_throttle - control_attitude - control_x + velocity_damping_x


    # Ensure motor throttles are within bounds
    u_1 = max(0, min(1, u_1))
    u_2 = max(0, min(1, u_2))
    print(state)
    action = (u_1, u_2)
    return action
