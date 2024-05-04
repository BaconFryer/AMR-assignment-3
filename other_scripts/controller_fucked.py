# Constants and Configuration
wind_active = False # Select whether you want to activate wind or not
group_number = 18 # Enter your group number here
G = 0.5 # Gravity compensation factor

class PID:
    def __init__(self, Kp, Ki, Kd, threshold=1) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.err_int = 0.0
        self.threshold = threshold

    def calculate(self, state, target, diff, dt):
        err = state - target
        self.err_int = max(min(self.err_int + err * dt, self.threshold), -self.threshold)
        return self.Kp * err + self.Ki * self.err_int + self.Kd * diff
        
# y = PID(20, 1, 10, 0.5)
y = PID(5, 0.5, 5)
x = PID(3, 0.4, 4, 3)
angle = PID(20, 1, 3, 2)

def controller(state, target_pos, dt):
    # state format: [position_x (m), position_y (m), velocity_x (m/s), velocity_y (m/s), attitude(radians), angular_velocity (rad/s)]
    # target_pos format: [x (m), y (m)]
    # dt: time step
    # return: action format: (u_1, u_2)
    global G, y, x, angle
    thrust = y.calculate(state[1], target_pos[1], state[3], dt) 
    pitch = x.calculate(state[0], target_pos[0], state[2], dt) + angle.calculate(state[4], 0, state[5], dt)
    return (thrust - pitch, thrust + pitch)