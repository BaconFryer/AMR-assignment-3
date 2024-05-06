wind_active = True  # Select whether you want to activate wind or not
group_number = 18  # Enter your group number here

from stable_baselines3 import PPO
model = PPO.load("./models/PPO_Drone_Kris_80m_best.zip")
time = 0

def controller(state, target, dt):
    # state format: [position_x (m), position_y (m), velocity_x (m/s), velocity_y (m/s), attitude(radians), angular_velocity (rad/s)]
    # target format: [x (m), y (m)]
    # dt: time step
    # return: action format: (u_1, u_2)
    # u_1 and u_2 are the throttle settings of the left and right motor
    
    global model
    global time
    action = model.predict(state)[0] #deterministic=True)[0]

    u_1 = action[0] / 160 if action[0] != 0 else 0
    u_2 = action[1] / 160 if action[1] != 0 else 0
    
    error_x = target[0] - state[0]
    error_y = target[1] - state[1]
    errors = (error_x, error_y)
    
    time += dt
    
    print(f"u_1: {u_1}, u_2: {u_2}, time: {time}, error: {errors}", end="\r")


    return (u_1, u_2)