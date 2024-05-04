wind_active = True  # Select whether you want to activate wind or not
group_number = 18  # Enter your group number here

def controller(state, target, dt):
    # state format: [position_x (m), position_y (m), velocity_x (m/s), velocity_y (m/s), attitude(radians), angular_velocity (rad/s)]
    # target format: [x (m), y (m)]
    # dt: time step
    # return: action format: (u_1, u_2)
    # u_1 and u_2 are the throttle settings of the left and right motor
    
    global model
    action = model.predict(state, deterministic=True)[0]
    u_1, u_2 = action[0] * 0.025, action[1] * 0.025
    
    print(f"u_1: {u_1}, u_2: {u_2}", end="\r")

    return (u_1, u_2)