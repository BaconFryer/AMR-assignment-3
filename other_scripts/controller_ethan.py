wind_active = False  # Select whether you want to activate wind or not
group_number = 18  # Enter your group number here

def controller(state, target_pos, dt):
    # PID parameters for position control
    kp_pos, ki_pos, kd_pos = 0.98, 0.09, 0.3  # Tune these values as per your requirements
    kp_att, ki_att, kd_att = 6, 10, 3.3  # Tune these for attitude control

    # Unpack state and target positions
    position_x, position_y, velocity_x, velocity_y, attitude, angular_velocity = state
    target_x, target_y = target_pos

    # Initialize or update static variables for integral and previous errors
    if not hasattr(controller, "integral_x"):
        controller.integral_x = controller.integral_y = controller.integral_att = 0
        controller.previous_error_x = controller.previous_error_y = controller.previous_error_att = 0

    # Calculate errors
    error_x = target_x - position_x
    error_y = position_y - target_y
    error_attitude = -attitude  # Assuming target attitude is zero for level flight

    # Update integrals
    controller.integral_x += error_x * dt
    controller.integral_y += error_y * dt
    controller.integral_att += error_attitude * dt

    # Calculate derivatives
    derivative_x = (error_x - controller.previous_error_x) / dt
    derivative_y = (error_y - controller.previous_error_y) / dt
    derivative_att = (error_attitude - controller.previous_error_att) / dt

    # Compute PID outputs for x, y, and attitude
    control_x = kp_pos * error_x + ki_pos * controller.integral_x + kd_pos * derivative_x
    control_y = kp_pos * error_y + ki_pos * controller.integral_y + kd_pos * derivative_y
    control_attitude = kp_att * error_attitude + ki_att * controller.integral_att + kd_att * derivative_att

    # Apply velocity damping (if applicable)
    velocity_damping_y = 0 * velocity_y  # Adjust as necessary
    velocity_damping_x = 0 * velocity_x  # Adjust as necessary

    # Base throttle calculation adjusted for vertical position control
    base_throttle = max(0, min(1, 0.5 + control_y - velocity_damping_y))

    # Adjust throttles based on attitude correction, horizontal movement, and x-velocity damping
    u_1 = base_throttle + control_attitude + control_x - velocity_damping_x
    u_2 = base_throttle - control_attitude - control_x + velocity_damping_x

    # Ensure motor throttles are within bounds
    u_1 = max(0, min(1, u_1))
    u_2 = max(0, min(1, u_2))

    # Update previous errors for the next calculation
    controller.previous_error_x = error_x
    controller.previous_error_y = error_y
    controller.previous_error_att = error_attitude

    return (u_1, u_2)