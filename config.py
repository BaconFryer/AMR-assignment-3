CONFIG = {
    "wind_active": True,
    "group_number": 18,
    "controller_type": "DO",
    
    "pid_gains": {
        "pos_y": {
            "kp": 0.55,
            "ki": 0.01,
            "kd": 0.35
        },
        "pos_x": {
            "kp": 0.11,
            "ki": 0.004,
            "kd": 0.16
        },
        "att": {
            "kp": 0.96,
            "ki": 0.0025,
            "kd": 0.4
        }
    },
    
    "base_throttle": 0.35,
    
    "rl_model_path": "./models/PPO_Drone_Kris_170m_steps.zip",
    
    "do_params": {
        "beta": 0.1
    },
    
    "max_control_angle": 1.0472,
    
    "throttle_limits": {
        "min": 0.0,
        "max": 1.0
    }
}