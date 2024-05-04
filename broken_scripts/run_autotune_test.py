from src.environment import Environment
import controller_unstable
import pygame_gui
import pygame
import csv
import sys
import importlib
import pathlib
import asyncio
import threading

# Some fucked up global vars
best_kp_pos = 0.25
best_ki_pos = 0.0
best_kd_pos = 0.05
best_kp_vel = 0.1
best_ki_vel = 0.01
best_kd_vel = 0.02

targets = []
with open("targets.csv", "r") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        if (
            float(row[0]) > 8
            or float(row[1]) > 8
            or float(row[0]) < 0
            or float(row[1]) < 0
        ):
            print(
                "WARNING: Target outside of environment bounds (0, 0) to (8, 8), not loading target"
            )
        else:
            targets.append((float(row[0]), float(row[1])))

environment = Environment(
    render_mode="human",
    render_path=True,
    screen_width=1000,
    ui_width=200,
    rand_dynamics_seed=controller_unstable.group_number,
    wind_active=controller_unstable.wind_active,
)

running = True
target_pos = targets[0]


theme_path = pathlib.Path("src/theme.json")
manager = pygame_gui.UIManager(
    (environment.screen_width, environment.screen_height), theme_path
)

reset_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((800, 0), (200, 50)),
    text="Reset",
    manager=manager,
)

wind_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((800, 50), (200, 50)),
    text="Toggle Wind",
    manager=manager,
)

autotune_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((800, 100), (200, 50)),
    text="Autotune",
    manager=manager,
)

target_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((800, 700), (200, 50)),
    text="Target: " + str(target_pos),
    manager=manager,
)

prev_target_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((800, 750), (100, 50)),
    text="Prev",
    manager=manager,
)

next_target_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((900, 750), (100, 50)),
    text="Next",
    manager=manager,
)


def reload():
    # re importing the controller module without closing the program
    try:
        importlib.reload(controller_unstable)
        environment.reset(controller_unstable.group_number, controller_unstable.wind_active)

    except Exception as e:
        print("Error reloading controller.py")
        print(e)


def check_action(unchecked_action):
    # Check if the action is a tuple or list and of length 2
    if isinstance(unchecked_action, (tuple, list)):
        if len(unchecked_action) != 2:
            print(
                "WARNING: Controller returned an action of length "
                + str(len(unchecked_action))
                + ", expected 2"
            )
            checked_action = (0, 0)
            pygame.quit()
            sys.exit()
        else:
            checked_action = unchecked_action

    else:
        print(
            "WARNING: Controller returned an action of type "
            + str(type(unchecked_action))
            + ", expected list or tuple"
        )
        checked_action = (0, 0)
        pygame.quit()
        sys.exit()

    return checked_action

def run_autotune():
    global best_kp_pos, best_ki_pos, best_kd_pos, best_kp_vel, best_ki_vel, best_kd_vel
    target_pos = targets[0]
    dt = 1 / 60

    best_kp_pos, best_ki_pos, best_kd_pos, best_kp_vel, best_ki_vel, best_kd_vel \
        = controller_unstable.autotune_pid(environment, target_pos, dt)

    print("Best PID gains:")
    print("Position control:")
    print("kp_pos:", best_kp_pos)
    print("ki_pos:", best_ki_pos)
    print("kd_pos:", best_kd_pos)
    print("Velocity control:")
    print("kp_vel:", best_kp_vel)
    print("ki_vel:", best_ki_vel)
    print("kd_vel:", best_kd_vel)
    
autotune_flag = False

# Game loop
while running:
    time_delta = environment.clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == reset_button:
                reload()
            if event.ui_element == wind_button:
                environment.toggle_wind()
            if event.ui_element == prev_target_button:
                target_pos = targets[targets.index(target_pos) - 1]
                target_label.set_text("Target: " + str(target_pos))
            if event.ui_element == next_target_button:
                target_pos = targets[(targets.index(target_pos) + 1) % len(targets)]
                target_label.set_text("Target: " + str(target_pos))
            if event.ui_element == autotune_button:
                autotune_flag = True

                # Print the best PID gains
                # print("Best PID gains:")
                # print("Position control:")
                # print("kp_pos:", best_kp_pos)
                # print("ki_pos:", best_ki_pos)
                # print("kd_pos:", best_kd_pos)
                # print("Velocity control:")
                # print("kp_vel:", best_kp_vel)
                # print("ki_vel:", best_ki_vel)
                # print("kd_vel:", best_kd_vel)

        manager.process_events(event)

    # Get the state of the drone
    state = environment.drone.get_state()
    # Call the controller function
    action = check_action(controller_unstable.controller(state, target_pos, 1 / 60, best_kp_pos, best_ki_pos, best_kd_pos, best_kp_vel, best_ki_vel, best_kd_vel))

    environment.step(action)

    manager.update(time_delta)
    environment.render(manager, target_pos)
    
    if autotune_flag:
        run_autotune()
        autotune_flag = False
