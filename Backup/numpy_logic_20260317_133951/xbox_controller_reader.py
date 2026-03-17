import sys
import time


try:
    import inputs
except ImportError:
    print(
        "Missing dependency: inputs\n"
        "Install it with: python -m pip install inputs"
    )
    sys.exit(1)


def main() -> None:
    gas = 0.0
    brake = 0.0
    steer = 0.0
    button_a = 0
    button_b = 0

    print("Reading Xbox controller values. Press Ctrl+C to stop.")
    print(
        "Expected mapping: gas=RT (ABS_RZ), brake=LT (ABS_Z), "
        "steer=LS-X (ABS_X), A=BTN_SOUTH, B=BTN_EAST"
    )

    def print_status() -> None:
        print(
            f"\rgas={gas:>5.3f}  brake={brake:>5.3f}  "
            f"steer={steer:>6.3f}  A={button_a}  B={button_b}    ",
            end="",
            flush=True,
        )

    try:
        while True:
            events = inputs.get_gamepad()
            changed = False

            for event in events:
                if event.ev_type == "Absolute":
                    if event.code == "ABS_RZ":
                        gas = round(event.state / 255.0, 3)
                        changed = True
                    elif event.code == "ABS_Z":
                        brake = round(event.state / 255.0, 3)
                        changed = True
                    elif event.code == "ABS_X":
                        steer = event.state / 32768.0
                        if abs(steer) < 0.1:
                            steer = 0.0
                        steer = round(max(-1.0, min(1.0, steer)), 3)
                        changed = True
                elif event.ev_type == "Key":
                    if event.code == "BTN_SOUTH":
                        button_a = int(event.state)
                        changed = True
                    elif event.code == "BTN_EAST":
                        button_b = int(event.state)
                        changed = True

            if changed:
                print_status()

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
