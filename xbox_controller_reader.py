import sys
import time

try:
    from XboxController import XboxControllerReader
except ImportError as exc:
    print(
        "Missing dependency while importing XboxControllerReader.\n"
        "Install it with: python -m pip install inputs"
    )
    print(str(exc))
    sys.exit(1)


def main() -> None:
    print("Reading Xbox controller values. Press Ctrl+C to stop.")
    print(
        "Expected mapping: gas=RT (ABS_RZ), brake=LT (ABS_Z), "
        "steer=LS-X (ABS_X), A=BTN_SOUTH, B=BTN_EAST"
    )

    last_status = None
    with XboxControllerReader() as controller:
        try:
            while True:
                state = controller.snapshot()
                status = (
                    round(state.gas, 3),
                    round(state.brake, 3),
                    round(state.steer, 3),
                    int(state.button_a),
                    int(state.button_b),
                )
                if status != last_status:
                    print(
                        f"\rgas={status[0]:>5.3f}  brake={status[1]:>5.3f}  "
                        f"steer={status[2]:>6.3f}  A={status[3]}  B={status[4]}    ",
                        end="",
                        flush=True,
                    )
                    last_status = status
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
