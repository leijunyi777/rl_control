from main13_common import run_main13


EXPORT_ANIMATION = True
EXPORT_PATH = "main13_move_high_level.gif"


def main():
    run_main13(
        enable_ego_control=True,
        export_animation_enabled=EXPORT_ANIMATION,
        export_path=EXPORT_PATH,
    )


if __name__ == "__main__":
    main()
