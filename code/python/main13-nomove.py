from main13_common import run_main13


EXPORT_ANIMATION = False
EXPORT_PATH = "main13_nomove_high_level.gif"


def main():
    run_main13(
        enable_ego_control=False,
        export_animation_enabled=EXPORT_ANIMATION,
        export_path=EXPORT_PATH,
    )


if __name__ == "__main__":
    main()
