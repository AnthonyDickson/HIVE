from video2mesh.interface import Interface
from video2mesh.pipeline import main
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Video2Mash")
    parser.add_argument("--dataset_path")
    args, leftovers = parser.parse_known_args()

    if args.dataset_path is not None:
        main()
    else:
        interface = Interface.get_interface()
        interface.launch(server_name="0.0.0.0", server_port=8081)