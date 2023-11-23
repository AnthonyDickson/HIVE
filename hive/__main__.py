#  HIVE, creates 3D mesh videos.
#  Copyright (C) 2023 Anthony Dickson anthony.dickson9656@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse

from hive.interface import Interface
from hive.pipeline import main

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Video2Mash")
    parser.add_argument("--dataset_path")
    args, leftovers = parser.parse_known_args()

    if args.dataset_path is not None:
        main()
    else:
        interface = Interface.get_interface()
        interface.launch(server_name="0.0.0.0", server_port=8081)
