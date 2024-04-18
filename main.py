from Lab4_3D_automatic import automatic_play_3D
from Lab4_3D_manual import manual_play_3D
from Lab4_2D_manual import manual_play_2D
from Lab4_2D_automatic import automatic_play_2D

import sys

while True:

    option = input("""Choose option to play: \n
        1. manual_play_2D
        2. manual_play_3D
        3. automatic_play_2D
        4. automatic_play_3D
        5. exit program\n
        Enter what you want to do (1 - 5): """)

    if option == '1':
        manual_play_2D()
    elif option == '2':
        manual_play_3D()
    elif option == '3':
        automatic_play_2D()
    elif option == '4':
        automatic_play_3D()
    elif option == '5':
        sys.exit(0)
    else:
        print("Unknown input, choose only from range 1-5")
