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

    size = int(input('Enter the size: (default 10) ').strip() or "10")

    items = int(input('Enter number of items: (default 10) ').strip() or "10")

    if option == '1':
        manual_play_2D(size,items)
    elif option == '2':
        manual_play_3D(size,items)
    elif option == '3':
        automatic_play_2D(size,items)
    elif option == '4':
        automatic_play_3D(size,items)
    elif option == '5':
        sys.exit(0)
    else:
        print("Unknown input, choose only from range 1-5")
