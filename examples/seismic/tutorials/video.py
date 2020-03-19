import numpy as np
def animateSnaps2d(nsnaps, snapsObj):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import animation, rc
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    # fps: 20 bitrate: 16000
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=16000)

    from IPython.display import HTML

    base_matrix = np.transpose(snapsObj[0, :, :])

    def update(i):
        base_matrix = np.transpose(snapsObj[i, :, :])
        matrice.set_array(base_matrix)

    fig, ax = plt.subplots()
    matrice = ax.matshow(base_matrix)
    plt.colorbar(matrice)
    
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Modelling one shot over a 2-layer velocity model with Devito.')    
    
    # A file named `snapshotting.mp4` is saved in the current directory.
    ani = animation.FuncAnimation(fig, update, frames=nsnaps, interval=500)
    plt.show()

    HTML(ani.to_html5_video())
    ani._repr_html_() is None
    rc('animation', html='html5')
    ani
    ani.save('snapshotting.mp4', writer=writer)

    
filename = "naivsnaps.bin"
nsnaps = 100
fobj = open(filename, "rb")
snapsObj = np.fromfile(fobj, dtype=np.float32)
nx = 201
nz = 201
vnx = nx+20 
vnz = nz+20
snapsObj = np.reshape(snapsObj, (nsnaps, vnx, vnz))
fobj.close()

anim = animateSnaps2d(nsnaps, snapsObj)
