import numpy as np
import pyvista as pv
import glob
import random
from multiprocessing import cpu_count, Pool
import itertools
import gc

if __name__ == "__main__":
    #generate_data("in/images/", "in/letter/", "out/", 1000)
    path_in_backgrounds, path_in_receipts, path_out, n, smooth = "data/backgrounds/images/", "data/in/", "data/out/", 5000, None
    textures = glob.glob(f"{path_in_backgrounds}*/*.jpg")
    receipts = glob.glob(f"{path_in_receipts}*.jpg")
    id_iter = itertools.count()
    s = []


    def create_image(smoothing):
        x = np.arange(10, dtype=float)
        xx, yy, zz = np.meshgrid(x, x, [0])
        points = np.column_stack((xx.ravel(order="F"),
                                  yy.ravel(order="F"),
                                  zz.ravel(order="F")))
        del x, xx, yy, zz
        # Perturb the points
        points[:, 0] += np.random.rand(len(points)) * random.uniform(0.2, 0.7)
        points[:, 1] += np.random.rand(len(points))
        points[:, 2] += np.random.rand(len(points)) * random.uniform(1, 2.5)
        # Create the point cloud mesh to triangulate from the coordinates
        cloud = pv.PolyData(points)
        del points
        name = random.choice(receipts)
        tex = pv.read_texture(name)
        name = name.split("/")[-1]
        name = name.rstrip(".jpg")
        surf = cloud.delaunay_2d()
        del cloud
        surf = surf.subdivide(1, 'linear')
        surf = surf.smooth(n_iter=smoothing)
        surf.texture_map_to_plane(inplace=True, use_bounds=True)

        plotter = pv.Plotter(lighting='none', window_size=(1000, 1000), off_screen=True)
        plotter.add_mesh(surf, texture=tex, smooth_shading=True, ambient=random.uniform(0.5, 0.8),
                         diffuse=random.uniform(0.5, 0.8))
        del surf
        light = pv.Light(light_type='scenelight')
        light.set_direction_angle(random.randint(25, 80), random.randint(-100, 100))
        plotter.add_light(light)
        del light

        plotter.camera_position = 'xy'
        plotter.camera.roll += random.randint(-5, 5)
        plotter.camera.azimuth += random.randint(-10, 10)
        plotter.camera.elevation += random.randint(-10, 10)
        plotter.camera.zoom(1.4)
        plotter.add_background_image(random.choice(textures))
        plotter.screenshot(f'{path_out}{name}_{next(id_iter)}.png')
        del plotter
        del name


    size = 100
    for i in range(0, n, size):
        print(i)
        pool = Pool(cpu_count(), maxtasksperchild=10)
        if smooth:
            buf = [smooth for j in range(size)]
        else:
            buf = [random.randint(1, 150) for j in range(size)]
        pool.map_async(create_image, buf)
        pool.close()
        pool.join()
        gc.collect()
