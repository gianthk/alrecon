import napari
from os import system
def napariView(image):
    viewer = napari.view_image(image)

def imagejView(ImageJ_exe_stack, image_dir, image_name=None):
    if image_name is not None:
        # system(ImageJ_exe_stack + image_dir + '/{:04.2f}'.format(ar.COR_range.value[0]) + '.tiff &')
        system(ImageJ_exe_stack + image_dir + image_name + '.tiff &')
    else:
        system(ImageJ_exe_stack + image_dir + '/slice.tiff &')
